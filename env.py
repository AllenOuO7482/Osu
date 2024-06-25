from gym import Env
from gym.spaces import Box
from collections import deque
import cv2
import mss
import time
import os
import threading
import numpy as np
import pygetwindow as gw
import pydirectinput as pyd
import multiprocessing as mp

class OsuEnv(Env):
    def __init__(self, raw_img_queue, test_mode=False):
        super(OsuEnv, self).__init__()
        self.action_range = (310, 70, 1610, 1045) # (x_min, y_min, x_max, y_max)
        self.action_space = Box(
            low=np.array([self.action_range[0], self.action_range[1]]), 
            high=np.array([self.action_range[2], self.action_range[3]]), 
            shape=(2,)
        )
        
        self.screen_shape = (60, 80)
        screen_space = Box(low=0, high=255, shape=self.screen_shape, dtype=np.float32)
        
        self.observation_space = screen_space
        self.state_shape = (4, 60, 80)
        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.action_queue = deque(maxlen=2)
        # self.action_delay = 0.02 # delay between actions in seconds
        # self.action_data = None
        
        self.reset_pos = ((self.action_range[0] + self.action_range[2]) // 2, (self.action_range[1] + self.action_range[3]) // 2)

        self.raw_img_queue = raw_img_queue 
        self.temp_img_queue = deque(maxlen=25)
        for i in range(self.temp_img_queue.maxlen):
            zero_img = {'img': np.zeros(self.screen_shape, dtype=np.float32), 'time': time.time()}
            self.temp_img_queue.append(zero_img)

        self.empty_frame = np.zeros(self.state_shape, dtype=np.float32)
        self.img_prev = np.zeros(self.screen_shape, dtype=np.float32)
        self.display = True
        self.game_end = True
        self.game_over = False # True when game is over, else False
        self.stop_mouse = True
        self.is_breaktime = False
        self.sd = {'completion': float('-inf'), 'hp': 0, 'is_breaktime': 0, 'status': 'MainMenu'}

        osu_window = gw.getWindowsWithTitle('osu!')
        if test_mode:
            pass
        
        elif not osu_window:
            raise Exception("Osu window is not found")

        stream_companion = gw.getWindowsWithTitle('StreamCompanion')
        if test_mode:
            pass

        elif not stream_companion:
            raise Exception("StreamCompanion window is not found")

        self.hits_prev = (0, 0, 0, 0)
        self.song_completion_prev = float('-inf')
        p1 = threading.Thread(target=self._detect_game_state, daemon=True)
        p1.start()

    def step(self, action: np.ndarray):
        if action.ndim != 1:
            action = action.flatten()

        # field = (310, 70, 1610, 1045)
        # (230, 40, 1310, 850)
        field = (230, 40, 1080, 810)
        
        x = field[0] + ((action[0] + 1) * (field[2] / 2))
        y = field[1] + ((action[1] + 1) * (field[3] / 2))
        x, y = round(x), round(y)

        pyd.moveTo(x, y, _pause=False)
        # print('move mouse to', pyd.position())

        self.state = self._process_frame() # state.shape = 
        reward = self._calc_score()
        reward = max(-1, min(1, reward))
        
        # if self.game_end and self.in_game:
        #     done = True # TODO: add game end detection
        # else:
        #     done = False
        done = None
        info = {}

        return self.state, reward, done, info

    def reset(self):
        # reset environment
        self.state = self.empty_frame

        self.hits_prev = (0, 0, 0, 0)
        self.song_completion_prev = float('-inf')
        if not self.raw_img_queue.empty():
            self.raw_img_queue.get()
        
        return self.state

    def render(self):
        if self.display:
            self._update_opencv_window()
        
    def _get_screen(self):
        bbox = (310, 70, 1610, 1045)
        with mss.mss() as sct:
            while True:
                img = sct.grab(bbox)
                img_np = np.array(img)  # Convert to numpy array for faster processing
                if self.raw_img_queue.full():
                    self.raw_img_queue.get()
                self.raw_img_queue.put({'img': img_np, 'time': time.time()})
    
    def _process_frame(self):
        while self.raw_img_queue.empty():
            time.sleep(0.01)

        _img = self.raw_img_queue.get() # img = {'img': img_np, 'time': time.time()}
        _img['img'] = cv2.resize(_img['img'], (self.screen_shape[1], self.screen_shape[0]), interpolation=cv2.INTER_AREA)
        _img['img'] = cv2.cvtColor(_img['img'], cv2.COLOR_BGR2GRAY)
        self.temp_img_queue.append(_img)

        img_lst = []
        # find the closest frame to the target time
        interval = 0.1
        target_time = self.temp_img_queue[-1]['time']
        for i in range(4):
            idx = min(range(len(self.temp_img_queue)), key=lambda j: abs(self.temp_img_queue[j]['time'] - (target_time - interval * i)))
            img_lst.append(self.temp_img_queue[idx]['img'])
        
        img = np.array(img_lst, dtype=np.float32)
        self.img_prev = img
        return img
        

    def _update_opencv_window(self):
        try:
            screen_np = self.state.squeeze()
            screen_np = np.repeat(np.repeat(screen_np, 4, axis=0), 4, axis=1)
            cv2.imshow('Osu', screen_np)
            cv2.waitKey(1)
        except Exception as e:
            print("update screen failed", e)
    
    def _record_game_state(self):
        pass
    
    def _calc_score(self):
        stream_companion_path = 'C:/Program Files (x86)/StreamCompanion/Files'
        with open(os.path.join(stream_companion_path, 'livepp_hits.txt'), 'r') as f:
            file = f.read()
            if len(file) <= 1:
                return 0
            
            # expect input 0 0 0 0 12345
            hits = file.split() 
            hits_300 = int(hits[0])
            hits_100 = int(hits[1])
            hits_50 = int(hits[2])
            hits_miss = int(hits[3])
            score = int(hits[-1])
            
        hits_count = (hits_300, hits_100, hits_50, hits_miss, score)
        score_delta = (
            hits_count[0] - self.hits_prev[0], 
            hits_count[1] - self.hits_prev[1], 
            hits_count[2] - self.hits_prev[2],
            hits_count[3] - self.hits_prev[3],
            hits_count[-1] - self.hits_prev[-1]
        )

        reward = 0
        if score_delta[0] > 0 or score_delta[1] > 0 or score_delta[2] > 0 or score_delta[3] > 0 and not self.game_end:
            reward += score_delta[0] * 1 + score_delta[1] * 1 + score_delta[2] * 0.5 + score_delta[3] * (-1)

        elif score_delta[-1] > 0:
            if score_delta[-1] == 10:
                reward += 1 # slide reward

            elif score_delta[-1] % 100 == 0 and score_delta[-1] != 0:
                reward += 0.5 # spinner reward
            
        self.hits_prev = hits_count

        return float(reward)

    def _detect_game_state(self):
        stream_companion_path = 'C:/Program Files (x86)/StreamCompanion/Files'
        T = 0
        while True:
            with open(os.path.join(stream_companion_path, 'song_completion.txt'), 'r') as f:
                file = f.read()
                if len(file) < 3:
                    time.sleep(0.05)
                    continue
                else:
                    s = [i for i in file.split()]
                    self.sd = {'completion': float(s[0]), 'hp': float(s[1]), 'is_breaktime': float(s[2]), 'status': s[3]}

            if self.sd['status'] == 'Playing': # is playing a song?
                self.game_over = False

                if self.sd['is_breaktime']:
                    self.is_breaktime = True
                else:
                    self.is_breaktime = False

                if self.sd['completion'] < self.song_completion_prev:
                    # reset time
                    self.song_completion_prev = float('-inf')

                elif self.sd['completion'] == self.song_completion_prev:
                    # Pausing and failed
                    self.game_end = False
                    self.stop_mouse = True
                    T += 1
                    if T % 30 == 0:
                        print('Pausing')

                elif self.sd['completion'] > self.song_completion_prev or self.sd['status'] == 'Playing':
                    # Gaming
                    self.game_end = False
                    self.stop_mouse = False
                    self.song_completion_prev = self.sd['completion']

                    T += 1
                    if T % 30 == 0:
                        print('Smashing keys')

                else:
                    T += 1
                    if T % 30 == 0:
                        print('other conditions')

            elif self.sd['status'] == 'ResultsScreen':
                # Completed
                self.game_end = True
                self.game_over = True
                self.stop_mouse = True
                self.hits_prev = (0, 0, 0, 0)
                self.song_completion_prev = float('-inf')
                T += 1
                if T % 30 == 0:
                    print('Song Completed')

            else:
                self.game_end = True
                self.game_over = False
                self.stop_mouse = True
                T += 1
                if T % 30 == 0:
                    if self.sd['status'] == 'SongSelect':
                        print("Choosing a beatmap...")

                    elif self.sd['status'] == 'MainMenu':
                        print("In main menu")
            
            time.sleep(0.05) # TODO Harder map need less time to pausing

if __name__ == '__main__':
    raw_img_queue = mp.Queue(maxsize=3)
    env = OsuEnv(raw_img_queue)
    num_processes = 3
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=env._get_screen, daemon=True)
        p.start()
        processes.append(p)
    
    while True:
        env.state = env._process_frame()
        env.render()
        time.sleep(0.01)

    cv2.destroyAllWindows()