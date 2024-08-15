from gym import Env
from gym.spaces import Box
from collections import deque
import cv2
import mss
import time
import json
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
        self.temp_img_queue = deque(maxlen=40)
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

        self.stream_companion_path = 'C:/Program Files (x86)/StreamCompanion/Files'
        stream_companion = gw.getWindowsWithTitle('StreamCompanion')
        if test_mode:
            pass

        elif not stream_companion:
            raise Exception("StreamCompanion window is not found")

        self.hits_prev = {'300': 0, '100': 0, '50': 0, 'miss': 0, 'score': 0}
        self.hit_errors_prev = ['']
        self.map_data_prev = {"combo": "0", "combo_left": "0", "slider_breaks": "0", "miss": "0"}
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
        reward = self.calc_reward()
        reward = max(-1, min(1, reward))
        
        # if self.game_end and self.in_game:
        #     done = True 
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

    def calc_reward(self):
        reward = 0
        with open(os.path.join(self.stream_companion_path, 'map_data.txt'), 'r') as f:
            map_data = json.loads('{' + f.read() + '}')
            if map_data == {}:
                return 0

        with open(os.path.join(self.stream_companion_path, 'hit_errors.txt'), 'r') as f:
            hit_errors = f.read().split(',')
            if hit_errors == [''] or hit_errors == [' ']:
                return 0
            
            elif hit_errors != self.hit_errors_prev:
                # print('hit error:', hit_errors)
                pass

        # print(map_data)
        if map_data == self.map_data_prev and hit_errors == self.hit_errors_prev and map_data['miss'] == self.map_data_prev['miss'] and self.map_data_prev['slider_breaks'] == map_data['slider_breaks']:
            # no new hit
            # print('no new hit', map_data, flush=True)
            return 0
        
        elif map_data['miss'] != self.map_data_prev['miss'] or map_data['slider_breaks'] != self.map_data_prev['slider_breaks']:
            # miss or slider break
            # print('miss or slider break', map_data, flush=True)
            reward += -1

        elif map_data['combo'] > self.map_data_prev['combo'] and hit_errors == self.hit_errors_prev:
            # slider not broken, but not hitted circle
            # print('slider not broken, but not hitted circle', map_data, flush=True)
            reward += 1

        elif hit_errors != self.hit_errors_prev:    
            # hitted circle, calculate last hit's score by offset
            # print('hitted circle', map_data, flush=True)
            offset = abs(int(hit_errors[-1]))
            reward += 1 - ((offset - 20) * (offset >= 20) / 100) # +-100ms error range

        else:
            print('unknown condition', map_data, flush=True)

        self.map_data_prev = map_data
        self.hit_errors_prev = hit_errors
        return reward

    def _calc_score(self):
        """
        This function is deprecated, use _calc_reward instead
        """

        with open(os.path.join(self.stream_companion_path, 'livepp_hits.txt'), 'r') as f:
            hits = json.loads('{' + f.read() + '}')
            if hits == {}:
                return 0

        hits_delta = {
            # expect input 0 0 0 0 12345
            '300':   int(hits['300']) - int(self.hits_prev['300']),
            '100':   int(hits['100']) - int(self.hits_prev['100']),
            '50':    int(hits['50']) - int(self.hits_prev['50']),
            'miss':  int(hits['miss']) - int(self.hits_prev['miss']),
            'score': int(hits['score']) - int(self.hits_prev['score']),
        }

        reward = 0
        if hits_delta['300'] > 0 or hits_delta['100'] > 0 or hits_delta['50'] > 0 or hits_delta['miss'] > 0 and not self.game_end:
            reward += hits_delta['300'] * 2 + hits_delta['100'] * 1.5 + hits_delta['50'] * 1 + hits_delta['miss'] * (-1)

        elif hits_delta['score'] > 0:
            if hits_delta['score'] == 10:
                reward += 2 # slide reward

            elif hits_delta['score'] % 100 == 0 and hits_delta['score'] != 0:
                reward += 2 # spinner reward

        self.hits_prev = hits

        return float(reward)

    def _detect_game_state(self):
        T = 0
        while True:
            with open(os.path.join(self.stream_companion_path, 'song_completion.txt'), 'r') as f:
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
        s, r, _, _ = env.step(env.action_space.sample())
        if r != 0:
            print(s.shape, r)
        # env.render()
        time.sleep(0.05)

    cv2.destroyAllWindows()