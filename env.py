from gym import Env
from gym.spaces import Box, Discrete, Dict
import numpy as np
import cv2
import mss
import pydirectinput as pyd
import time
import os
from collections import deque
import pygetwindow as gw
import threading
from multiprocessing import Pool, Queue

class OsuEnv(Env):
    def __init__(self, test_mode=False):
        super(OsuEnv, self).__init__()
        self.action_range = (310, 70, 1610, 1045) # (x_min, y_min, x_max, y_max)
        self.action_space = Box(
            low=np.array([self.action_range[0], self.action_range[1]]), 
            high=np.array([self.action_range[2], self.action_range[3]]), 
            shape=(2,)
        )
        
        self.screen_shape = (61, 81)
        screen_space = Box(low=0, high=255, shape=self.screen_shape, dtype=np.float32)
        
        mouse_button_space = Discrete(2)
        
        self.observation_space = Dict({
            'screen': screen_space,
            'm_pos': self.action_space,
            'key_pressed': mouse_button_space
        })
        
        self.state = {
            'screen': np.zeros(self.screen_shape, dtype=np.float32),
            'm_pos': np.array([self.action_range[0], self.action_range[1]], dtype=np.float32),
            'key_pressed': 0
        }
        
        self.reset_pos = ((self.action_range[0] + self.action_range[2]) // 2, (self.action_range[1] + self.action_range[3]) // 2)

        self.raw_img_queue = deque(maxlen=2)

        self.empty_frame = np.zeros(self.screen_shape, dtype=np.float32)
        self.img_prev = np.zeros(self.screen_shape, dtype=np.float32)
        self.display = True
        self.game_end = True
        self.stop_mouse = True
        self.in_game = False
        self.is_breaktime = False
        self.sd = {'completion': float('-inf'), 'hp': 0}

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

        num_threads = 2
        for i in range(num_threads):
            p = threading.Thread(target=self._get_screen, daemon=True)
            p.start()
            for i in range(10000):
                s = ''

        self.hits_prev = (0, 0, 0, 0)
        self.np_playing = ''
        self.song_completion_prev = float('-inf')
        p1 = threading.Thread(target=self._detect_game_state, daemon=True)
        p1.start()

    def step(self, action: np.ndarray):
        if action.ndim != 1:
            action = action.flatten()

        # screen: (250, 55, 250+1040, 55+780)
        # action: (x, y)
        x, y = action
        x, y = round(x), round(y)
        # m_pos = np.clip(m_pos + action, np.array([0, 0]), np.array([60, 80]))
        # x, y = round(250 + 1040 * (m_pos[1] / 80)), round(55 + 780 * (m_pos[0] / 60))

        print(x, y)
        pyd.moveTo(x, y, _pause=False)

        self.state['screen'] = self._process_frame()
        self.state['m_pos'] = action
        self.state['key_pressed'] = 0
        
        reward = self._calc_score()
        
        if self.game_end and self.in_game:
            done = True # TODO: add game end detection
        else:
            done = False
        
        info = {}

        return self.state, reward, done, info

    def reset(self):
        # reset environment
        self.state = {
            'screen': self.empty_frame,
            'm_pos': np.array([self.reset_pos[0], self.reset_pos[1]], dtype=np.float32),
            'key_pressed': 0
        }

        self.hits_prev = (0, 0, 0, 0)
        self.song_completion_prev = float('-inf')
        self.raw_img_queue.clear()
        
        return self.state

    def render(self):
        if self.display:
            self._update_opencv_window()
        
    def _get_screen(self):
        bbox = (310, 70, 1610, 1045)
        while True:
            if not self.stop_mouse and not self.is_breaktime and self.sd['completion'] >= -0.1 and self.sd['completion'] < 100:
                with mss.mss() as sct:
                    # 截圖並轉換大小為 80x60
                    img = sct.grab(bbox)
                    self.raw_img_queue.append(img)
            else:
                self.raw_img_queue.clear()
                time.sleep(1/30)
    
    def _process_frame(self):
        if len(self.raw_img_queue) > 0:
            img = self.raw_img_queue.popleft()
            img = np.array(img)
            img = cv2.resize(img, (self.screen_shape[1], self.screen_shape[0]), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img)
            self.img_prev = img
            return img
        else:
            return self.img_prev

    def _update_opencv_window(self):
        try:
            screen_np = np.repeat(np.repeat(self.state['screen'], 4, axis=0), 4, axis=1)
            cv2.imshow('Osu', screen_np)
            cv2.waitKey(1)
        except:
            print("update screen failed")
    
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
        if score_delta[0] > 0 or score_delta[1] > 0 or score_delta[2] > 0 or score_delta[3] > 0 or score_delta[-1] > 0:
            self.hits_prev = hits_count
            if score_delta[-1] == 10:
                slide_score = 2000
            else:
                slide_score = 0

            reward += score_delta[0] * 2000 + score_delta[1] * 1000 + score_delta[2] * 500 + score_delta[3] * (-100) + slide_score
            # TODO Add combo
        # if np.array_equal(self.state['mouse_position'], np.array([0, 0], dtype=np.float16)):
        #     reward -= 1000
        # if np.array_equal(self.state['mouse_position'], np.array([0, 80], dtype=np.float16)):
        #     reward -= 1000
        # if np.array_equal(self.state['mouse_position'], np.array([60, 0], dtype=np.float16)):
        #     reward -= 1000
        # if np.array_equal(self.state['mouse_position'], np.array([60, 80], dtype=np.float16)):
        #     reward -= 1000

        return reward

    def _detect_game_state(self):
        stream_companion_path = 'C:/Program Files (x86)/StreamCompanion/Files'
        T = 0
        while True:
            with open(os.path.join(stream_companion_path, 'np_playing_DL.txt'), 'r') as f:
                self.np_playing = f.read()
                self.in_game = True if len(self.np_playing) > 3 else False
            
            if self.in_game: # is playing a song?
                with open(os.path.join(stream_companion_path, 'song_completion.txt'), 'r') as f:
                    file = f.read()
                    if len(file) <= 1:
                        time.sleep(0.2)
                        continue
                    else:
                        s = [float(i) for i in file.split()]
                        self.sd = {'completion': s[0], 'hp': s[1], 'is_breaktime': s[2]}
                
                if self.sd['is_breaktime']:
                    self.is_breaktime = True
                else:
                    self.is_breaktime = False

                if self.sd['completion'] < self.song_completion_prev and self.in_game:
                    # reset time
                    self.state['mouse_position'] = np.array([self.reset_pos[0], self.reset_pos[1]], dtype=np.float16)
                    self.song_completion_prev = float('-inf')
                                
                elif self.sd['completion'] >= 100 and self.in_game:
                    # Completed
                    self.game_end = True
                    self.stop_mouse = True
                    self.hits_prev = (0, 0, 0, 0)
                    self.song_completion_prev = float('-inf')
                    T += 1
                    if T % 30 == 0:
                        print('Song Completed')

                elif self.sd['completion'] > self.song_completion_prev and self.in_game:
                    # Gaming
                    self.game_end = False
                    self.stop_mouse = False
                    self.song_completion_prev = self.sd['completion']
                    T += 1
                    if T % 30 == 0:
                        print('Smashing keys')

                elif self.sd['completion'] == self.song_completion_prev and self.in_game:
                    # Pausing and failed
                    self.game_end = False
                    self.stop_mouse = True
                    T += 1
                    if T % 30 == 0:
                        print('Pausing')

                else:
                    T += 1
                    if T % 30 == 0:
                        print('other conditions')
            
            else:
                self.game_end = True
                self.stop_mouse = True
                T += 1
                if T % 30 == 0:
                    print("Choosing a beatmap...")
            
            time.sleep(0.1)

if __name__ == '__main__':
    env = OsuEnv()
    time.sleep(3)
    for i in range(20):
        310, 70, 1610, 1045
        pyd.moveTo(310, 70)
        pyd.moveTo(1610, 70)
        pyd.moveTo(1610, 1045)
        pyd.moveTo(310, 1045)
    # for i in range(1000):
    #     action = env.action_space.sample()
    #     state, reward, done, info = env.step(action)
    #     env.render()
    #     time.sleep(1/60)
    
    cv2.destroyAllWindows()