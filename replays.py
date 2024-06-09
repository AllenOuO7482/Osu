"""
used to process replays function
"""

import numpy as np
import os
import multiprocessing
from collections import deque
import random
import time
from pathlib import Path
import keyboard
import pydirectinput as pyd

from env import OsuEnv

def record_game_state():
    env = OsuEnv()
    s = env.reset()
    s_screen = np.expand_dims(s['screen'], axis=0)
    x, y = pyd.position()
    m_pos_prev = np.array([x, y], dtype=np.float32)
    key_pressed_prev = 0

    for i in range(20000):
        while not env.in_game or env.stop_mouse or env.is_breaktime:
            # waiting for game start, or pausing, or break time
            time.sleep(0.1)

        # save last screen array, action, reward, now screen array
        x, y = pyd.position()
        m_pos = np.array([x, y], dtype=np.float32)
        key_pressed = 1 if keyboard.is_pressed('z') or keyboard.is_pressed('x') else 0

        reward = env._calc_score()
        s1_screen = env._process_frame()
        if s1_screen.shape[0] != 1:
            s1_screen = np.expand_dims(s1_screen, axis=0)

        s = {'screen': s_screen, 'm_pos': m_pos_prev, 'key_pressed': key_pressed_prev}
        s1 = {'screen': s1_screen, 'm_pos': m_pos, 'key_pressed': key_pressed}

        batch = {'s': s, 'action': m_pos, 'reward': reward / 10, 's1': s1} # s, a, r, s1
        np.savez_compressed(Path(__file__).parent / 'game_state' / f'game_state_{i}.npz', batch)
        
        s_screen = s1_screen
        m_pos_prev = m_pos
        key_pressed_prev = key_pressed

        time.sleep(1/30)

def worker(q, q_lock, r_lock, replays):
    while True:
        try:
            with q_lock:
                if not q.empty():
                    file_path = q.get()
                else:
                    print("Queue empty, exiting worker")
                    return

            with np.load(file_path, allow_pickle=True) as replay:
                replay_data = replay['arr_0'].item()
                s = replay_data['s']
                action = replay_data['action']
                reward = replay_data['reward']
                s1 = replay_data['s1']
                std_replay_data = (s, action, reward, s1)

            with r_lock:
                if (reward == 0 and random.random() < 0.1) or reward != 0:
                    replays.append(std_replay_data)
        except Exception as e:
            print(f"Error loading replay: {file_path}, Exception: {e}")

def load(folder_path = "C:/Users/sword/.vscode/Osu/game_state"):
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    _replays = manager.list(deque(maxlen=20000))
    queue_lock = manager.Lock()
    replays_lock = manager.Lock()

    print("Loading replays from:", folder_path)
    for file in os.listdir(folder_path):
        file_queue.put(os.path.join(folder_path, file))

    processes = []
    for i in range(16):
        p = multiprocessing.Process(target=worker, args=(file_queue, queue_lock, replays_lock, _replays), daemon=True)
        processes.append(p)
        p.start()

    # 等待所有進程完成
    for p in processes:
        p.join()

    print("Total replays loaded:", len(_replays))
    replays = deque(_replays)
    return replays

# 使用示例
if __name__ == "__main__":
    replays = load()
    print("Loaded replays:", len(replays), type(replays))
