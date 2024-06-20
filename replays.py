"""
used to process replays function
"""

import numpy as np
import os
import multiprocessing as mp
from collections import deque
import random
import time
from pathlib import Path
import pydirectinput as pyd

from env import OsuEnv

def record_game_state(save_folder=r'C:\Users\sword\.vscode\vtb\Osu\Replays'):
    raw_img_queue = mp.Queue(maxsize=3)
    env = OsuEnv(raw_img_queue)
    num_processes = 3
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=env._get_screen, daemon=True)
        p.start()
        processes.append(p)

    s = env.reset() # (2, 60, 80)
    print('initialization done')
    replay_count = 4261
    while replay_count < 5000:
        while not env.sd['status'] == 'Playing' or env.stop_mouse or env.is_breaktime:
            # waiting for game start, or pausing, or break time
            time.sleep(0.1)

        # save last screen array, action, reward, now screen array
        x, y = pyd.position()
        a = np.array([0, 0], dtype=np.float32)
        field = (230, 40, 1080, 810)

        a[0] = (x - field[0]) / (field[2] / 2) - 1
        a[1] = (y - field[1]) / (field[3] / 2) - 1
        r = env._calc_score()

        if (r == 0 and random.random() < 0.1) or r != 0:
            s1 = env._process_frame() # (2, 60, 80)

            batch = {'s': s, 'a': a, 'r': [r / 10], 's1': s1} # s, a, r, s1
            np.savez_compressed(os.path.join(save_folder, f'{replay_count+1}.npz') , batch)
            replay_count += 1
            s = s1

        time.sleep(1/30)

def record_2_frames(save_folder=r'C:\Users\sword\.vscode\vtb\Osu\Replays'):
    raw_img_queue = mp.Queue(maxsize=3)
    env = OsuEnv(raw_img_queue)
    num_processes = 3
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=env._get_screen, daemon=True)
        p.start()
        processes.append(p)

    s = env.reset() # (4, 60, 80)
    print('initialization done')
    replay_count = 0
    while replay_count < 5000:
        while not env.in_game or env.stop_mouse or env.is_breaktime:
            # waiting for game start, or pausing, or break time
            time.sleep(0.1)

        # save last screen array, action, reward, now screen array
        x, y = pyd.position()
        a = np.array([0, 0], dtype=np.float32)
        field = (230, 40, 1080, 810)

        a[0] = (x - field[0]) / (field[2] / 2) - 1
        a[1] = (y - field[1]) / (field[3] / 2) - 1
        r = env._calc_score()

        if (r == 0 and random.random() < 0.1) or r != 0:
            new_s = env._process_frame() # (60, 80)
            new_s = np.expand_dims(new_s, axis=0) # (1, 60, 80)
            s1 = np.delete(s, 0, axis=0) # (3, 60, 80)
            s1 = np.concatenate((s1, new_s), axis=0) # (4, 60, 80)

            batch = {'s': s, 'a': a, 'r': [r / 10], 's1': s1} # s, a, r, s1
            np.savez_compressed(os.path.join(save_folder, f'{replay_count+1}.npz') , batch)
            replay_count += 1   
            s = s1

        time.sleep(1/30)

def load_worker(q, q_lock, r_lock, replays):
    while True:
        try:
            with q_lock:
                if not q.empty():
                    file_path = q.get()
                else:
                    return # no more replays to load

            with np.load(file_path, allow_pickle=True) as replay:
                replay_data = replay['arr_0'].item()
                s = replay_data['s']
                a = replay_data['a']
                r = replay_data['r']
                s1 = replay_data['s1']
                std_replay_data = (s, a, r, s1)

            with r_lock:
                replays.append(std_replay_data)

        except Exception as e:
            print(f"Error loading replay: {file_path}, Exception: {e}")

def load(buffer, folder_path = "C:/Users/sword/.vscode/vtb/Osu/Replays"):
    """
    load replays from a folder
    """
    manager = mp.Manager()
    file_queue = manager.Queue()
    _replays = manager.list(deque(maxlen=buffer))
    queue_lock = manager.Lock()
    replays_lock = manager.Lock()

    print("Loading replays from:", folder_path)
    for file in os.listdir(folder_path):
        file_queue.put(os.path.join(folder_path, file))

    processes = []
    for i in range(16):
        p = mp.Process(target=load_worker, args=(file_queue, queue_lock, replays_lock, _replays), daemon=True)
        processes.append(p)
        p.start()

    # 等待所有進程完成
    for p in processes:
        p.join()

    print("Total replays loaded:", len(_replays))
    replays = deque(list(_replays), maxlen=buffer)
    return replays # output a deque

def save_worker(q, q_lock, replays_folder):
    while True:
        try:
            with q_lock:
                if not q.empty():
                    arr = q.get()
                else:
                    return

            np.savez_compressed(f'{replays_folder}/{arr[1]}.npz', arr[0])
            if arr[1] % 500 == 0:
                print(f"Saved replay: {arr[1]}.npz")

        except Exception as e:
            print(f"Error saving replay: {arr[1]}.npz, Exception: {e}")

def save(replays: deque, enable_save_replay):
    """
    save the current replay buffer to a folder
    """
    enable_save_replay.value = False

    manager = mp.Manager() # TODO it may cause some issues
    data_queue = manager.Queue()
    queue_lock = manager.Lock()

    replays_folder = Path(__file__).parent / 'Replays'
    os.makedirs(replays_folder, exist_ok=True)

    if len(replays) > 0:
        for i in range(len(replays)):
            batch = {
                's': replays[i][0],
                'a': replays[i][1],
                'r': replays[i][2],
                's1': replays[i][3]
            }
            data_queue.put((batch, i+1))

        now = time.time()
        print("start saving replays")

        processes = []
        for i in range(16):
            p = mp.Process(target=save_worker, args=(data_queue, queue_lock, replays_folder), daemon=True)
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        print("All processes done, time elapsed: %.2f" % (time.time() - now))
        enable_save_replay.value = True
    
    else:
        print('queue is empty')

if __name__ == "__main__":
    # replays = load(buffer=20000)
    # print("Loaded replays:", len(replays), type(replays))
    record_game_state()
