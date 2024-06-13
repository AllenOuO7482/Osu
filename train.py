import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import keyboard
import time
import copy
import threading
import pydirectinput as pyd
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from IPython.display import clear_output

import replays as r
from env import OsuEnv
from models import Actor, Critic

def detect_command(replays, auto_choose_song, enable_save_replay):
    while True:
        if keyboard.is_pressed('alt+q'):
            auto_choose_song.value = not auto_choose_song.value
            print('auto_choose_song:', auto_choose_song.value)
            while keyboard.is_pressed('alt+q'):
                time.sleep(0.1)
        
        elif keyboard.is_pressed('alt+s') and enable_save_replay:
            r.save(replays, enable_save_replay)
            while keyboard.is_pressed('alt+s'):
                time.sleep(0.1)

        time.sleep(0.05)

def get_batch(replays: deque, batch_size, device):
    """
    ### Get a batch of data from replay buffer.

    # Params:
        ``replay``: deque, replay buffer
        ``batch_size``: int, size of the batch
        ``device``: torch.device, device to store the tensors

    # Returns:
        ``s``: torch.tensor, state tensor with shape ``[batch_size, 4, 60, 80]``
        ``a``: torch.tensor, action tensor with shape ``[batch_size, 2]``
        ``r``: torch.tensor, reward tensor with shape ``[batch_size, 1]``
        ``s1``: torch.tensor, next state tensor with shape ``[batch_size, 4, 60, 80]``
    """

    batch = random.sample(replays, batch_size)
    
    s = torch.tensor(np.array([s for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    a = torch.tensor(np.array([a for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    r = torch.tensor(np.array([r for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    s1 = torch.tensor(np.array([s1 for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)

    return s, a, r, s1

def choose_song():
    time.sleep(7), pyd.keyDown('esc'), pyd.keyUp('esc'), print('key esc down')
    time.sleep(1), pyd.click(x=494, y=808), print('click at 494, 808')
    time.sleep(5), pyd.keyDown('enter'), pyd.keyUp('enter'), print('key enter down')
    time.sleep(1), pyd.keyDown('space'), pyd.keyUp('space'), print('key space down')

def target_update(model_main, model_target, tau):
    """
    ### update the target network using soft update

    # Params:
        ``model_main``: nn.Module, main model
        ``model_target``: nn.Module, target model
        ``tau``: float, interpolation parameter
    """
    for w in model_target.state_dict().keys():
        eval('model_target.'+w+'.data.mul_((1-tau))')
        eval('model_target.'+w+'.data.add_(tau*model_main.'+w+'.data)')

def train_agent(episodes: int, time_steps: int, buffer: int, 
                batch_size: int, gamma: float, tau: float, sigma: float):
    """
    ### train the agent using DDPG algorithm

    # Params:
        ``episodes``: number of episodes to train
        ``time_steps``: number of time steps per episode

        ``buffer``: size of replay buffer
        ``replays``: replay buffer
        ``batch_size``: size of the batch
        ``gamma``: discount factor
        ``tau``: interpolation parameter
        ``sigma``: noise standard deviation
    """

    for i in range(epoch, episodes):
        s = env.reset() # (4, 60, 80)
        s = torch.tensor(s, dtype=torch.float32).to(device)

        episode_reward = 0
        done = False
        while len(env.np_playing) <= 3:
            # not in game
            time.sleep(0.1)
        
        now = time.time()
        frame_count = 0
        while not env.game_over:
            # in game and not pausing
            while env.stop_mouse or env.is_breaktime:
                # pausing or break time
                time.sleep(0.05)
                if auto_choose_song.value and env.stop_mouse and len(env.np_playing) <= 1:
                    choose_song()

            # collect experience 
            a = A_main(s).detach()

            a0 = np.clip(np.random.normal(a.cpu().numpy(), sigma), act_low, act_high)
            try: a0 = a0.squeeze(0) # remove batch dimension
            except: pass

            s1, r, done, _ = env.step(a0)
            env.render()

            s1 = torch.tensor(s1, dtype=torch.float32).to(device)
            if (r == 0 and random.random() <= 0.05) or r != 0:
                replays.append((s.cpu().numpy(), a0, [r / 10], s1.cpu().numpy()))

                if len(replays) >= buffer * 0.25:
                    # when buffer is full, start training
                    # sample a batch of data from replay buffer
                    s_bat, a0_bat, r_bat, s1_bat = get_batch(replays, batch_size, device)
                    a_bat = A_main(s_bat)
                    q_bat = Q_main(s_bat, a_bat)
                    # calculate loss and update Actor
                    loss_a = -torch.mean(q_bat)
                    opt_a.zero_grad()
                    loss_a.backward()
                    opt_a.step()
                    losses_a.append(loss_a.item())
                    # update Q-function
                    y_hat = Q_main(s_bat, a0_bat) # predicted Q-value
                    with torch.no_grad():
                        a1_bat = A_target(s1_bat)
                        q1_bat = Q_target(s1_bat, a1_bat) # next state's Q-value
                        y = r_bat + gamma * q1_bat
                    # calculate loss and update Critic
                    loss_c = loss_fn(y.detach(), y_hat) # Something issue here
                    opt_c.zero_grad()
                    loss_c.backward()
                    opt_c.step()
                    losses_c.append(loss_c.item())
                    # update target networks
                    target_update(A_main, A_target, tau)
                    target_update(Q_main, Q_target, tau)

                    if r < 0:
                        sigma *= 1.001 # increase noise standard deviation
                        hyperparam_dict['sigma'] = sigma
                    elif r > 0:
                        sigma *= 0.996 # anneal noise standard deviation
                        hyperparam_dict['sigma'] = sigma

                    print('sigma:', sigma)

            episode_reward += r
            
            s = s1
            elapsed_time = time.time() - now
            if elapsed_time >= 1:
                print('FPS: %.2f' % (frame_count / elapsed_time))
                now = time.time()
                frame_count = 0
            else:
                frame_count += 1

        env.game_over = False
        print('Episode:', i, ', reward: %i' % episode_reward)
        rewards.append(episode_reward)

        if i > 1 and i % 2 == 0:
            checkpoint = {
                'epochs': i+1,
                'hyperparameters': hyperparam_dict,
                'actor_state_dict': A_main.state_dict(),
                'critic_state_dict': Q_main.state_dict(),
                'optimizer_a_state_dict': opt_a.state_dict(),
                'optimizer_c_state_dict': opt_c.state_dict(),
                'losses_a': losses_a,
                'losses_c': losses_c
            }
            torch.save(checkpoint, f'{model_folder}/checkpoint_{i}.pth')

        clear_output(wait=True)

    torch.save(A_main.state_dict(), f'{model_folder}/actor_final.pth')

    plt.figure(figsize=(6, 4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available. Running on the GPU")
    else:
        raise Exception("Cuda is not available. You should run on the GPU")

    hyperparam_dict = {
        'episodes': 250, 'time_steps': 200, 'buffer': 20000,
        'batch_size': 16, 'gamma': 0.9, 'tau': 0.01, 'sigma': 0.12
    }
    # hyperparameters and initialization
    episodes = hyperparam_dict['episodes']      # episodes
    gamma = hyperparam_dict['gamma']            # discount factor
    time_steps = hyperparam_dict['time_steps']  # time steps per episode
    tau = hyperparam_dict['tau']                # target update rate
    sigma = hyperparam_dict['sigma']            # noise standard deviation
    buffer = hyperparam_dict['buffer']          # replay buffer size
    batch_size = hyperparam_dict['batch_size']  # batch size
    replays = r.load(buffer)                    # deque of replays

    # replays = deque(maxlen=buffer)
    print('replays loaded, time elapsed: %.2f seconds' % (time.time() - start_time))

    state_dim = (4, 60, 80)
    action_dim = 2
    act_low = np.array([-1, -1], dtype=np.float32)
    act_high = np.array([1, 1], dtype=np.float32)

    # define actor-critic models
    A_main = Actor(batch_size, state_dim, action_dim).to(device)
    Q_main = Critic(batch_size, state_dim, action_dim).to(device)

    losses_a = []
    losses_c = []

    opt_a = optim.Adam(A_main.parameters(), lr=1e-5) # Actor optimizer
    opt_c = optim.Adam(Q_main.parameters(), lr=1e-4) # Critic optimizer
    loss_fn = nn.MSELoss() # Critic loss function

    rewards = []
    model_folder = Path(__file__).parent / 'saved_models'
    while True:
        _ = input("load a saved model? (y/n): ")
        if _ == 'y' or _ == 'Y':
            is_load = True
            model_name = input("Enter the name of the saved model: ")
            break
        elif _ == 'n' or _ == 'N':
            is_load = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    
    if is_load:
        checkpoint = torch.load(f'{model_folder}/{model_name}.pth')
        epoch = checkpoint['epochs']
        hyperparam_dict = checkpoint['hyperparameters']
        A_main.load_state_dict(checkpoint['actor_state_dict'])
        Q_main.load_state_dict(checkpoint['critic_state_dict'])
        opt_a.load_state_dict(checkpoint['optimizer_a_state_dict'])
        opt_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
        losses_a = checkpoint['losses_a']
        losses_c = checkpoint['losses_c']
    else:
        epoch = 0

    A_target = copy.deepcopy(A_main).to(device)
    A_target.load_state_dict(A_main.state_dict())
    Q_target = copy.deepcopy(Q_main).to(device)
    Q_target.load_state_dict(Q_main.state_dict())

    print('model initialization done, time elapsed: %.2f seconds' % (time.time() - start_time))
    
    raw_img_queue = mp.Queue(maxsize=3)
    env = OsuEnv(raw_img_queue)
    num_processes = 3
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=env._get_screen, daemon=True)
        p.start()
        processes.append(p)
    
    auto_choose_song = mp.Value('b', False)
    enable_save_replay = mp.Value('b', True)
    dc_process = threading.Thread(target=detect_command, args=(replays, auto_choose_song, enable_save_replay), daemon=True)
    dc_process.start()

    print('all initialization done, time elapsed: %.2f seconds' % (time.time() - start_time))
 
    # train agent with replays
    train_agent(episodes, time_steps, buffer, batch_size, gamma, tau, sigma)