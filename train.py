import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import copy
import keyboard
import matplotlib.pyplot as plt
import pydirectinput as pyd
from pathlib import Path
from collections import deque
from IPython.display import clear_output

import replays as r
from env import OsuEnv
from models import Actor, Critic

def get_batch(replay, batch_size, device):
    """
    get a batch of data from replay buffer
    """
    batch = random.sample(replay, batch_size)
    # s = {'screen': np.ndarray, 'm_pos': np.ndarray, 'key_pressed': bool}
    # a = np.ndarray with shape [batch_size, 2]
    # r = float with shape [batch_size]
    # s1.shape = {'screen': np.ndarray, 'm_pos': np.ndarray, 'key_pressed': bool}
    screens, m_pos, key_pressed = [], [], []
    _s = [s for (s, a, r, s1) in batch]
    for i in range(batch_size):
        screens.append(_s[i]['screen'])
        m_pos.append(_s[i]['m_pos'])
        key_pressed.append(_s[i]['key_pressed'])

    screens = torch.tensor(np.array(screens), dtype=torch.float32).to(device)
    m_pos = torch.tensor(np.array(m_pos), dtype=torch.float32).to(device)
    key_pressed = torch.tensor(np.array(key_pressed), dtype=torch.float32).to(device)
    s = {'screen': screens,'m_pos': m_pos, 'key_pressed': key_pressed}

    a = torch.tensor(np.array([a for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    r = torch.tensor(np.array([r for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)

    screens, m_pos, key_pressed = [], [], []
    _s1 = [_s1 for (s, a, r, s1) in batch]
    for i in range(batch_size):
        screens.append(_s1[i]['screen'])
        m_pos.append(_s1[i]['m_pos'])
        key_pressed.append(_s1[i]['key_pressed'])

    screens = torch.tensor(np.array(screens), dtype=torch.float32).to(device)
    m_pos = torch.tensor(np.array(m_pos), dtype=torch.float32).to(device)
    key_pressed = torch.tensor(np.array(key_pressed), dtype=torch.float32).to(device)
    s1 = {'screen': screens,'m_pos': m_pos, 'key_pressed': key_pressed}

    # if s.shape != (batch_size, 1, 60, 80):
    #     raise Exception("s.shape should be (batch_size, 1, 60, 80)")
    # if a.shape != (batch_size, 2):
    #     raise Exception("a.shape should be (batch_size, 2)")
    # if r.ndim != 1:
    #     raise Exception("r.shape should be (batch_size)")
    # if s_.shape != (batch_size, 1, 60, 80):
    #     raise Exception("s_.shape should be (batch_size, 1, 60, 80)")
    
    return s, a, r, s1

def target_update(model_main, model_target, tau):
    """
    update the target network using soft update
    """
    for w in model_target.state_dict().keys():
        # filter out ScreenModel keys, because they are only used in processing the screen
        if w.startswith('cnn'):
            pass
        else:
            eval('model_target.'+w+'.data.mul_((1-tau))')
            eval('model_target.'+w+'.data.add_(tau*model_main.'+w+'.data)')

def train_agent(episodes: int, time_steps: int, buffer: int, replays: deque, 
                batch_size: int, gamma: float, tau: float, sigma: float):
    """
    train the agent using DDPG algorithm
    """
    env = OsuEnv()
    for i in range(episodes):
        s = env.reset()
        if s['screen'].shape[0] != 1:
            s_screen = np.expand_dims(s['screen'], axis=0)

        else:
            s_screen = s['screen']

        s_screen = torch.tensor(s_screen, dtype=torch.float32).to(device)

        if s['m_pos'].shape[0] != 1:
            m_pos = np.expand_dims(s['m_pos'], axis=0)

        else:
            m_pos = s['m_pos']

        m_pos = torch.tensor(m_pos, dtype=torch.float32).to(device)

        # print(s_screen)
        episode_reward = 0
        done = False
        while len(env.np_playing) <= 1:
            # not in game
            time.sleep(0.1)
        
        now = time.time()
        frame_count = 0
        while not done:
            # in game and not pausing
            while env.stop_mouse or env.is_breaktime:
                # pausing or break time
                time.sleep(0.1)

            # collect experience 
            a = A_main(s_screen, m_pos).detach()
            a0 = np.clip(np.random.normal(a.cpu().numpy(), sigma), act_low, act_high) # TODO
            a0 = a0.squeeze(0)

            s1, r, done, _ = env.step(a0)
            env.render()
            
            s1_screen = np.expand_dims(s1['screen'], axis=0)
            s1_screen = torch.tensor(s1_screen, dtype=torch.float32).to(device)
            replays.append((s_screen.cpu().numpy(), a0, r / 10, s1_screen.cpu().numpy()))
            s_screen = s1_screen
            if len(replays) >= buffer * 0.4:
                # when buffer is full, start training
                # sample a batch of data from replay buffer
                s_bat, a0_bat, r_bat, s1_bat = get_batch(replays, batch_size, device)
                a_bat = A_main(s_bat['screen'], s_bat['m_pos'])
                q_bat = Q_main(s_bat['screen'], s_bat['m_pos'], a_bat)
                # calculate loss and update Actor
                loss_a = -torch.mean(q_bat)
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()
                losses_a.append(loss_a.item())
                # update Q-function
                y_hat = Q_main(s_bat['screen'], s_bat['m_pos'], a0_bat) # predicted Q-value
                with torch.no_grad():
                    a1_bat = A_target(s1_bat['screen'], s1_bat['m_pos'])
                    q1_bat = Q_target(s1_bat['screen'], s1_bat['m_pos'], a1_bat) # next state's Q-value
                    y = r_bat + gamma * q1_bat
                # calculate loss and update Critic
                loss_c = loss_fn(y.detach(), y_hat)
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()
                losses_c.append(loss_c.item())
                # update target networks
                target_update(A_main, A_target, tau)
                target_update(Q_main, Q_target, tau)
                sigma *= 0.9998 # anneal noise standard deviation

            episode_reward += r

            frame_count += 1
            if time.time() - now >= 1:
                print(f'FPS: {frame_count}')
                now = time.time()
                frame_count = 0

        print('Episode:', i, ', reward: %i' % episode_reward, ', sigma: %.2f' % sigma)
        rewards.append(episode_reward)

        if i > 1 and i % 5 == 0:
            checkpoint = {
                'epochs': i,
                'actor_state_dict': A_main.state_dict(),
                'critic_state_dict': Q_main.state_dict(),
                'optimizer_a_state_dict': opt_a.state_dict(),
                'optimizer_c_state_dict': opt_c.state_dict(),
                'losses_a': losses_a,
                'losses_c': losses_c
            }
            torch.save(checkpoint, f'AI/Osu/saved_model/checkpoint_{i}.pth')

        clear_output(wait=True)

    torch.save(A_main.state_dict(), 'AI/Osu/saved_model/actor_final.pth')

    plt.figure(figsize=(6, 4))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def test_agent(model: nn.Module, time_steps=200, mode='human', display=True):
    env = OsuEnv()
    s = env.reset()
    s = torch.tensor(s, dtype=torch.float32).to(device)
    for t in range(time_steps):
        a = np.clip(model(s).data.cpu().numpy(), act_low, act_high)
        while env.stop_mouse:
            time.sleep(0.1)
        s1, r, done, info = env.step(a)
        s1 = torch.tensor(s1, dtype=torch.float32).to(device)
        if display: # TODO modify there
            print(f"t={t}\ns={s.cpu().numpy()}\na={a}\nr={r}\ns1={s1.cpu().numpy()}")
            clear_output(wait=True)
        if np.sum(np.fabs(s1.cpu().numpy()-s.cpu().numpy())) < 0.01:
            if display:
                print("Your Steps:", t+1)
            break
        s = s1
    return t+1

if __name__ == '__main__':
    # load replays from file
    replays = r.load()

    env = OsuEnv()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available. Running on the GPU")
    else:
        raise Exception("Cuda is not available. You should run on the GPU")

    # hyperparameters and initialization
    state_space = env.observation_space['screen'].shape # state space size (1, 61, 81)
    action_space = env.action_space.shape[0] # action space size
    episodes = 250 # episodes
    gamma = 0.9  # discount factor
    time_steps = 200  # time steps per episode
    tau = 0.01 # target update rate
    sigma = 3 # noise standard deviation
    buffer = 50000 # replay buffer size
    batch_size = 32 # batch size
    # replay = deque(maxlen=buffer) # replay buffer

    done = False

    state_dim = (61, 81, 1)     
    action_dim = 2
    act_low = np.array([env.action_range[0], env.action_range[1]], dtype=np.float32)
    act_high = np.array([env.action_range[2], env.action_range[3]], dtype=np.float32)

    # define actor-critic models
    A_main = Actor(state_dim, action_dim).to(device)
    A_target = copy.deepcopy(A_main).to(device)
    A_target.load_state_dict(A_main.state_dict())

    Q_main = Critic(state_dim, action_dim).to(device)
    Q_target = copy.deepcopy(Q_main).to(device)
    Q_target.load_state_dict(Q_main.state_dict())

    losses_a = []
    losses_c = []
    
    opt_a = optim.Adam(A_main.parameters(), lr=1e-4) # Actor optimizer

    loss_fn = nn.MSELoss()
    opt_c = optim.Adam(Q_main.parameters(), lr=1e-3) # Critic optimizer

    rewards = []

    # record game state
    # r.record_game_state()

    # train agent with replays
    train_agent(episodes, time_steps, buffer, replays, batch_size, gamma, tau, sigma)