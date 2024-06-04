import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import copy
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output

import replays as r
from env import OsuEnv
from models import Actor, Critic

def get_batch(replay, batch_size, device):
    """
    Get a batch of data from replay buffer.

    # Params:
        ``replay``: deque, replay buffer
        ``batch_size``: int, size of the batch
        ``device``: torch.device, device to store the tensors

    # Returns:
        ``s``: torch.tensor, state tensor with shape ``[batch_size, 1, 61, 81]``
        ``a``: torch.tensor, action tensor with shape ``[batch_size, 2]``
        ``r``: torch.tensor, reward tensor with shape ``[batch_size]``
        ``s1``: torch.tensor, next state tensor with shape ``[batch_size, 1, 61, 81]``
    """

    batch = random.sample(replay, batch_size)
    
    s = torch.tensor(np.array([s for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    a = torch.tensor(np.array([a for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    r = torch.tensor(np.array([r for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)
    s1 = torch.tensor(np.array([s1 for (s, a, r, s1) in batch]), dtype=torch.float32).to(device)

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
        if s.shape[0] != 1:
            s = np.expand_dims(s, axis=0)

        s = torch.tensor(s, dtype=torch.float32).to(device)

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
            a = A_main(s).detach()

            a0 = np.clip(np.random.normal(a.cpu().numpy(), sigma), act_low, act_high) # TODO
            a0 = a0.squeeze(0)

            s1, r, done, _ = env.step(a0)
            env.render()
            
            s1 = np.expand_dims(s1, axis=0) # (61, 81) to (1, 61, 81)
            s1 = torch.tensor(s1, dtype=torch.float32).to(device)
            replays.append((s.cpu().numpy(), a0, r / 10, s1.cpu().numpy()))

            s = s1

            if len(replays) >= buffer * 0.4:
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
    state_space = env.observation_space.shape # state space size (1, 61, 81)
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
    act_low = np.array([-1, -1], dtype=np.float32)
    act_high = np.array([1, 1], dtype=np.float32)

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

    # print(get_batch(replays, batch_size, device))
    # record game state
    # r.record_game_state()

    # train agent with replays
    train_agent(episodes, time_steps, buffer, replays, batch_size, gamma, tau, sigma)