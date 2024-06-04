import cv2
from collections import deque
import torch

# local imports
from env import OsuEnv
import train as t

env = OsuEnv()
# hyperparameters and initialization
episodes = 10 # episodes
time_steps = 200  # time steps per episode
buffer = 50000 # replay buffer size
replays = deque(maxlen=buffer) # replay buffer
batch_size = 32 # batch size
gamma = 0.9  # discount factor
tau = 0.01 # target update rate
sigma = 3 # noise standard deviation

if __name__ == '__main__':
    # detect if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda is available. Running on the GPU")
    else:
        raise Exception("Cuda is not available. You should run on the GPU")
    
    # load replays from file
    replays = t.load_replays()

    # train agent with replays
    t.train_agent(episodes, time_steps, buffer, replays, batch_size, gamma, tau, sigma)
    
    # record_game_state()
    # env = OsuEnv()
    # now = datetime.now() 
    # elapsed_time = 0
    # frame_count = 0
    # env.display = True

    # done = False
    # keyboard.add_hotkey('alt+q', env._toggle_mouse_movement, timeout=1.5)
    # cnn = ScreenModel(state_dim[0], state_dim[1], state_dim[2]).to(device)

    # while not done:
    #     if not env.stop_mouse:
    #         action = env.action_space.sample()  # 隨機動作
    #         state, reward, done, info = env.step(action)
    #         screen_np = state['screen']
    #         screen_np = np.expand_dims(screen_np, axis=0)
    #         screen_tensor = torch.tensor(screen_np, dtype=torch.float32).to(device)
    #         cnn_output = cnn(screen_tensor)
    #         env.render()

    #         if env.display:
    #             # 計算fps
    #             frame_count += 1
    #             elapsed_time = (datetime.now() - now).total_seconds()
    #             if elapsed_time >= 1:
    #                 print("FPS: ", frame_count / elapsed_time)
    #                 frame_count = 0
    #                 now = datetime.now()
    #     else:
    #         time.sleep(0.1)

    cv2.destroyAllWindows()
