import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

def init_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weight_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        # s_dim = (4, 60, 80) 4 frames, 60x80 pixels
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(s_dim[0], 16, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.conv1 = nn.Conv2d(s_dim[0], 8, kernel_size=5, stride=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)

        to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2])
        self.h1 = nn.Linear(to_linear, 512)
        self.fc = nn.Linear(512, a_dim)

        self.apply(init_weights_he)
        self.fc.apply(init_weight_xavier)
        
        # self.output_history = deque(maxlen=1000)
        # self.ema_mean = None
        # self.ema_min = None
        # self.ema_max = None
        # self.alpha = 0.01  # EMA更新率
        # self.epsilon = 0.8  # 最小差值阈值
        # self.full = False

    def forward(self, x: torch.Tensor):
        # 223 to 255 -> -1 to 255 -> 0 to 255
        # to reduce distractions, enhance feautres
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # x = torch.where(x < 223, torch.tensor(0, dtype=x.dtype, device=x.device), x)
        # mask = (x >= 223)
        # x[mask] = ((x[mask] - 223) / (255 - 223)) * 256 - 1
        # x = torch.where(x < 0, torch.tensor(0, dtype=x.dtype, device=x.device), x)
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        else:
            raise ValueError('Invalid input shape in actor')
        x = F.relu(self.h1(x))
        x = torch.tanh(self.fc(x))

        # if not self.full:
        #     self.update_output_history(x)

        # x = self.adjust_output(x)

        return x
    
    def _get_conv_output(self, channels, height, width):
        x: torch.Tensor = torch.zeros(1, channels, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()

    # def update_output_history(self, output):
    #     output_np = output.detach().cpu().numpy()
    #     self.output_history.append(output_np)
        
    #     if len(self.output_history) == 1:
    #         self.ema_mean = output_np.mean(axis=0)
    #         self.ema_min = output_np.min(axis=0)
    #         self.ema_max = output_np.max(axis=0)
    #     else:
    #         self.ema_mean = self.alpha * output_np.mean(axis=0) + (1 - self.alpha) * self.ema_mean
    #         self.ema_min = self.alpha * output_np.min(axis=0) + (1 - self.alpha) * self.ema_min
    #         self.ema_max = self.alpha * output_np.max(axis=0) + (1 - self.alpha) * self.ema_max
        
    #     self.alpha *= 0.9996  # 衰减率
    #     # if len(self.output_history) == self.output_history.maxlen:
    #     #     self.full = True
    #     #     print('Stop adjusting output')

    # def adjust_output(self, output):
    #     if len(self.output_history) < 500:
    #         return output
        
    #     mean = torch.tensor(self.ema_mean, device=output.device, dtype=output.dtype)
    #     min_val = torch.tensor(self.ema_min, device=output.device, dtype=output.dtype)
    #     max_val = torch.tensor(self.ema_max, device=output.device, dtype=output.dtype)

    #     output = output - mean # minus mean
    #     diff = max_val - min_val # calculate diff
    #     diff = torch.where(diff < self.epsilon, self.epsilon, diff) # check diff is not smaller than epsilon
        
    #     scale = 1 / (diff + 1e-6)  
    #     output = output * scale # scale to [-1, 1]
    #     output = torch.clamp(output, -1, 1)
    #     return output

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(s_dim[0], 16, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.conv1 = nn.Conv2d(s_dim[0], 8, kernel_size=5, stride=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)

        to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2])
        self.h1s = nn.Linear(to_linear, 32)
        self.h1a = nn.Linear(a_dim, 32)
        self.h2 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, 1)

        self.apply(init_weights_he)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        # 223 to 255 -> -1 to 255 -> 0 to 255
        if s.ndim == 3:
            s = s.unsqueeze(0)
        # s = torch.where(s < 223, torch.tensor(0, dtype=s.dtype, device=s.device), s)
        # mask = (s >= 223)
        # s[mask] = ((s[mask] - 223) / (255 - 223)) * 256 - 1
        # s = torch.where(s < 0, torch.tensor(0, dtype=s.dtype, device=s.device), s)
        xs = s / 255.0
        xs = F.relu(self.conv1(xs))
        xs = F.relu(self.conv2(xs))
        xs = F.relu(self.conv3(xs))
        if xs.ndim == 4:
            xs = xs.view(xs.size(0), -1)
        else:
            raise ValueError('Invalid input shape in critic')
        xs = F.relu(self.h1s(xs))
        xa = F.relu(self.h1a(a))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.h2(x))
        q_value = self.fc(x)
        return q_value
    
    def _get_conv_output(self, channels, height, width):
        x: torch.Tensor = torch.zeros(1, channels, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()

if __name__ == '__main__':
    s_dim = (4, 60, 80)
    a_dim = 2
    actor = Actor(s_dim, a_dim)
    critic = Critic(s_dim, a_dim)
    print(actor)
    print(critic)

    for i in range(1000):
        # 示例输入
        state = torch.randint(0, 255, (4, 60, 80), dtype=torch.float32)
        action = actor(state)
        print(action)
