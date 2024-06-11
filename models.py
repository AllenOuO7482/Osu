import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class FixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, fixed_weight=0.07):
        super(FixedConv2d, self).__init__()
        self.fixed_weight = fixed_weight
        self.weight = nn.Parameter(torch.full((out_channels, in_channels, kernel_size, kernel_size), fixed_weight), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

class ScreenModel(nn.Module):
    def __init__(self, height, width, channels):
        super(ScreenModel, self).__init__()
        self.fixed_conv1 = FixedConv2d(channels, 1, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fixed_conv2 = FixedConv2d(1, 1, kernel_size=3, stride=1)
        
        self._to_linear = None
        self.convs = nn.Sequential(
            self.fixed_conv1,
            self.pool1,
            self.fixed_conv2
        )
        self._get_conv_output(height, width, channels)

    def _get_conv_output(self, height, width, channels):
        x = torch.zeros(1, channels, height, width)
        x = self.convs(x)
        self._to_linear = x.numel()

    def forward(self, x):
        x = x / 255.0  # 正规化
        x = x.to(torch.float32)
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.cnn = ScreenModel(state_dim[0], state_dim[1], state_dim[2])
        self.h1 = nn.Linear(self.cnn._to_linear, 512)
        self.fc = nn.Linear(512, action_dim)
        self.apply(init_weights_he)
        self.fc.apply(init_weight_xavier)
        
        self.output_history = deque(maxlen=1000)
        self.ema_mean = None
        self.ema_min = None
        self.ema_max = None
        self.alpha = 0.0001  # EMA更新率
        self.epsilon = 0.8  # 最小差值阈值
        self.full = False

    def forward(self, screen):
        x = self.cnn(screen)
        x = F.relu(self.h1(x))
        x = torch.tanh(self.fc(x))
        if not self.full:
            self.update_output_history(x)
        x = self.adjust_output(x)
        return x

    def update_output_history(self, output):
        output_np = output.detach().cpu().numpy()
        self.output_history.append(output_np)
        
        if len(self.output_history) == 1:
            self.ema_mean = output_np.mean(axis=0)
            self.ema_min = output_np.min(axis=0)
            self.ema_max = output_np.max(axis=0)
        else:
            self.ema_mean = self.alpha * output_np.mean(axis=0) + (1 - self.alpha) * self.ema_mean
            self.ema_min = self.alpha * output_np.min(axis=0) + (1 - self.alpha) * self.ema_min
            self.ema_max = self.alpha * output_np.max(axis=0) + (1 - self.alpha) * self.ema_max
        
        if len(self.output_history) == self.output_history.maxlen:
            self.full = True
            print('Stop adjusting output')

    def adjust_output(self, output):
        if len(self.output_history) < 500:
            return output
        
        mean = torch.tensor(self.ema_mean, device=output.device, dtype=output.dtype)
        min_val = torch.tensor(self.ema_min, device=output.device, dtype=output.dtype)
        max_val = torch.tensor(self.ema_max, device=output.device, dtype=output.dtype)
        
        # 平移
        output = output - mean
        
        # 计算差值
        diff = max_val - min_val
        
        # 检查差值是否小于阈值
        diff = torch.where(diff < self.epsilon, self.epsilon, diff)
        
        # 缩放
        scale = 1 / (diff + 1e-6)  # 防止除以零
        output = output * scale
        
        # 限制在[-1, 1]范围内
        output = torch.clamp(output, -1, 1)
        return output

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.cnn = ScreenModel(state_dim[0], state_dim[1], state_dim[2])
        self.h1s = nn.Linear(self.cnn._to_linear, 32)
        self.h1a = nn.Linear(action_dim, 32)
        self.h2 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, 1)
        self.apply(init_weights_he)

    def forward(self, screen, action):
        xs = self.cnn(screen)
        xs = F.relu(self.h1s(xs))
        xa = F.relu(self.h1a(action))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.h2(x))
        q_value = self.fc(x)
        return q_value

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

if __name__ == '__main__':
    state_dim = (61, 81, 1)
    action_dim = 2
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    print(actor)
    print(critic)

    for i in range(1000):
        # 示例输入
        state = torch.randn(1, 1, state_dim[0], state_dim[1])
        action = actor(state)
        print(action)
