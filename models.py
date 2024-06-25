import torch
import torch.nn as nn
import torch.nn.functional as F
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

def enhance_features(x: torch.Tensor):
    # 223 to 255 -> -1 to 255 -> 0 to 255
    x = torch.where(x < 223, torch.tensor(0, dtype=x.dtype, device=x.device), x)
    mask = (x >= 223)
    x[mask] = ((x[mask] - 223) / (255 - 223)) * 256 - 1
    x = torch.where(x < 0, torch.tensor(0, dtype=x.dtype, device=x.device), x)
    return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat([x, out], 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.dense_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_block(x)

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, growth_rate=32, num_layers=4):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(s_dim[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dense_block1 = DenseBlock(num_layers, 64, growth_rate)
        self.transition1 = nn.Conv2d(64 + num_layers * growth_rate, 128, kernel_size=1, bias=False)
        self.dense_block2 = DenseBlock(num_layers, 128, growth_rate)
        self.transition2 = nn.Conv2d(128 + num_layers * growth_rate, 256, kernel_size=1, bias=False)
        to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2])
        self.h1 = nn.Linear(to_linear, 512)
        self.fc = nn.Linear(512, a_dim)
        
        self.apply(init_weights_he)
        self.fc.apply(init_weight_xavier)
        
    def forward(self, x: torch.Tensor, scale):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # x = enhance_features(x)
        x = x / 255.0
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dense_block1(x)
        x = F.relu(self.transition1(x))
        x = self.dense_block2(x)
        x = F.relu(self.transition2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.h1(x))
        x = torch.tanh((self.fc(x) * scale))
        return x
    
    def _get_conv_output(self, channels, height, width):
        x: torch.Tensor = torch.zeros(1, channels, height, width)
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        return x.numel()

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, growth_rate=32, num_layers=4):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(s_dim[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dense_block1 = DenseBlock(num_layers, 64, growth_rate)
        self.transition1 = nn.Conv2d(64 + num_layers * growth_rate, 128, kernel_size=1, bias=False)
        self.dense_block2 = DenseBlock(num_layers, 128, growth_rate)
        self.transition2 = nn.Conv2d(128 + num_layers * growth_rate, 256, kernel_size=1, bias=False)
        to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2])
        self.h1s = nn.Linear(to_linear, 512)
        self.h1a = nn.Linear(a_dim, 512)
        self.h2 = nn.Linear(1024, 512)
        self.fc = nn.Linear(512, 1)

        self.apply(init_weights_he)
        
    def forward(self, s: torch.Tensor, a: torch.Tensor):
        if s.ndim == 3:
            s = s.unsqueeze(0)
        # s = enhance_features(s)
        s = s / 255.0
        xs = F.relu(self.bn1(self.conv1(s)))
        xs = self.dense_block1(xs)
        xs = F.relu(self.transition1(xs))
        xs = self.dense_block2(xs)
        xs = F.relu(self.transition2(xs))
        xs = xs.view(xs.size(0), -1)
        xs = F.relu(self.h1s(xs))
        xa = F.relu(self.h1a(a))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.h2(x))
        q_value = self.fc(x)
        return q_value
    
    def _get_conv_output(self, channels, height, width):
        x: torch.Tensor = torch.zeros(1, channels, height, width)
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.transition2(x)
        return x.numel()

if __name__ == '__main__':
    s_dim = (4, 60, 80)
    a_dim = 2
    actor = Actor(s_dim, a_dim)
    critic = Critic(s_dim, a_dim)
    print(actor)
    print(critic)

    for i in range(10):
        state = torch.randint(0, 255, (4, 60, 80), dtype=torch.float32)
        # state = torch.full((4, 60, 80), 225, dtype=torch.float32)
        action = actor(state, scale=0.5)
        print(action)
