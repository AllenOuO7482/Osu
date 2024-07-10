import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
    # for i in range(x.shape[0]):
    #     # 1 0.85 0.7 0.55
    #     x[i].mul_(1.0 - 0.15 * i)
    return x

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(s_dim[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, a_dim)
        # self.conv1 = nn.Conv2d(s_dim[0], 16, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2]) # flattens the input
        # self.h1 = nn.Linear(to_linear, 512)
        # self.fc = nn.Linear(512, a_dim)

        self.apply(init_weights_he)
        # self.fc.apply(init_weight_xavier)

    def forward(self, x: torch.Tensor, scale: list):
        if x.ndim == 3: 
            x = x.unsqueeze(0)
        x = enhance_features(x)
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        else:
            raise ValueError('Invalid input shape in actor')
        x = F.relu(self.h1(x))
        x = self.fc(x)
        if x.shape[0] == 1:
            x[0, 0].mul_(scale[0])
            x[0, 1].mul_(scale[1])
        x = torch.tanh(x)
        return x

    def _get_conv_output(self, channels, height, width):
        x: torch.Tensor = torch.zeros(1, channels, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(s_dim[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2])
        self.h1s = nn.Linear(to_linear, 32)
        self.h1a = nn.Linear(a_dim, 32)
        self.h2 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, 1)

        self.apply(init_weights_he)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        if s.ndim == 3:
            s = s.unsqueeze(0)
        s = enhance_features(s)
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
        state = torch.randint(0, 255, s_dim, dtype=torch.float32)
        action = actor(state, [1.0, 1.0])
        print(action)