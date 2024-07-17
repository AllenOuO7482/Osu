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
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.conv1 = nn.Conv2d(s_dim[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, a_dim)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = enhance_features(x)
        x = x / 255.0
        x = torch.tanh(self.resnet50(x).mul(5.0))
        return x

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.conv1 = nn.Conv2d(s_dim[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        to_linear = self._get_conv_output(s_dim[0], s_dim[1], s_dim[2])
        self.h1s = nn.Linear(to_linear, 64)
        self.h1a = nn.Linear(a_dim, 64)
        self.h2 = nn.Linear(128, 64)
        self.h3 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, 1)

        self.apply(init_weights_he)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        if s.ndim == 3:
            s = s.unsqueeze(0)
        s = enhance_features(s)
        xs = s / 255.0
        xs = self.resnet50(xs)
        xs = xs.view(xs.size(0), -1)
        # else:
            # raise ValueError(f'Invalid input shape in critic, error value: {xs.shape}')
        xs = F.relu(self.h1s(xs))
        xa = F.relu(self.h1a(a))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        q_value = self.fc(x)
        return q_value
    
    def _get_conv_output(self, channels, height, width):
        x: torch.Tensor = torch.zeros(1, channels, height, width)
        x = self.resnet50(x)
        return x.numel()

if __name__ == '__main__':
    s_dim = (4, 60, 80)
    a_dim = 2
    actor = Actor(s_dim, a_dim)
    critic = Critic(s_dim, a_dim)
    print(actor)
    print(critic)

    import time
    st = time.time()
    f_count = 0
    for i in range(1000):
        # 示例输入
        state = torch.randint(0, 255, s_dim, dtype=torch.float32)
        action = actor(state)
        q = critic(state, action)
        print(action, q)
        elapsed_time = time.time() - st
        f_count += 1
        if elapsed_time >= 1:
            print('FPS: %.2f' % (f_count / elapsed_time))
            st = time.time()
            f_count = 0