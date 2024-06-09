import torch
import torch.nn as nn
import torch.nn.functional as F

class ScreenModel(nn.Module):
    def __init__(self, height, width, channels):
        # height = 61, width = 81, channels = 1
        super(ScreenModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
        
        # 計算flatten層的輸入大小
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2
        )
        self._get_conv_output(height, width, channels)
        self.apply(init_weights_he)

    def _get_conv_output(self, height, width, channels):
        # 建立假數據以便計算flatten層的輸入大小
        x = torch.zeros(1, channels, height, width).to(next(self.parameters()).device)  # 修正順序並移動到正確設備
        x = self.convs(x)
        self._to_linear = x.numel()

    def forward(self, x):
        x = x / 255.0  # 正規化
        x = x.to(torch.float32)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        # state_dim = (61, 81, 1)
        super(Actor, self).__init__()
        self.cnn = ScreenModel(state_dim[0], state_dim[1], state_dim[2])
        self.h1 = nn.Linear(self.cnn._to_linear, 128)
        self.h2 = nn.Linear(128, 64)
        self.h3 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, action_dim)
        self.apply(init_weights_he)
        self.fc.apply(init_weight_xavier)
        # self.h1.weight.data.normal_(0, 1)
        # self.h2.weight.data.normal_(0, 1)
        # self.h3.weight.data.normal_(0, 1)
        # self.fc.weight.data.normal_(0, 1)

    def forward(self, screen):
        x: torch.Tensor
        x = self.cnn(screen)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = torch.tanh(self.fc(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.cnn = ScreenModel(state_dim[0], state_dim[1], state_dim[2])
        self.h1s = nn.Linear(self.cnn._to_linear, 32)
        self.h1a = nn.Linear(action_dim, 32)
        self.h2 = nn.Linear(64, 32)
        self.fc = nn.Linear(32, 1)
        self.apply(init_weights_he)
        # self.h1s.weight.data.normal_(0, 1)
        # self.h1a.weight.data.normal_(0, 1)
        # self.h2.weight.data.normal_(0, 1)
        # self.fc.weight.data.normal_(0, 1)

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
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weight_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    state_dim = (61, 81, 1)
    action_dim = 2
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    print(actor)
    print(critic)