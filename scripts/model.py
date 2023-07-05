import torch
import torch.nn as nn


class QNET(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class PolicyNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=5):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class DPolicyNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(DPolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class ValueNet(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=output_dim),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)



if __name__ == '__main__':
    dqn = PolicyNet()
    input = torch.tensor([[2, 1], [3, 1]])
    print(dqn)
    print(dqn(input))
