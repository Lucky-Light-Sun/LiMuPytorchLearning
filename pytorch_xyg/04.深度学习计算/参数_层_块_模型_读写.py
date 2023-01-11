import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # 
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


def model_construction():
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    X = torch.rand(2, 20)
    print(net(X))


def main():
    model_construction()


if __name__ == '__main__':
    main()
