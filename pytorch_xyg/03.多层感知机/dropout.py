import torch
import torch.nn as nn
import torch.nn.functional as F

import d2l.torch as d2l
import numpy as np
import matplotlib.pyplot as plt


def dropout_layer(X, p):
    assert 0 <= p <= 1
    if p == 0:
        return X
    elif p == 1:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) >= p).float()  # [0, 1)
    return mask * X / (1 - p)


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight.data)
        nn.init.normal_(layer.bias.data)


def scratch():
    dropout1, dropout2 = 0.2, 0.5
    class Net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                    is_training = True):
            super(Net, self).__init__()
            self.num_inputs = num_inputs
            self.training = is_training
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()

        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
            # 只有在训练模型时才使用dropout
            if self.training == True:
                # 在第一个全连接层之后添加一个dropout层
                H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
            if self.training == True:
                # 在第二个全连接层之后添加一个dropout层
                H2 = dropout_layer(H2, dropout2)
            out = self.lin3(H2)
            return out

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


def concise():
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.normal_(m.weight, std=0.01)

    dropout1, dropout2 = 0.2, 0.5
    net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

    net.apply(init_weights)
    num_epochs, lr, batch_size = 10, 0.5, 256

    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    loss = nn.CrossEntropyLoss(reduction='none')

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    

def main():
    scratch()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()

