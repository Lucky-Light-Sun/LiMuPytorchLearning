import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.torch as d2l

import numpy as np
import matplotlib.pyplot as plt


def relu(X):
    return torch.max(X, torch.zeros_like(X))


def scratch():
    batch_size = 16
    train_dataloder, valid_dataloader = d2l.load_data_fashion_mnist(batch_size)

    in_features = 28 * 28
    mid_features = 256
    out_features = 10
    num_epochs = 10
    lr = 0.01

    w1 = nn.Parameter(
        torch.normal(0, 1, (in_features, mid_features))
    )
    b1 = nn.Parameter(
        torch.zeros(mid_features)
    )
    w2 = nn.Parameter(
        torch.normal(0, 1, (mid_features, out_features))
    )
    b2 = nn.Parameter(
        torch.zeros(out_features)
    )
    params = [w1, b1, w2, b2]
    net = lambda X: relu(X.reshape(len(X), -1) @ w1 + b1) @ w2 + b2
    updater = torch.optim.SGD(params, lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    d2l.train_ch3(net, train_dataloder, valid_dataloader, loss, num_epochs, updater)

    d2l.predict_ch3(net, valid_dataloader)


def concise():
    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            
    batch_size = 16
    train_dataloder, valid_dataloader = d2l.load_data_fashion_mnist(batch_size)

    in_features = 28 * 28
    mid_features = 256
    out_features = 10
    num_epochs = 10
    lr = 0.01

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, mid_features),
        nn.ReLU(),
        nn.Linear(mid_features, out_features)
    )

    net.apply(init_weight)
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    d2l.train_ch3(net, train_dataloder, valid_dataloader, loss, num_epochs, updater)

    plt.ioff()
    plt.show()


def main():
    # scratch()
    # concise()
    alpha = 0.9
    cls_num = 5
    num_samples = 10
    y = torch.randint(low=0, high=cls_num, size=(num_samples, 1), dtype=torch.int64)
    ground_truth = torch.ones((num_samples, cls_num)) * ((1 - alpha) / cls_num)
    ground_truth[range(0, num_samples), y.reshape(-1)] = alpha
    y_hat = torch.normal(0, 1, (num_samples, cls_num)).softmax(dim=1)
    print(y_hat)
    print(F.cross_entropy(y_hat, ground_truth))
    print(F.cross_entropy(y_hat, y.reshape(-1), label_smoothing=1-alpha))

if __name__ == '__main__':
    main()