import torch
import torch.nn as nn

import d2l.torch as d2l
import random
import math
from torch.utils.data import DataLoader, TensorDataset


def synthesis_data(num_samples, w, b):
    assert w.shape[-1] == b.shape[-1]
    X = torch.normal(0, 1, (num_samples, len(w)))
    y = torch.mm(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, b.shape[-1])


def data_iter(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

def main():
    num_samples = 1000
    batch_size = 10
    in_features, out_features = 3, 4
    true_w = torch.randn((in_features, out_features))
    true_b = torch.randn(out_features)
    features, labels = synthesis_data(num_samples, true_w, true_b)
    dataloader = data_iter((features, labels), batch_size, True)

    net = nn.Sequential(
        nn.Linear(in_features, out_features)
    )
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    lr = 0.01
    num_epoches = 10
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss = torch.nn.MSELoss()

    for epoch in range(num_epoches):
        for X, y in dataloader:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch}, loss {l}.')
    
    print(true_w, net[0].weight.data, sep='\n\r')
    print(true_b, net[0].bias.data, sep='\n\r')


if __name__ == '__main__':
    main()
