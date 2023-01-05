import torch
import d2l.torch as d2l
import random
import math


def synthesis_data(w, b, num_samples):
    assert w.shape[-1] == b.shape[-1]
    X = torch.normal(0, 1, (num_samples, len(w)))
    y = torch.mm(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, b.shape[-1])


def data_iter(batch_size, features, labels):
    num_samples = len(features)
    indices = torch.arange(num_samples)
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size): # 注意这个step的应用，不然很麻烦
        j = min(num_samples, i + batch_size)
        batch_indices = indices[i:j]
        yield features[batch_indices], labels[batch_indices]


def linear(w, b, X):
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    y_hat = y_hat.reshape(y.shape)
    return (y_hat - y) ** 2 / 2 / len(y)    # 除以 2, 除以 batch_size


def sgd(params, lr):
    with torch.no_grad():   # 一定要注意这个操作
        for param in params:
            param -= param.grad * lr
            param.grad.zero_()


def main():
    # 生成数据
    num_samples = 1000
    true_w = torch.tensor([[3.0, 2.0, 1.0], [0.5, 3, 2.5]])
    true_b = torch.tensor([2.4, 3.8, 5.0])

    features, labels = synthesis_data(true_w, true_b, num_samples=num_samples)
    
    # 定义模型
    w = torch.normal(0, 1, (2, 3), requires_grad=True)
    b = torch.zeros(3)
    b.requires_grad_(True)

    # 开始训练
    num_epoches = 10
    batch_size = 10
    lr = 0.01
    net = linear
    loss = squared_loss
    for epoch in range(num_epoches):
        for X, y in data_iter(batch_size, features, labels):
            y_hat = net(w, b, X)
            l = loss(y_hat, y).sum()
            l.backward()
            sgd([w, b], lr)
        with torch.no_grad():
            print(loss(net(w, b, features), labels).sum())
    print(true_w, w, sep='\n\r')
    print(true_b, b, sep='\n\r')

if __name__ == '__main__':
    main()