import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
import d2l.torch as d2l

import numpy as np
import time
import matplotlib.pyplot as plt


def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    # 注意 scale 是先 cols, 然后才是 rows
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()


def get_mnist_iter(batch_size):
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True
    )
    valid_dataset = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True
    )
    
    test = False
    if test is True:
        print(len(train_dataset), len(valid_dataset), train_dataset[0][0].shape)
        X, y = next(iter(data.DataLoader(
            train_dataset, batch_size=18, shuffle=False
        )))
        show_images(
            X.reshape(18, 28, 28), num_rows=2, num_cols=9, 
            titles=get_fashion_mnist_labels(y)
        )
    
    return (
        data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4),
        data.DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=4)
    )


def cross_entropy(y_hat, y):
    y = y.reshape(-1)
    return -torch.log(y_hat[torch.arange(len(y)), y])


def softmax(X):
    X_exp = torch.exp(X)
    return X_exp  / X_exp.sum(dim=1, keepdim=True)


def linear(X, W, b):
    return softmax(torch.mm(X.reshape(len(X), -1), W) + b)


def accuracy(y_hat, y):
    if y_hat.dim() > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum()) # 这里避免为 torch.int 是有用的，因为 torch.int 除法不会变为 torch.float


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, nn.Module):
        net.eval()
    
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics


def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 28 * 28
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros((1, num_outputs), requires_grad=True)

    lr = 0.1
    updater = lambda batch_size: d2l.sgd([W, b], lr, batch_size)
    
    num_epochs = 10
    net = lambda X: linear(X, W, b)
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


if __name__ == '__main__':
    main()
