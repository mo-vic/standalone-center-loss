import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, Pad, RandomCrop, ToTensor

from models.resnet import ResNet

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_dataset(dataset, batch_size, use_gpu, num_workers):
    if dataset == "mnist":
        transform = Compose([RandomRotation(degrees=5, resample=Image.BILINEAR), ToTensor()])
        trainset = torchvision.datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST("data/mnist", train=False, download=True, transform=ToTensor())
        input_shape = (1, 28, 28)
        classes = torchvision.datasets.MNIST.classes
    elif dataset == "fashion-mnist":
        transform = Compose([RandomHorizontalFlip(), RandomRotation(degrees=5, resample=Image.BILINEAR), ToTensor()])
        trainset = torchvision.datasets.FashionMNIST("data/fashion-mnist", train=True, download=True,
                                                     transform=transform)
        testset = torchvision.datasets.FashionMNIST("data/fashion-mnist", train=False, download=True,
                                                    transform=ToTensor())
        input_shape = (1, 28, 28)
        classes = torchvision.datasets.FashionMNIST.classes
    elif dataset == "cifar-10":
        transform_tr = Compose([RandomHorizontalFlip(), Pad(4), RandomCrop(32), ToTensor()])
        trainset = torchvision.datasets.CIFAR10("data/cifar-10", train=True, download=True,
                                                transform=transform_tr)
        testset = torchvision.datasets.CIFAR10("data/cifar-10", train=False, download=True,
                                               transform=ToTensor())
        input_shape = (3, 32, 32)
        classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    trainloader = DataLoader(trainset, batch_size, True, num_workers=num_workers, pin_memory=use_gpu, drop_last=True)
    testloader = DataLoader(testset, batch_size, False, num_workers=num_workers, pin_memory=use_gpu, drop_last=False)

    return trainloader, testloader, input_shape, classes


def build_model(model, input_shape, feature_dims, num_classes):
    if model == "resnet":
        model = ResNet(input_shape, feature_dims, num_classes)
    else:
        raise NotImplementedError

    return model


def train(model, dataloader, criterion, optimizer, use_gpu, writer, epoch, max_epoch, vis, feat_dim, classes):
    model.train()

    if vis:
        if feat_dim == 2 or epoch == max_epoch - 1:
            all_features, all_labels = [], []
            all_images = []

    all_acc = []
    all_loss = []

    for idx, (data, labels) in tqdm(enumerate(dataloader), desc="Training Epoch {}".format(epoch)):
        optimizer.zero_grad()
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        acc = (outputs.data.max(1)[1] == labels.data).double().mean()
        all_acc.append(acc.item())

        writer.add_scalar("loss", loss.item(), global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("acc", acc.item(), global_step=epoch * len(dataloader) + idx)

        if vis:
            if feat_dim == 2 or epoch == max_epoch - 1:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            if feat_dim != 2 and epoch == max_epoch - 1:
                all_images.append(data.data.cpu().numpy())

    print("Epoch {}: total trainset loss: {}, global trainset accuracy:{}".format(epoch, np.mean(all_loss),
                                                                                  np.mean(all_acc)))

    if vis:
        if feat_dim == 2 or epoch == max_epoch - 1:
            visualize(all_images, all_features, all_labels, feat_dim, classes, epoch, writer, tag="train")


def eval(model, dataloader, criterion, scheduler, use_gpu, writer, epoch, max_epoch, vis, feat_dim, classes):
    model.eval()

    if vis:
        if feat_dim == 2 or epoch == max_epoch - 1:
            all_features, all_labels = [], []
            all_images = []

    all_acc = []
    all_loss = []

    with torch.no_grad():
        for idx, (data, labels) in tqdm(enumerate(dataloader), desc="Evaluating Epoch {}".format(epoch)):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            loss = criterion(outputs, labels)

            all_loss.append(loss.item())
            acc = (outputs.data.max(1)[1] == labels.data).double().mean()
            all_acc.append(acc.item())

            if vis:
                if feat_dim == 2 or epoch == max_epoch - 1:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                if feat_dim != 2 and epoch == max_epoch - 1:
                    all_images.append(data.data.cpu().numpy())

        val_loss = np.mean(all_loss)
        val_acc = np.mean(all_acc)
        writer.add_scalar("val_loss", val_loss, global_step=epoch)
        writer.add_scalar("val_acc", val_acc, global_step=epoch)
        print("Epoch {}: testset loss: {}, testset accuracy:{}".format(epoch, val_loss, val_acc))

        scheduler.step(val_acc)

        if vis:
            if feat_dim == 2 or epoch == max_epoch - 1:
                visualize(all_images, all_features, all_labels, feat_dim, classes, epoch, writer, tag="val")


def visualize(images, features, labels, feat_dim, classes, epoch, writer, tag):
    if feat_dim == 2:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        figure = plt.figure(figsize=[8., 8.])
        for idx in range(len(classes)):
            plt.scatter(features[labels == idx, 0],
                        features[labels == idx, 1],
                        c=colors[idx], s=1)
        figure.legend(classes, loc="upper right")
        writer.add_figure(tag=tag, figure=figure, global_step=epoch, close=True)
    else:
        images = torch.tensor(np.concatenate(images, axis=0))
        labels = np.concatenate(labels, axis=0)
        features = torch.tensor(np.concatenate(features, axis=0))
        writer.add_embedding(features, tag=tag, metadata=np.array(classes)[labels], label_img=images)
