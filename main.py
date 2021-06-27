import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root="./data", train=True, transform=transform)
