import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=32, shuffle=True)
data_iter = iter(data_loader)
images, labels = data_iter.next()
print(torch.min(images), torch.max(images))
