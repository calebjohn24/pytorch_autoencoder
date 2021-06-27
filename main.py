import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from  model import Autoencoder




transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root="./data", download=True, train=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=10, shuffle=True)
data_iter = iter(data_loader)
images, labels = data_iter.next()
print(torch.min(images), torch.max(images))
model = Autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3, 
                             weight_decay=1e-5)


num_epochs = 8
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
        recon = model(img)
        loss = criterion(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    outputs.append((epoch, img, recon))
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')

print("done")
for output in outputs:
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = output[1].detach().numpy()
    recon = output[2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])

plt.show()