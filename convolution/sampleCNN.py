#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False,
    transform = ToTensor()
)


# In[3]:


figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# In[4]:


loaders = {
    'train' : torch.utils.data.DataLoader(train_data,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=1),
    'test' : torch.utils.data.DataLoader(test_data,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1)
}


# In[5]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # (int) -> number of channels in input image    =1 because output is grayscale image
                out_channels=16, # (int) -> number of channels produced by the convolution
                kernel_size=5, # (int, tuple) -> size of convolving kernel
                stride=1, # (int, tuple, optional) -> stride of convolution, default is 1
                padding=2, # (int, tuple, optional) -> zero-padding added to both sides of input, default is 0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x       


# In[6]:


cnn = CNN()


# In[7]:


loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)  


# In[19]:

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


import time
start_time = time.time()


# In[8]:


num_epochs = 10
def train(num_epochs, cnn, loaders):
    cnn.train()
    total_step = len(loaders['train'])
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalizes x when iterating train_loader
            b_x = Variable(images)
            b_y = Variable(labels)
            
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this trainign step
            optimizer.zero_grad()
            # backpropogation, computing gradients
            loss.backward()
            # apply gradients
            optimizer.step()
            if (i+1) & 100 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i+1, total_step, loss.item()))
                pass
            pass
        pass
train(num_epochs, cnn, loaders)


# In[20]:


curr_time = time.time()
print(f"Training took {curr_time - start_time} seconds")


# In[9]:


def test():
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
    print('Test accuracy of model on 10000 test images: %.2f' % accuracy)
    pass
test()


# In[10]:


sample = next(iter(loaders['test']))
imgs, lbls = sample
actual_number = lbls[:10].numpy()


# In[11]:


test_output, last_layer = cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')


# In[15]:


from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path 
MODEL_NAME = "01_pytorch_mnist_cnn.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=cnn.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)


# In[16]:


# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = CNN()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))


# In[18]:


loaded_model_0.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loaders['test']:
        test_output, last_layer = loaded_model_0(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        pass
print('Test accuracy of model on 10000 test images: %.2f' % accuracy)

