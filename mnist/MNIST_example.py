import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import mnist_dataset as ds
# import torchvision
# import matplotlib.pyplot as plt
# import numpy as np
# import time
from tensorboard_logger import configure, log_value, log_images


def save_model(epoch, model,
               optimizer, path='./ckpt'):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path)


def load_model(model,
               optimizer, path='./ckpt'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

BATCH = 4

# tensorboard
configure("runs/run-1234")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # single input channel and 6 output, 5*5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), 2)
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

mnistmTrainSet = ds.mnistmTrainingDataset(
                    text_file='/home/tom/Downloads/mnist_png/testing/list.txt',
                    root_dir='/home/tom/Downloads/mnist_png/testing')

mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet, batch_size=BATCH,
                                                shuffle=True, num_workers=2)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#put on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

count = 0

for epoch in range(100):
    
    #load prev model
    net, optimizer, epoch = load_model(model=net, optimizer=optimizer)

    for i, sample_batched in enumerate(mnistmTrainLoader, 0):
        input_batch = f.pad(sample_batched['image'].float(), (2, 2, 2, 2))
        input_batch = input_batch.to(device)

        optimizer.zero_grad()

        output = net(input_batch)
        loss = loss_fn(output, sample_batched['labels'].to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 50 == 0:
            count = count + 1
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))
            log_value('loss', loss.item(), count)
            _, ind = output.max(1)
            name = 'pred_' + str(ind[0])
            sample_image = sample_batched['image'][0]
            log_images(name, sample_image, count)
    
    #save model
    save_model(epoch=epoch, model=net, optimizer=optimizer)
    

