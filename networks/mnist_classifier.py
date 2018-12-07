import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from tensorboard_logger import configure, log_value, log_images
import os
import sys
sys.path.append('../torch_utils')
import dataset as ds
import torch_io as tio

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


def main(args):
    net = Net()

    mnistmTrainSet = ds.mnistmTrainingDataset(
                        text_file=args.dataset_list)

    mnistmTrainLoader = torch.utils.data.DataLoader(
                                            mnistmTrainSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    count = 0

    epoch = [0]

    # load prev model
    tio.load_model(model=net, optimizer=optimizer, epoch=epoch)
    epoch = epoch[0]

    while epoch < 100:

        for i, sample_batched in enumerate(mnistmTrainLoader, 0):
            input_batch = f.pad(sample_batched['image'].float(), (2, 2, 2, 2))
            input_batch = input_batch.to(device)

            optimizer.zero_grad()

            output = net(input_batch)
            loss = loss_fn(output, sample_batched['labels`'].to(device))
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

        # save model
        tio.save_model(epoch=epoch, model=net, optimizer=optimizer)
        epoch = epoch + 1
