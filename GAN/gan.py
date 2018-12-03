import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import mnist_dataset as ds

BATCH = 12

class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.g_fc1 = nn.Linear(100, 256)
        self.g_norm = torch.nn.BatchNorm2d(1)
        self.g_fc2 = nn.Linear(256, 512)
        self.g_fc3 = nn.Linear(512, 28 * 28)

    def forward(self, input):
        # input vector of length 100
        self.gen_imgs = f.leaky_relu(self.g_fc1(input))
        self.gen_imgs = self.g_norm(self.gen_imgs)

        self.gen_imgs = f.leaky_relu(self.g_fc2(self.gen_imgs))
        self.gen_imgs = self.g_norm(self.gen_imgs)

        self.gen_imgs = torch.sigmoid(self.g_fc3(self.gen_imgs))
        # self.gen_imgs = self.g_norm(self.gen_imgs)

        self.gen_imgs = self.gen_imgs.view(-1, 1, 28, 28)

        return self.gen_imgs


class descrimanator(nn.Module):

    def __init__(self):
        super(descrimanator, self).__init__()
        self.d_fc1 = nn.Linear(28 * 28, 512)
        self.d_fc2 = nn.Linear(512, 256)
        self.d_fc3 = nn.Linear(256, 1)

    def forward(self, input):
        self.desc = input.view(-1, 1, 1, 28 * 28)

        self.desc = f.leaky_relu(self.d_fc1(self.desc))

        self.desc = f.leaky_relu(self.d_fc2(self.desc))

        self.desc = torch.sigmoid(self.d_fc3(self.desc))

        return self.desc

gen = generator()
desc = descrimanator()

mnistmTrainSet = ds.mnistmTrainingDataset(
                    text_file='/home/tom/Downloads/mnist_png/testing/list.txt',
                    root_dir='/home/tom/Downloads/mnist_png/testing')

mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet, batch_size=BATCH,
                                                shuffle=True, num_workers=2)


# put on gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss = torch

gen.to(device)
desc.to(device)

gen_optimizer = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
desc_optimizer = optim.SGD(desc.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    for i, sample_batched in enumerate(mnistmTrainLoader, 0):
        input_batch = sample_batched['image'].float()
        input_batch = input_batch.to(device)
        
        gen_optimizer.zero_grad()
        desc_optimizer.zero_grad()

        noise = torch.empty(input_batch.shape[0], 1, 1, 100).uniform_().to(device)
        gen_imgs = gen(noise)

        gen_desc = desc(gen_imgs)
        real_desc = desc(input_batch)

        # # train descrimanator
        desc_loss_real = torch.log(real_desc.clamp(min=1e-4))
        loss_gen = torch.log((1 - gen_desc).clamp(min=1e-4))
        desc_loss = 0.5 * (loss_gen + desc_loss_real)
        
        desc_loss.sum().backward(retain_graph=True)
        loss_gen.sum().backward(retain_graph=True)

        print(desc_loss.sum())

        gen_optimizer.step()
        desc_optimizer.step()
        

        # train generator
        

        

        
