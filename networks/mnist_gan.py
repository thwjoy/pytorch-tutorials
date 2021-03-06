import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from tensorboard_logger import configure, log_value, log_images
import os
import sys
from torch_utils import dataset as ds
from torch_utils import torch_io as tio
import time
import math


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.g_fc1 = nn.Linear(100, 256)
        self.g_norm_1 = torch.nn.BatchNorm2d(1)
        self.g_norm_64 = torch.nn.BatchNorm2d(64)
        self.g_fc2 = nn.Linear(256, 28 * 28)
        self.dropout = nn.Dropout2d(p=0.2)
        self.g_c1 = nn.Conv2d(1, 64, 5, padding=2)
        self.g_c2 = nn.Conv2d(64, 64, 5, padding=2)
        self.g_c3 = nn.Conv2d(64, 64, 5, padding=2)
        self.g_c4 = nn.Conv2d(64, 1, 5, padding=2)

    def forward(self, input):
        # input vector of length 100
        self.gen_imgs = f.leaky_relu(self.g_fc1(input))
        self.gen_imgs = self.g_norm_1(self.gen_imgs)

        self.gen_imgs = f.leaky_relu(self.g_fc2(self.gen_imgs))
        self.gen_imgs = self.g_norm_1(self.gen_imgs)

        self.gen_imgs = self.gen_imgs.view(-1, 1, 28, 28)

        self.gen_imgs = self.dropout(f.leaky_relu(self.g_c1(self.gen_imgs)))
        self.gen_imgs = self.g_norm_64(self.gen_imgs)

        self.gen_imgs = self.dropout(f.leaky_relu(self.g_c2(self.gen_imgs)))
        self.gen_imgs = self.g_norm_64(self.gen_imgs)

        self.gen_imgs = self.dropout(f.leaky_relu(self.g_c3(self.gen_imgs)))
        self.gen_imgs = self.g_norm_64(self.gen_imgs)

        self.gen_imgs = self.dropout(f.leaky_relu(self.g_c4(self.gen_imgs)))
        self.gen_imgs = self.g_norm_1(self.gen_imgs)

        self.gen_imgs = torch.tanh(self.gen_imgs)

        return self.gen_imgs


class descrimanator(nn.Module):

    def __init__(self):
        super(descrimanator, self).__init__()
        self.d_c1 = nn.Conv2d(1, 64, 5, padding=2)
        self.d_c2 = nn.Conv2d(64, 64, 5, padding=2)
        self.d_c3 = nn.Conv2d(64, 1, 5, padding=2)
        self.d_norm_1 = torch.nn.BatchNorm2d(1)
        self.d_norm_64 = torch.nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(p=0.2)
        self.d_fc1 = nn.Linear(28 * 28, 512)
        self.d_fc2 = nn.Linear(512, 256)
        self.d_fc3 = nn.Linear(256, 1)

    def forward(self, input):
        self.desc = self.dropout(f.leaky_relu(self.d_c1(input)))

        self.desc = self.dropout(f.leaky_relu(self.d_c2(self.desc)))

        self.desc = self.dropout(f.leaky_relu(self.d_c3(self.desc)))

        self.desc = input.view(-1, 1, 1, 28 * 28)

        self.desc = f.leaky_relu(self.d_fc1(self.desc))

        self.desc = f.leaky_relu(self.d_fc2(self.desc))

        self.desc = torch.sigmoid(self.d_fc3(self.desc))

        return self.desc


def normalize(tensor):
    return tensor.div_(torch.norm(tensor,2))


def train(args):
    # tensorboard
    run_name = "./runs/run-GAN_batch_" + str(args.batch_size) \
                    + "_epochs_" + str(args.epochs) + "_" + args.log_message

    configure(run_name)

    gen = generator()
    desc = descrimanator()

    mnistmTrainSet = ds.mnistmTrainingDataset(
            text_file=args.dataset_list)

    mnistmTrainLoader = torch.utils.data.DataLoader(
                                        mnistmTrainSet,
                                        batch_size=args.batch_size,
                                        shuffle=True, num_workers=2)

    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = torch

    gen.to(device)
    desc.to(device)

    gen_optimizer = optim.Adam(gen.parameters())
    desc_optimizer = optim.SGD(desc.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss()

    epoch = [0]

    #load prev model
    tio.load_model(model=gen, optimizer=gen_optimizer, epoch=epoch, path=run_name + '/ckpt_gan')
    tio.load_model(model=desc, optimizer=desc_optimizer, epoch=epoch, path=run_name + '/ckpt_desc')
    epoch = epoch[0]

    while epoch < args.epochs:

        for i, sample_batched in enumerate(mnistmTrainLoader, 0):
            input_batch = sample_batched['image'].float()
            input_batch = 2 * (input_batch - 0.5)
            input_batch = input_batch.to(device)

            # Create the labels which are later used as input for the BCE loss
            # ones_labels = torch.ones(input_batch.shape[0], 1, 1, 1).to(device)
            real_labels = 0.99 * torch.ones(input_batch.shape[0], 1, 1, 1).to(device)
            fake_labels = 0.01 * torch.ones(input_batch.shape[0], 1, 1, 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            real_desc = desc(input_batch)
            desc_loss_real = criterion(real_desc, real_labels)

            noise = torch.randn(input_batch.shape[0], 1, 1, 100).to(device)
            gen_imgs = gen(noise)
            gen_desc = desc(gen_imgs)
            desc_loss_fake = criterion(gen_desc, fake_labels)
            desc_loss = desc_loss_fake + desc_loss_real

            gen_optimizer.zero_grad()
            desc_optimizer.zero_grad()
            desc_loss.backward()
            desc_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            noise = torch.empty(input_batch.shape[0], 1, 1, 100).uniform_().to(device)
            gen_imgs = gen(noise)
            gen_desc = desc(gen_imgs)

            gen_loss = criterion(gen_desc, real_labels)

            gen_optimizer.zero_grad()
            desc_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()    

            if (i+1) % 200 == 0:
                count = int(epoch * math.floor(len(mnistmTrainSet) / (args.batch_size * 200)) + (i / 200))
                print('Epoch [{}], Step [{}], desc_loss: {:.4f}, gen_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, i+1, desc_loss.item(), gen_loss.item(), 
                        real_desc.mean().item(), gen_desc.mean().item()))
                log_value("Gen Loss", gen_loss.item(), count)
                log_value("Desc Loss", desc_loss.item(), count)
                log_value("D(x)", real_desc.mean().item(), count)
                log_value("D(G(z))", gen_desc.mean().item(), count)
                for i in range(input_batch.shape[0]):
                    log_images("generated", gen_imgs[i].detach(), count)
            

            # train generator
            #save model
        tio.save_model(model=gen, optimizer=gen_optimizer, epoch=epoch, path=run_name + '/ckpt_gan')
        tio.save_model(model=desc, optimizer=desc_optimizer, epoch=epoch, path=run_name + '/ckpt_desc')
        epoch = epoch + 1


