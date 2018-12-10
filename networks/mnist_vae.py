import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import configure, log_value, log_images
import os
import sys
from torch_utils import dataset as ds
from torch_utils import torch_io as tio
from torchvision.utils import save_image

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


def train(args):

    # tensorboard
    run_name = "./runs/run-vae_batch_" + str(args.batch_size) \
                    + "_epochs_" + str(args.epochs) + "_" + args.log_message
    configure(run_name)
    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VAE().to(device)

    mnistmTrainSet = ds.mnistmTrainingDataset(
                        text_file=args.dataset_list)

    mnistmTrainLoader = torch.utils.data.DataLoader(
                                            mnistmTrainSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    criterion = nn.BCELoss(size_average=False)
    optimizer = optim.Adam(net.parameters())

    epoch = [0]

    # load prev model
    tio.load_model(model=net, optimizer=optimizer, epoch=epoch, path=run_name + '/ckpt')
    epoch = epoch[0]

    while epoch < args.epochs:

        for i, sample_batched in enumerate(mnistmTrainLoader, 0):
            input_batch = sample_batched['image'].float()
            input_batch = input_batch.to(device)

            input_batch = input_batch.view(-1, image_size)

            reconstruct, mu, log_var = net(input_batch)

            reconstruct_loss = criterion(reconstruct, input_batch)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = reconstruct_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 200 == 0:
                count = int(epoch * math.floor(len(mnistmTrainSet) / (args.batch_size * 200)) + (i / 200))
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item()))
                log_value('loss', loss.item(), count)
                z = torch.randn(args.batch_size, z_dim).to(device)
                gen = net.decode(z).view(-1, 1, 28, 28)
                log_images('generated', gen[0].detach(), count)
                out, _, _ = net(input_batch)
                out = out.view(args.batch_size, 1, 28, 28)
                log_images('image', sample_batched['image'][0], count)
                log_images('recon', out[0].detach(), count)
                
        # save model
        tio.save_model(epoch=epoch, model=net, optimizer=optimizer, path=run_name + '/ckpt')
        epoch = epoch + 1
