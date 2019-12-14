import argparse
import os
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='| lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='inputs batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
  os.makedirs(opt.outf)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
  # folder dataset
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
elif opt.dataset == 'lsun':
  dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                      transform=transforms.Compose([
                        transforms.Resize(opt.imageSize),
                        transforms.CenterCrop(opt.imageSize),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
elif opt.dataset == 'fake':
  dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                          transform=transforms.ToTensor())

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ndf = int(opt.ndf)
ngf = int(opt.ngf)
nc = 3

# Loss weight for gradient penalty
lambda_gp = 10


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  def __init__(self, gpus):
    super(Generator, self).__init__()
    self.ngpu = gpus
    self.main = nn.Sequential(
      # inputs is Z, going into a convolution
      nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, inputs):
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs


netG = Generator(ngpu)
netG.apply(weights_init)

if opt.netG != '':
  netG = torch.load(opt.netG)


class Discriminator(nn.Module):
  def __init__(self, gpus):
    super(Discriminator, self).__init__()
    self.ngpu = gpus
    self.main = nn.Sequential(
      # inputs is (nc) x 64 x 64
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
    )

  def forward(self, inputs):
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)

    return outputs.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu)
netD.apply(weights_init)

if opt.netD != '':
  netD = torch.load(opt.netD)

if opt.cuda:
  netD.to(device)
  netG.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))


def calculate_gradient_penatly(netD, real_imgs, fake_imgs):
  """Calculates the gradient penalty loss for WGAN GP"""
  eta = torch.FloatTensor(real_imgs.size(0), 1, 1, 1).uniform_(0, 1).to(device)
  eta = eta.expand(real_imgs.size(0), real_imgs.size(1), real_imgs.size(2), real_imgs.size(3)).to(device)

  interpolated = eta * real_imgs + ((1 - eta) * fake_imgs)
  interpolated.to(device)

  # define it to calculate gradient
  interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

  # calculate probaility of interpolated examples
  prob_interpolated = netD(interpolated)

  # calculate gradients of probabilities with respect to examples
  gradients = autograd.grad(
    outputs=prob_interpolated,
    inputs=interpolated,
    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
  )[0]

  gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

  return gradients_penalty


def train():
  for epoch in range(opt.niter):
    for i, (real_imgs, _) in enumerate(dataloader):

      # configure input
      real_imgs = real_imgs.to(device)

      # Get real imgs batch size
      batch_size = real_imgs.size(0)

      # -----------------
      #  Train Discriminator
      # -----------------

      netD.zero_grad()

      # Sample noise as generator input
      noise = torch.randn(batch_size, nz, 1, 1, device=device)

      # Generate a batch of images
      fake_imgs = netG(noise)

      # Real images
      real_validity = netD(real_imgs)
      # Fake images
      fake_validity = netD(fake_imgs)
      # Gradient penalty
      gradient_penalty = calculate_gradient_penatly(netD, real_imgs.data, fake_imgs.data)

      # Loss measures generator's ability to fool the discriminator
      errD = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

      errD.backward()
      optimizerD.step()

      optimizerG.zero_grad()

      # Train the generator every n_critic iterations
      if i % opt.n_critic == 0:

        # ---------------------
        #  Train Generator
        # ---------------------

        # Generate a batch of images
        fake_imgs = netG(noise)
        # Adversarial loss
        errG = -torch.mean(netD(fake_imgs))

        errG.backward()
        optimizerG.step()

      print(f'[{epoch + 1}/{opt.niter}][{i}/{len(dataloader)}] '
            f'Loss_D: {errD.item():.4f} '
            f'Loss_G: {errG.item():.4f}.')

      if epoch % 5 == 0:
        vutils.save_image(real_imgs,
                          f'{opt.outf}/real_samples.png',
                          normalize=True)
        vutils.save_image(netG(noise).detach(),
                          f'{opt.outf}/fake_samples_epoch_{epoch}.png',
                          normalize=True)

    # do checkpointing
    torch.save(netG, f'{opt.outf}/netG_epoch_{epoch + 1}.pth')
    torch.save(netD, f'{opt.outf}/netD_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
  train()
