# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generative Adversarial Networks (GANs) are one of the most interesting ideas
in computer science today. Two models are trained simultaneously by
an adversarial process. A generator ("the artist") learns to create images
that look real, while a discriminator ("the art critic") learns
to tell real images apart from fakes.
"""

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
import torchsummary


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets', help='path to datasets')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the inputs image to network')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam. default=0.9')
parser.add_argument('--epochs', type=int, default=200, help="Train loop")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./imgs', help='folder to output images')
parser.add_argument('--checkpoint_dir', default='./checkpoints', help='folder to output checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--phase', type=str, default='train', help='model mode. default=`train`, option=`generate`')

opt = parser.parse_args()

try:
  os.makedirs(opt.outf)
  os.makedirs(opt.checkpoint_dir)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = int(opt.ngpu)

# Loss weight for gradient penalty
lambda_gp = 10

fixed_noise = torch.randn(opt.batch_size, 100, 1, 1, device=device)


dataset = dset.ImageFolder(root=opt.dataroot,
                          transform=transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=int(opt.workers))


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # inputs is Z, going into a convolution
      nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
      nn.BatchNorm2d(64 * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32
      nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if torch.cuda.is_available() and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != "":
  netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.main = nn.Sequential(
      # inputs is (nc) x 64 x 64
      nn.Conv2d(3, 64, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64 * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if torch.cuda.is_available() and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != "":
  netD.load_state_dict(torch.load(opt.netD))
print(netD)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


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
  for epoch in range(opt.epochs):
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
      noise = torch.randn(batch_size, 100, 1, 1, device=device)

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

      print(f'[{epoch + 1}/{opt.epochs}][{i+1}/{len(dataloader)}] '
            f'Loss_D: {errD.item():.4f} '
            f'Loss_G: {errG.item():.4f}.')

      if i % 100 == 0:
        vutils.save_image(real_imgs,
                          f'{opt.outf}/real_samples.png',
                          normalize=True)
        fake = netG(fixed_noise).to(device)
        vutils.save_image(fake.detach(),
                          f'{opt.outf}/fake_samples_epoch_{epoch}.png',
                          normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), f'{opt.outf}/netG_epoch_{epoch + 1}.pth')
    torch.save(netD.state_dict(), f'{opt.outf}/netD_epoch_{epoch + 1}.pth')


def generate():
  """ random generate fake image.
  """
  ################################################
  #               load model
  ################################################
  print(f"Load model...\n")
  netG = Generator(ngpu).to(device)
  if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG))
  print(f"Load model successful!")
  with torch.no_grad():
    for i in range(64):
      z = torch.randn(1, 100, 1, 1, device=device)
      fake = netG(z)
      vutils.save_image(fake.detach(), f"unknown/fake_{i + 1:04d}.png", normalize=True)
  print("Images have been generated!")


if __name__ == '__main__':
  if opt.phase == 'train':
    train()
  elif opt.phase == 'generate':
    generate()
  else:
    print(opt)
