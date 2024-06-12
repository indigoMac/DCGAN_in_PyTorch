# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([
    transforms.Resize(imageSize), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]) 

# Loading the dataset
dataset = dset.CIFAR10(root='./data', download=True, transform=transform) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2) 

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator - I found this archetiecture online from a research paper, I will find a link for this somewhere.
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Creating the generator
netG = G()
netG.apply(weights_init)

# Defining the discriminator
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# Creating the discriminator
netD = D()
netD.apply(weights_init)

# Training the DCGANs

criterion = nn.BCELoss() # We create the criterion that will measure the error function (loss)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999)) # We create the optimizers for the discriminator
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999)) # We create the optimizers for the generator

# Adding device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)
criterion.to(device)

if __name__ == "__main__":
    for epoch in range(25): # We iterate over 25 epochs
        for i, data in enumerate(dataloader, 0):
            # 1st Step: Updating the weights of the neural network of the discriminator
            netD.zero_grad() # We initialize to 0 the gradients of the discriminator with respect to the weights
            real, _ = data # We get the real images of the minibatch
            input = real.to(device) # Move data to device
            target = torch.ones(input.size()[0], device=device) # We get the target
            output = netD(input) # We forward propagate the real images into the discriminator to get the prediction (a value between 0 and 1)
            errD_real = criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1)
            errD_real.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator

            noise = torch.randn(input.size()[0], 100, 1, 1, device=device) # We generate noise in the latent space to feed the generator
            fake = netG(noise) # We pass the noise through the generator to get a fake image
            target = torch.zeros(input.size()[0], device=device) # We get the target
            output = netD(fake.detach()) # We forward propagate the fake generated images into the discriminator to get the prediction (a value between 0 and 1)
            errD_fake = criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 0)
            errD_fake.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator

            errD = errD_real + errD_fake # We compute the total error of the discriminator
            optimizerD.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator

            # 2nd Step: Updating the weights of the neural network of the generator
            netG.zero_grad() # We initialize to 0 the gradients of the generator with respect to the weights
            target = torch.ones(input.size()[0], device=device) # We get the target
            output = netD(fake) # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1)
            errG = criterion(output, target) # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1)
            errG.backward() # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator
            optimizerG.step() # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator

            # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.item(), errG.item())) # We print the losses of the discriminator (Loss_D) and the generator (Loss_G)
            if i % 100 == 0: # Every 100 steps
                vutils.save_image(real, '%s/real_samples.png' % "/content/drive/My Drive/GANs/results2", normalize=True) # We save the real images of the minibatch
                fake = netG(noise) # We get our fake generated images
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("/content/drive/My Drive/GANs/results2", epoch), normalize=True) # We also save the fake generated images of the minibatch
