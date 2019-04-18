#!/usr/bin/env python
import argparse
#TODO: Implement a GAN
import scipy
import time
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils

from torch.utils.data import dataset
from torch import nn
# from torch.nn.modules import upsampling
# from torch.functional import F
from torch.optim import Adam

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])


def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader



# This will be used to implement a WGAN. 
# It is basically the TA classifier, but tweaked to be a discriminator
# by using the tips the paper. https://arxiv.org/pdf/1704.00028.pdf
# Such as no batch norm.

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            #nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            nn.Dropout2d(p=0.1),
            nn.Conv2d(8, 16, 3, padding=1, stride=2),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            

            nn.Conv2d(16, 16, 3, padding=1),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            

            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            

            nn.Conv2d(32, 64, 3, padding=1),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            

            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2), #As in the paper https://arxiv.org/pdf/1511.06434.pdf

            nn.Conv2d(128, 512, 2),
        )

        self.mlp = nn.Sequential(
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.5),
            nn.Linear(512, 1), #We just want to know if the image is fake or not.
        )

        init_weights(self)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x):
        x  = self.mlp(self.extract_features(x))
        return x 

    def extract_features(self, x):
        return self.conv_stack(x)[:, :, 0, 0]

# For the generator, we will get out inspiration form
# https://arxiv.org/pdf/1511.06434.pdf. 
# and attempt to follow their guidelines.
# Architecture guidelines for stable Deep Convolutional GANs
# •Replace any pooling layers with strided convolutions (discriminator) and fractional-stridedconvolutions (generator).
# •Use batchnorm in both the generator and the discriminator.
# •Remove fully connected hidden layers for deeper architectures.
# •Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# •Use LeakyReLU activation in the discriminator for all layers.

# We will do the same architecture as in the paper as a starting point.
# Obviously, our images are smaller so we will tweek it a little bit.

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Step 1, project to 1024x4x4
        self.conv1 = nn.ConvTranspose2d(100,1024,4,1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

            #Step 2, project to 8x8 x 512
        self.conv2 = nn.ConvTranspose2d(1024,512,5)
        self.relu2 =     nn.ReLU()
        self.bn2 =     nn.BatchNorm2d(512)

            #Step 3, stride of 2 to projet 8x8 to 16x16
        self.conv3=    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.relu3=    nn.ReLU()
        self.bn3 =    nn.BatchNorm2d(256)

            #Step 4, stride of 2 to projet 16x16 to 32x32
        self.conv4 =    nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1)
        self.tanh4 =    nn.Tanh()

        init_weights(self)

    def forward(self, x):
        #x  = self.conv_transpose_stack(x) [:, :, 0, 0]
        x = self.conv1(x) # Step 1, project to 1024x4x4
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x) #Step 2, project to 8x8 x 512
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x) #Step 3, stride of 2 to projet 8x8 to 16x16
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x) #Step 4, stride of 2 to projet 16x16 to 32x32
        x = self.tanh4(x)

        return x 


#Designed to remove checkboard artifact https://distill.pub/2016/deconv-checkerboard/

class GeneratorV2(nn.Module):

    def __init__(self):
        super(GeneratorV2, self).__init__()
        # Step 1, project to 1024x4x4
        self.conv1 = nn.ConvTranspose2d(100,1024,4,1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

            #Step 2, project to 8x8 x 512
        self.us2 = nn.Upsample((8,8))
        self.conv2 = nn.ConvTranspose2d(1024,512,3, stride=1, padding=1)
        self.relu2 =     nn.ReLU()
        self.bn2 =     nn.BatchNorm2d(512)

            #Step 3, stride of 2 to projet 8x8 to 16x16
        self.us3 = nn.Upsample((16,16))
        self.conv3=    nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1)
        self.relu3=    nn.ReLU()
        self.bn3 =    nn.BatchNorm2d(256)

            #Step 4, stride of 2 to projet 16x16 to 32x32
        self.us4 = nn.Upsample((32,32))
        self.conv4 =    nn.ConvTranspose2d(256, 3, 3, stride=1, padding=1)
        self.tanh4 =    nn.Tanh()

        init_weights(self)

    def forward(self, x):
        #x  = self.conv_transpose_stack(x) [:, :, 0, 0]
        x = self.conv1(x) # Step 1, project to 1024x4x4
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.us2(x)
        x = self.conv2(x) #Step 2, project to 8x8 x 512
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.us3(x)
        x = self.conv3(x) #Step 3, stride of 2 to projet 8x8 to 16x16
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.us4(x)
        x = self.conv4(x) #Step 4, stride of 2 to projet 16x16 to 32x32
        x = self.tanh4(x)

        return x 

#https://arxiv.org/pdf/1511.06434.pdf
# All weights were initialized from a zero-centered Normal distributionwith standard deviation 0.02
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.02)
        m.bias.data.fill_(0)

#batch_size=64

#This one computes the gradient penalty.
def compute_gp(true_picture, fake_pictures, critic):
    cuda = torch.cuda.is_available()
    epsilon = torch.rand(true_picture.shape[0],1,1,1).uniform_(0, 1)
    gradient_outputs = torch.ones((true_picture.shape[0],1))
    ones_epsilon = torch.ones((true_picture.shape[0],1,1,1))

    if cuda:
        epsilon = epsilon.cuda()
        gradient_outputs = gradient_outputs.cuda()
        ones_epsilon = ones_epsilon.cuda()

    merged_pictures = epsilon*true_picture + ((ones_epsilon-epsilon)*fake_pictures)
    out_merged = critic(merged_pictures)
    
    gradients = torch.autograd.grad(out_merged, merged_pictures, gradient_outputs, create_graph=True)
    saved_gradient = gradients[0]
    gp = torch.pow((saved_gradient.view(64,-1).norm(dim=1) -1),2).mean()
    return gp

# Training of WGAN-GP

def train_wgan():
    gp_scaling = 10 #Lambda in the paper
    parser = argparse.ArgumentParser(
        description='Run a wgan-gp with parameters')
    parser.add_argument('--start_iteration', type=int, default=0,
                        help='Iteration number. If more than 0, it will load a saved critic/generator pair.')
    parser.add_argument('--lr_generator', type=float, default=0.00005,
                        help='Generator Learning Rate')
    parser.add_argument('--lr_critic', type=float, default=0.0001,
                        help='Generator Learning Rate')
    parser.add_argument('--n_critic', type=float, default=20,   #5 in the paper, but the TA told us that we might need to tweak this one.
                        help='Number of critic iterations')
    parser.add_argument('--gp_scaling', type=float, default=10,  #10 in the paper
                        help='Gradient Penalty Constant (Lambda)')
    parser.add_argument('--max_iteration', type=int, default=100,  
                        help='Iteration when to stop')
    parser.add_argument('--suffix', type=str, default="default",  
                        help='id for the test run')
    parser.add_argument('--batch_size', type=int, default=64,  #64 in the paper
                        help='number of element per batch')
    parser.add_argument('--no-save', type=bool, default=False, 
                        help='Activate in orde to not save at each epoch')
    args = parser.parse_args()

    suffix= args.suffix
    n_critic = args.n_critic
    gp_scaling = args.gp_scaling
    batch_size=args.batch_size

    train, valid, test = get_data_loader("svhn", batch_size)
    critic = Critic()
    generator = GeneratorV2()
    params_critic = critic.parameters()
    optimizer_critic = Adam(params_critic,lr=args.lr_critic, betas=(0.0,0.9))

    params_generator = generator.parameters()
    optimizer_generator = Adam(params_generator, lr=args.lr_generator, betas=(0.0,0.9))

    cuda = torch.cuda.is_available()
    if cuda:
        critic = critic.cuda()
        generator=generator.cuda()

    logfile = open("log"+suffix+".txt","w")
    print(optimizer_generator, file=logfile)
    print(optimizer_critic, file=logfile)
    logfile.flush()
    
    if(args.start_iteration!=0):
        critic.load_state_dict(torch.load("critic"+ suffix + str(args.start_iteration)+ ".pt"))
        generator.load_state_dict(torch.load("generator"+ suffix + str(args.start_iteration)+ ".pt"))

    for epoch in range(args.start_iteration, args.max_iteration):
        critic.train()
        #For training, we will borrow ideas heavily from this paper.
        #https://arxiv.org/pdf/1706.08500.pdf

        #We will use:
        # The two time-scale update rule for GANs (One LR for the discriminator, and another one for the generator)
        #                                         (The discriminator must be faster than the )
        # Adam as the optimizer
        # Use Wasserstein GAN loss function
        #


        #https://arxiv.org/pdf/1704.00028.pdf Following this algorithm

        
        
        for i, (true_pictures, y) in enumerate(train):
            #Player 1: The discriminator plays the game
            #where it wants to tell a generated sample appart from a true sample.
            time.sleep(0.1)
            
            z = torch.FloatTensor(true_pictures.shape[0],100,1,1).normal_(0,1)
            
            if cuda:
                true_pictures = true_pictures.cuda()
                z=z.cuda()
            fake_pictures = generator(z)
            #

            #Get the gradient for the merged picture part
            critic.zero_grad()
            generator.zero_grad()

            out_fake = critic(fake_pictures)
            out_real = critic(true_pictures)
            wd = out_real.mean() - out_fake.mean() 
            gp = compute_gp(true_pictures, fake_pictures, critic)

            loss = -wd + gp_scaling * gp
            loss.backward()

            #torch.nn.utils.clip_grad_value_(params_critic,0.001)
            generator.zero_grad() #Make sure the generator does not play here
            optimizer_critic.step()
            
            D_real = out_real.mean().item()
            D_fake1 = out_fake.mean().item()
            
            if (i%n_critic)==0:
                generator.zero_grad()

                fake_pictures = generator(z)
                out_fake2 = critic(fake_pictures)
                D_fake2 = out_fake2.mean().item()

                loss_generator = - out_fake2.mean()
                loss_generator.backward()

                print("Epoch: %2d, Iteration: %3d, C_Loss: % 5.5f, G_Loss: % 5.2f, D_real: % 5.2f D_fake1: % 5.2f D_fake2: % 5.2f Wd: % 5.2f" 
                    %(epoch, i, loss.item(), loss_generator.item(), D_real, D_fake1, D_fake2, wd)) 

                print("Epoch: %2d, Iteration: %3d, C_Loss: % 5.5f, G_Loss: % 5.2f, D_real: % 5.2f D_fake1: % 5.2f D_fake2: % 5.2f Wd: % 5.2f" 
                    %(epoch, i, loss.item(), loss_generator.item(), D_real, D_fake1, D_fake2, wd), file=logfile) 

                critic.zero_grad() #Make sure no gradient is applied on the discriminator for the generator turn.
                optimizer_generator.step()
        print ("Epoch done")
        
        #new pictures:
        z = torch.FloatTensor(64,100,1,1).normal_(0,1)
        if cuda:
            z = z.cuda()
        fake_pictures = generator(z)

        if not args.no_save:
            torch.save(critic.state_dict(), "critic"+ suffix + str(epoch)+ ".pt")
            torch.save(generator.state_dict(), "generator" +suffix + str(epoch)+ ".pt")
        torchvision.utils.save_image(fake_pictures.detach().cpu(), 'out' + suffix + str(epoch) +'.png', range=(-1,1))
        #torchvision.utils.save_image(fake_pictures.detach().cpu(), 'out' + suffix + 'actual_'+ str(image)+'.png', range=(-1,1))
           

    print ("Training done, results in log.txt")

if __name__ == "__main__":
    train_wgan()
