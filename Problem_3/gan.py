#!/usr/bin/env python

#TODO: Implement a GAN
import time
import torch
import torchvision.datasets
import torchvision.transforms as transforms
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


# For the basic GAN, we will use the same architecture as the classifier supplied 
# by the TA. Then I will adapt it to follow the guidelines in this paper:

# For the generator, we will get out inspiration form
# https://arxiv.org/pdf/1511.06434.pdf. 
# and attempt to follow their guidelines.
# Architecture guidelines for stable Deep Convolutional GANs
# •Replace any pooling layers with strided convolutions (discriminator) and fractional-stridedconvolutions (generator).
# •Use batchnorm in both the generator and the discriminator.
# •Remove fully connected hidden layers for deeper architectures.
# •Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# •Use LeakyReLU activation in the discriminator for all layers.



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            
            nn.Dropout2d(p=0.1),
            nn.Conv2d(8, 16, 3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            

            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            

            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2), #As in the paper https://arxiv.org/pdf/1511.06434.pdf

            nn.Conv2d(128, 512, 2),
        )

        self.mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.5),
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


# This will be used to implement a WGAN. GAN is too hard to train!

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

#Seems to mean the 
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


#https://arxiv.org/pdf/1511.06434.pdf
# All weights were initialized from a zero-centered Normal distributionwith standard deviation 0.02
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.02)
        m.bias.data.fill_(0)


def evaluate(classify, dataset):
    with torch.no_grad():
        classify.eval()
        correct = 0.
        total = 0.
        for x, y in dataset:
            if cuda:
                x = x.cuda()
                y = y.cuda()

            c = (classify(x).argmax(dim=-1) == y).sum().item()
            t = x.size(0)
            correct += c
            total += t
    acc = correct / float(total)
    return acc



def wgan_critic_loss(critic, real_pictures, fake_pictures):
    out_real = critic(real_pictures)
    out_fake = critic(fake_pictures)
    return torch.mean(out_fake-out_real)

def wgan_generator_loss(critic, real_pictures, fake_pictures):
    out_real = critic(real_pictures)
    out_fake = critic(fake_pictures)
    return torch.mean(out_fake-out_real)

import scipy




def train_gan():
    train, valid, test = get_data_loader("svhn", batch_size)
    discriminator = Discriminator()
    generator = Generator()
    params_discriminator = discriminator.parameters()
    optimizer_discriminator = Adam(params_discriminator,lr=0.0002)

    params_generator = generator.parameters()
    optimizer_generator = Adam(params_generator, lr=0.0001)

    bceloss = nn.BCELoss() #Just for unit testing the discriminator, we will need to change the loss function
    best_acc = 0.
    cuda = torch.cuda.is_available()
    if cuda:
        discriminator = discriminator.cuda()
        generator=generator.cuda()

    logfile = open("log.txt","w")
    print(optimizer_generator, file=logfile)
    print(optimizer_discriminator, file=logfile)
    logfile.flush()
    
    for epoch in range(50):
        discriminator.train()
        #For training, we will borrow ideas heavily from this paper.
        #https://arxiv.org/pdf/1706.08500.pdf

        #We will use:
        # The two time-scale update rule for GANs (One LR for the discriminator, and another one for the generator)
        #                                         (The discriminator must be faster than the )
        # Adam as the optimizer
        # Use Wasserstein GAN loss function
        #

        for i, (x, y) in enumerate(train):
            #Player 1: The discriminator plays the game
            #where it wants to tell a generated sample appart from a true sample.
            time.sleep(0.1)
            discriminator.zero_grad()
            m = nn.Sigmoid()
            z = torch.FloatTensor(batch_size,100,1,1).uniform_(-1, 1)
            
            if cuda:
                x_true = x.cuda()
                z=z.cuda()
            generated_picture = generator(z)
            out_true = m(discriminator(x_true))
            out_fake = m(discriminator(generated_picture))

            generated_picture.register_hook(save_gradient)

            out = torch.cat((out_fake,out_true),0)

            always_true = torch.ones(out_true.shape).cuda() #Just for unit testing the discriminator
            always_fake = torch.zeros(out_fake.shape).cuda() #Just for unit testing the discriminator
            always_true_fake = torch.ones(out_fake.shape).cuda() #Just for unit testing the discriminator

            target = torch.cat((always_fake,always_true),0)
            
            loss_discriminator = bceloss(out, target) 
            loss_discriminator.backward(retain_graph=True)
            optimizer_discriminator.step()
            
            D_real = out_true.mean().item()
            D_fake1 = out_fake.mean().item()
            
            if (i % 10)== 0: #Each 5 iterations
                generator.zero_grad()
                generated_picture = generator(z)
                out_fake2 = m(discriminator(generated_picture))
                D_fake2 = out_fake2.mean().item()

                loss_generator = bceloss(out_fake, always_true_fake) #If the generator is not fooling the discriminator, it's bad
                loss_generator.backward()

                print("Epoch: %2d, Iteration: %3d, D_Loss: % 5.5f, G_Loss: % 5.2f, D_real: % 5.2f D_fake1: % 5.2f D_fake2: % 5.2f " 
                    %(epoch, i, loss_discriminator.item(), loss_generator.item(), D_real, D_fake1, D_fake2)) 

                print("Epoch: %2d, Iteration: %3d, D_Loss: % 5.5f, G_Loss: % 5.2f, D_real: % 5.2f D_fake1: % 5.2f D_fake2: % 5.2f " 
                    %(epoch, i, loss_discriminator.item(), loss_generator.item(), D_real, D_fake1, D_fake2), file=logfile) 
                optimizer_generator.step()
        print ("Epoch done")
        scipy.misc.imsave('out' + str(epoch) +'.png',generated_picture[0].detach().cpu().numpy().swapaxes(0,2))
        torch.save(generator.state_dict(), "generator_epoch" + str(epoch) + ".pt")
        torch.save(discriminator.state_dict(), "discriminator_epoch" + str(epoch) + ".pt")
        
    discriminator = torch.load("discriminator.pt")
    print("Test accuracy:", evaluate(discriminator, test))



saved_gradient = None
def save_gradient(gradient):
    global saved_gradient
    saved_gradient= gradient.detach()
    #print("gradient")

batch_size=256


#Will will train a WGAN instead. Should be easier to make it work.
def train_wgan():
    gp_scaling = 10 #Lambda in the paper
    n_critic=5
    train, valid, test = get_data_loader("svhn", batch_size)
    critic = Critic()
    generator = Generator()
    params_critic = critic.parameters()
    optimizer_critic = Adam(params_critic,lr=0.0001, betas=(0.0,0.9))

    params_generator = generator.parameters()
    optimizer_generator = Adam(params_generator, lr=0.0001, betas=(0.0,0.9))

    cuda = torch.cuda.is_available()
    if cuda:
        critic = critic.cuda()
        generator=generator.cuda()

    logfile = open("log.txt","w")
    print(optimizer_generator, file=logfile)
    print(optimizer_critic, file=logfile)
    logfile.flush()
    
    for epoch in range(50):
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

        
        
        for i, (true_picture, y) in enumerate(train):
            #Player 1: The discriminator plays the game
            #where it wants to tell a generated sample appart from a true sample.
            time.sleep(0.1)
            
            z = torch.FloatTensor(true_picture.shape[0],100,1,1).uniform_(-1, 1)
            epsilon = torch.FloatTensor(true_picture.shape[0],1,1,1).uniform_(0, 1)
            if cuda:
                true_picture = true_picture.cuda()
                z=z.cuda()
                epsilon=epsilon.cuda()
            generated_picture = generator(z)
            merged_picture = epsilon*true_picture - (1-epsilon)*generated_picture

            #Get the gradient for the merged picture part
            critic.zero_grad()
            merged_picture.register_hook(save_gradient) #This will save gradient with regards to X 
                                                        #in the "saved_gradient" global variable.
            out_fake = critic(generated_picture)
            #out_fake = critic(merged_picture)
            loss_out_fake = out_fake.mean()
            loss_critic = loss_out_fake
            
            loss_out_fake.backward()


            out_real = critic(true_picture)
            

            
            #Time to compute the gradient penalty
            #gp = torch.mul(torch.tensor(gp_scaling).float().cuda(), torch.pow((saved_gradient.view(64,-1).norm(dim=1) -1),2).mean())
            #gp.backward()
            loss_out_real = -out_real.mean() #+ gp
            loss_out_real.backward()
            loss_critic+=loss_out_real
            
            torch.nn.utils.clip_grad_norm(params_critic,2)
            optimizer_critic.step()
            
            D_real = out_real.mean().item()
            D_fake1 = out_fake.mean().item()
            
            if (i%n_critic)==0:
                scipy.misc.imsave('/tmp/out.png',generated_picture[0].detach().cpu().numpy().swapaxes(0,2))
                generator.zero_grad()

                generated_picture = generator(z)
                out_fake2 = critic(generated_picture)
                D_fake2 = out_fake2.mean().item()

                #Wasserstein distance
                wd = D_real - D_fake1

                #loss_generator = bceloss(out_fake, always_true_fake) #If the generator is not fooling the discriminator, it's bad
                loss_generator = out_fake2.mean()
                loss_generator.backward()

                print("Epoch: %2d, Iteration: %3d, C_Loss: % 5.5f, G_Loss: % 5.2f, D_real: % 5.2f D_fake1: % 5.2f D_fake2: % 5.2f Wd: % 5.2f" 
                    %(epoch, i, loss_critic.item(), loss_generator.item(), D_real, D_fake1, D_fake2, wd)) 

                print("Epoch: %2d, Iteration: %3d, C_Loss: % 5.5f, G_Loss: % 5.2f, D_real: % 5.2f D_fake1: % 5.2f D_fake2: % 5.2f Wd: % 5.2f" 
                    %(epoch, i, loss_critic.item(), loss_generator.item(), D_real, D_fake1, D_fake2, wd), file=logfile) 

                torch.nn.utils.clip_grad_norm(params_generator,2)
                optimizer_generator.step()
        print ("Epoch done")
        scipy.misc.imsave('out' + str(epoch) +'.png',generated_picture[0].detach().cpu().numpy().swapaxes(0,2))

    discriminator = torch.load("discriminator.pt")
    print("Test accuracy:", evaluate(discriminator, test))


if __name__ == "__main__":
    train_gan()
