#!/usr/bin/env python

#TODO: Implement a GAN
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

        #self.conv_transpose_stack = nn.Sequential(

            #Input will be a 100 random uniform distribution
            #Step 1 - Projet and reshape
            #nn.ConvTranspose2d(100,1024,4),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(),

            #Step 2, project to 8x8 x 512
            #nn.ConvTranspose2d(1024,512,8),
            #nn.ReLU(),
            #nn.BatchNorm2d(512),

            #Step 3, stride of 2 to projet 8x8 to 16x16
            #nn.Conv2d(512, 256, 5, stride=2),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),

            #Step 4, stride of 2 to projet 16x16 to 32x32
            #nn.Conv2d(256, 3, 5, stride=2),
            #nn.Tanh(),
        #)

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


if __name__ == "__main__":
    train, valid, test = get_data_loader("svhn", 32)
    discriminator = Discriminator()
    generator = Generator()
    params = discriminator.parameters()
    optimizer = Adam(params)
    bceloss = nn.BCELoss() #Just for unit testing the discriminator, we will need to change the loss function
    best_acc = 0.
    cuda = torch.cuda.is_available()
    if cuda:
        discriminator = discriminator.cuda()

    for _ in range(50):
        discriminator.train()
        #Generator unit test
        z = torch.FloatTensor(32,100,1,1).uniform_(-1, 1)
        generated_picture = generator(z)

        for i, (x, y) in enumerate(train):
            if cuda:
                x = x.cuda()
                y = y.cuda()
            out = discriminator(x)
            always_true = torch.ones(out.shape).cuda() #Just for unit testing the discriminator
            m = nn.Sigmoid()
            loss = bceloss(m(out), always_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % 200 == 0:
                print(loss.item())
        acc = evaluate(discriminator, valid)
        print("Validation acc:", acc,)

        if acc > best_acc:
            best_acc = acc
            torch.save(discriminator, "discriminator.pt")
            print("Saved.")
    discriminator = torch.load("discriminator.pt")
    print("Test accuracy:", evaluate(discriminator, test))
