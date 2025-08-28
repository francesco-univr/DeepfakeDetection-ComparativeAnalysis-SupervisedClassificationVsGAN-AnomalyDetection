import torch.nn as nn

#Helper function to initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator architecture
# Takes a latent vector (random noise) as input and generates an image
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution that creates a 7x7 base
            nn.ConvTranspose2d(latent_dim, 1024, 7, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024 x 7 x 7
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 14 x 14
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 28 x 28
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 56 x 56
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 112 x 112
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # final state size. channels x 224 x 224
        )

    def forward(self, input):
        return self.main(input)


# Discriminator architecture
# Takes an image as input and outputs a single probability score (real vs. fake).
class Discriminator(nn.Module):
    def __init__(self, channels=3, num_classes=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False), # 112x112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 56x56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 14x14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), # 7x7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # The kernel size is changed from 7 to 8 to handle the 8x8 feature map
            # that results from a 256x256 input, ensuring a 1x1 output
            nn.Conv2d(1024, num_classes, 7, 1, 0, bias=False), 
        )
       

    def forward(self, input):
        # The output is a single logit
        return self.main(input).view(-1, 1)