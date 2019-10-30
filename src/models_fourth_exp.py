import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
"""
Generator network
"""
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=True))#bias=False
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True))#bias=False
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)




class Generator(nn.Module):
    def __init__(self, opt, nclasses):
        super(Generator, self).__init__()

        self.deconv1 = deconv(1035, 64 * 4, 4, 4, 0)
        self.deconv2 = deconv(64 * 4, 64 * 2, 4)
        self.deconv3 = deconv(64 * 2, 64, 4)
        self.deconv4 = deconv(64, 3, 4, bn=False)

        self.ndim = 2 * opt.ndf
        self.ngf = opt.ngf
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses
        #
        # self.main = nn.Sequential(
        #
        #     nn.ConvTranspose2d(self.nz + self.ndim + nclasses + 1, self.ngf * 8, 2, 1, 0, bias=False),
        #     nn.BatchNorm2d(self.ngf * 8),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ngf * 4),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ngf * 2),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(self.ngf),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, input):
        batchSize = input.size()[0]

        input = input.view(-1, (self.ndim*4) + self.nclasses + 1, 1, 1)
        noise = torch.FloatTensor(batchSize, self.nz, 1, 1).normal_(0, 1)
        if self.gpu >= 0:
            noise = noise.cuda()
        noisev = Variable(noise)
        # output = self.main(torch.cat((input, noisev), 1))
        torch.cat((input, noisev), 1)
        out = F.leaky_relu(self.deconv1(torch.cat((input, noisev), 1)), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = torch.tanh(self.deconv4(out))
        return out



"""
Discriminator network
"""


class D1(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64):
        super(D1, self).__init__()
        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim * 2, 5, 2, 2)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 5, 2, 2)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 1, 0)
        self.fc1 = conv(conv_dim * 8, 11, 1, 1, 0, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = self.fc1(out).squeeze()

        return out


class D2(nn.Module):
    """Discriminator for mnist."""

    def __init__(self, conv_dim=64):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim * 2, 5, 2, 2)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 5, 2, 2)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 1, 0)
        self.fc1 = conv(conv_dim * 8, 1, 1, 1, 0, bn=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = self.fc1(out).squeeze()

        return out



class Discriminator(nn.Module):
    def __init__(self, opt, nclasses):
        super(Discriminator, self).__init__()

        self.d1 = D1()
        self.d2 = D2()


    def forward(self, input,is_src=True):
        if is_src:
            return self.d1(input)
        return self.d2(input)

        # output = self.feature(input)
        # output_s = self.classifier_s(output.view(-1, self.ndf * 2))
        # output_s = output_s.view(-1)
        # output_c = self.classifier_c(output.view(-1, self.ndf * 2))
        # return output_s, output_c


"""
Feature extraction network
"""


class Mixer(nn.Module):
    def __init__(self, opt,conv_dim=64):
        super(Mixer, self).__init__()



        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim * 2, 5, 2, 2)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 5, 2, 2)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, 1, 0)


    def forward(self, input):
        outc = F.leaky_relu(self.conv1(input), 0.05)
        outc = F.leaky_relu(self.conv2(outc), 0.05)

        outc = F.leaky_relu(self.conv3(outc), 0.05)
        outc = F.leaky_relu(self.conv4(outc), 0.05)

        # out = torch.cat((outc, code), 1)
        return outc.squeeze()


"""
Classifier network
"""


class Classifier(nn.Module):
    def __init__(self, opt, nclasses):
        super(Classifier, self).__init__()
        self.ndf = opt.ndf
        self.main = nn.Sequential(
            nn.Linear(8 * self.ndf, 2 * self.ndf),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.ndf, nclasses),
        )

    def forward(self, input):
        output = self.main(input)
        return output

