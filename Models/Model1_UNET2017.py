import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """ CNN Encoder """
        ## Encoder 1
        self.e1 = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 64)
        )

        self.e2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )

        self.e3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )

        self.e4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )

        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.maxpool3 = torch.nn.MaxPool2d(2)
        self.maxpool4 = torch.nn.MaxPool2d(2)

        """ CNN Decoder """
        ## Decoder 1
        self.d1 = DeconvBlock(512, 512)
        self.c1 = nn.Sequential(
            ConvBlock(512 + 512, 512),
            ConvBlock(512, 512)
        )

        ## Decoder 2
        self.d2 = DeconvBlock(512, 256)
        self.c2 = nn.Sequential(
            ConvBlock(256 + 256, 256),
            ConvBlock(256, 256)
        )

        ## Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.c3 = nn.Sequential(
            ConvBlock(128 + 128, 128),
            ConvBlock(128, 128)
        )

        ## Decoder 4
        self.d4 = DeconvBlock(128, 64)
        self.c4 = nn.Sequential(
            ConvBlock(64 + 64, 64),
            ConvBlock(64, 64)
        )

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, X):
        """ CNN Encoder """
        ## Encoder 1
        e1 = self.e1(X)
        #print('Encoder1', e1.shape)
        e1m = self.maxpool1(e1)
        #print('Encoder1m', e1m.shape)

        ## Encoder 2
        e2 = self.e2(e1m)
        #print('Encoder2', e2.shape)
        e2m = self.maxpool2(e2)
        #print('Encoder2m', e2m.shape)

        ## Encoder 3
        e3 = self.e3(e2m)
        #print('Encoder3', e3.shape)
        e3m = self.maxpool3(e3)
        #print('Encoder3m', e3m.shape)

        ## Encoder 4
        e4 = self.e4(e3m)
        #print('Encoder4', e4.shape)
        e4m = self.maxpool4(e4)
        #print('Encoder4m', e4m.shape)

        """ CNN Decoder """
        #print('Decoder')

        ## Decoder 1
        d1 = self.d1(e4m)
        # print('Decoder1', d1.shape)
        c1d = torch.cat([d1, e4], dim=1)
        c1 = self.c1(c1d)
        # print('Concatenation1+Reduction', c1.shape)

        ## Decoder 2
        d2 = self.d2(c1)
        # print('Decoder2', d2.shape)
        c2d = torch.cat([d2, e3], dim=1)
        c2 = self.c2(c2d)
        # print('Concatenation2+Reduction', c2.shape)

        ## Decoder 3
        d3 = self.d3(c2)
        # print('Decoder3', d3.shape)
        c3d = torch.cat([d3, e2], dim=1)
        c3 = self.c3(c3d)

        ## Decoder 4
        d4 = self.d4(c3)
        # print('Decoder4', d4.shape)
        c4d = torch.cat([d4, e1], dim=1)
        c4 = self.c4(c4d)

        """ Output """
        output = self.output(c4)

        #print('Output', output.shape)

        return output