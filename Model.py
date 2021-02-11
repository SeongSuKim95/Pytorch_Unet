import torch
import torch.nn as nn

import torchvision.transforms.functional as F

## Valid Convolution , Mirror padding


class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DoubleConv,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            # We set bias for False because we use Batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )


    def forward(self,x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels = 3 , out_channels = 1, features = [64,128,256,512]): #Binary output
        super(Unet,self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size = 2, stride =2 )
        # Problem
        # 161 x 161 -Maxpool--> 80 X 80 -output---> 160 X 160

        # Down part for Unet

        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        # Up part for Unet

        for feature in reversed(features):
            
            self.ups.append(
                
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size =2 , stride = 2,
                )

            )
            self.ups.append(DoubleConv(feature*2, feature))


        self.bottleneck = DoubleConv(features[-1],features[-1]*2) # 512, 1024
        self.final_conv = nn.Conv2d(features[0],out_channels, kernel_size = 1 )

    def forward(self,x):

        skip_connections = []

        for down in self.downs:

            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reversing

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x) # Up sampling
            skip_connection = skip_connections[idx//2] # Step in 2

            if x.shape != skip_connection.shape:
                x = F.resize(x, size = skip_connection.shape[2:]) # Except Batchsize

            concat_skip = torch.cat((skip_connection,x), dim = 1) # Concatnate
            x = self.ups[idx+1](concat_skip) # Run Dovble Conv

        return self.final_conv(x)



def test():

    x = torch.randn((3,1,161,161))
    model = Unet(in_channels =1 , out_channels =1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
