import torch
import torch.nn as nn

class deconv2dBlock(nn.Module):
    def __init__(self , layers):
        super(deconv2dBlock , self ).__init__()
        self.relu = nn.ReLU(inplace = True)

        self.conv1 = nn.Conv2d(layers[0], layers[1] ,  3, stride = 1, padding = 1,bias = False)
        self.bn1 = nn.BatchNorm2d(layers[1])

        self.conv2 = nn.Conv2d( layers[1],  layers[2] , 3, stride = 1, padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(layers[2])
        self.deconv = nn.ConvTranspose2d(layers[2] , layers[2] , kernel_size = 2 , stride = 2, padding = 0)

    def forward(self , x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv(x)

        return x


class conv2dBlock(nn.Module):
    def __init__(self , layers):
        super(conv2dBlock , self ).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.MaxPool2d = nn.MaxPool2d(kernel_size = 2 , stride = 2 , padding = 0)

        self.conv1 = nn.Conv2d(layers[0], layers[1] ,  3, stride = 1, padding = 1,bias = False)
        self.bn1 = nn.BatchNorm2d(layers[1])

        self.conv2 = nn.Conv2d( layers[1],  layers[2] , 3, stride = 1, padding = 1,bias = False)
        self.bn2 = nn.BatchNorm2d(layers[2])

    def forward(self , x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        downsampled = self.MaxPool2d(x)

        return downsampled, x


class UNet(nn.Module):
    def __init__(self):
        super(UNET , self).__init__()

        self.convblock1 = conv2dBlock([1 , 32 , 64])
        self.convblock2 = conv2dBlock([64 , 64 , 128])
        self.convblock3 = conv2dBlock([128 , 128 , 256])
        self.convblock4 = conv2dBlock([256 , 256 , 512])

        self.deconvblock1 = deconv2dBlock([512,512,1024])
        self.deconvblock2 = deconv2dBlock([512 + 1024 , 512 , 512])
        self.deconvblock3 = deconv2dBlock([256 + 512, 256 , 256])
        self.deconvblock4 = deconv2dBlock([128 + 256 ,128 ,128])

        self.convblock5 = conv2dBlock([64 + 128 , 64 , 64])
        self.finalConv2d = torch.nn.Conv2d(64,1,kernel_size = 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self , x) :
        downsampled1, conv_out1 = self.convblock1(x)
        downsampled2, conv_out2  = self.convblock2(downsampled1)
        downsampled3, conv_out3 = self.convblock3(downsampled2)
        downsampled4, conv_out4 = self.convblock4(downsampled3)

        deconv_out1 = self.deconvblock1(downsampled4)
        deconv_out2 = self.deconvblock2(torch.cat((conv_out4 ,deconv_out1) , dim = 1 ))
        deconv_out3 = self.deconvblock3(torch.cat((conv_out3 ,deconv_out2), dim = 1))
        deconv_out4 = self.deconvblock4(torch.cat((conv_out2 ,deconv_out3) , dim = 1))

        _ , conv_out = self.convblock5(torch.cat((conv_out1 , deconv_out4 ) , dim = 1))
        out = self.finalConv2d(conv_out)
        return out

# def main():
#     a = torch.rand([5,1,128,128])
#     net = UNET()
#     b = net(a)
#     print(b.shape)

# if __name__ == "__main__":
#     main()
