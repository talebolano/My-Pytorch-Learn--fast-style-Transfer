import torch
import torch.nn as nn
from torchvision.models import vgg16,resnet


class residual(nn.Module):
    def __init__(self,input_channel,out_channel,kernel_size,strides):
        super(residual,self).__init__()
        self.conv1 = nn.Conv2d(input_channel,out_channel,kernel_size,strides,(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size,strides,(kernel_size-1)//2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + x

        return out


class genrate_model(nn.Module):
    def __init__(self):
        super(genrate_model,self).__init__()


        self.model = nn.Sequential(
            nn.ConstantPad2d((10,10,10,10),0),
            nn.Conv2d(3,32,9,1,4),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,64,3,2,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,3,2,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            residual(128, 128, 3, 1),
            residual(128, 128, 3, 1),
            residual(128, 128, 3, 1),
            residual(128, 128, 3, 1),
            residual(128, 128, 3, 1),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,1,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,3,9,1,4),
            nn.InstanceNorm2d(3),
            nn.Tanh()
        )

        # self.res1 = residual(128,128,3,1)
        # self.res2 = residual(128,128,3,1)
        # self.res3 = residual(128,128,3,1)
        # self.res4 = residual(128,128,3,1)
        # self.res5 = residual(128,128,3,1)
    def forward(self, x):
        x = self.model(x)
        b,c,h,w = x.size()
        x = x[:,:,10:h-10,10:w-10]
        return x


class style_model(nn.Module):
    def __init__(self):
        super(style_model,self).__init__()
        vggmodel = vgg16(pretrained=True)
        self.model1 = nn.Sequential(
            vggmodel.features[0],
            vggmodel.features[1],
            vggmodel.features[2],
            vggmodel.features[3]
        )
        self.model2 = nn.Sequential(
            vggmodel.features[4],
            vggmodel.features[5],
            vggmodel.features[6],
            vggmodel.features[7],
            vggmodel.features[8]
        )
        self.model3 = nn.Sequential(
            vggmodel.features[9],
            vggmodel.features[10],
            vggmodel.features[11],
            vggmodel.features[12],
            vggmodel.features[13],
            vggmodel.features[14],
            vggmodel.features[15]
        )
        self.model4 = nn.Sequential(
            vggmodel.features[16],
            vggmodel.features[17],
            vggmodel.features[18],
            vggmodel.features[19],
            vggmodel.features[20],
            vggmodel.features[21],
            vggmodel.features[22]
        )

    def forward(self, x,mode='style'):
        if mode=='style':
            out1 = self.model1(x)
            out2 = self.model2(out1)
            out3 = self.model3(out2)
            out4 = self.model4(out3)
            return out1, out2, out3, out4
        else:
            out1 = self.model1(x)
            out2 = self.model2(out1)
            out3 = self.model3(out2)

            return out3


if  __name__=="__main__":
    model = style_model()
    print(model.model1[0].bias)
    x = torch.randn(1,3,128,128)

    print(model(x)[-2].shape)


