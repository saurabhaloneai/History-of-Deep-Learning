import torch 
from torch import nn
from essential import double_conv, down_sample, up_sample

import torch 
from torch import nn

# Double Conv as per the arc

class double_conv(nn.Module):
    
    def __init__(self, in_chan, out_chan):
        
        super().__init__()
        self.conv = nn.Sequential (
            
            nn.Conv2d(in_channels=in_chan,out_channels=out_chan,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=in_chan,out_channels=out_chan,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
     
        )
        
    def forward(self,x):
        
        return  self.conv(x)
    
    
# Down smapling layer nothting but the conv and pool 

class down_sample(nn.Module):
    
    def __init__(self,in_chan,out_chan):
        
        super().__init__()
        
        self.conv  = double_conv(in_chan,out_chan)
        
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1)
        
        
    def forward(self, x):
        
        down = self.conv(x)
        
        pl = self.pool(down)
        
        return down, pl
   
   
# Up sampling (Done by the Transposed_conv)


class up_sample(nn.Module):
    
    def __init__(self,in_chan,out_chan):
        
        super.__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels=in_chan,out_channels=out_chan // 2,kernel_size=2,stride=2), # // 2 this goes for torch.cat
        self.conv  = double_conv(in_chan,out_chan)
        
    def forward(self,x1,x2):
        
        x1 = self.up_conv(x1)
        
        x = torch.cat([x1,x2],1) #skip connections
        
        return self.conv(x)


class Unet(nn.Module):
    
    def __init__(self,in_chan,out_chan):
        
        super().__init__()
        self.down_conv_1 = down_sample(in_chan, 64)
        self.down_conv_2 = down_sample(64, 128)
        self.down_conv_3 = down_sample(128, 256)
        self.down_conv_4 = down_sample(256, 512)

        self.bottle_neck = double_conv(512, 1024)

        self.up_conv_1 = up_sample(1024, 512)
        self.up_conv_2 = up_sample(512, 256)
        self.up_conv_3 = up_sample(256, 128)
        self.up_conv_4 = up_sample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_chan, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_conv_1(x) # down_1 to donw_4 are skip connections. 
       down_2, p2 = self.down_conv_2(p1)
       down_3, p3 = self.down_conv_3(p2)
       down_4, p4 = self.down_conv_4(p3)

       b = self.bottle_neck(p4)

       up_1 = self.up_conv_1(b, down_4)
       up_2 = self.up_conv_2(up_1, down_3)
       up_3 = self.up_conv_3(up_2, down_2)
       up_4 = self.up_conv_4(up_3, down_1)

       out = self.out(up_4)
       
       return out
        
        
        
     