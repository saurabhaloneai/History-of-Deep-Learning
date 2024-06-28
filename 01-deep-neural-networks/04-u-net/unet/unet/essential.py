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
        
        self.up_conv = nn.ConvTranspose2d(in_channels=in_chan,out_channels=out_chan // 2,kernel_size=2,stride=2),
        self.conv  = double_conv(in_chan,out_chan)
        
    def forward(self,x1,x2):
        
        x1 = self.up_conv(x1)
        
        x = torch.cat([x1,x2],1) #main func
        
        return self.conv(x)
