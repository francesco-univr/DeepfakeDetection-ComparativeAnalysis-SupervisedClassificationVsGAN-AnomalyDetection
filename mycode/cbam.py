import torch, torch.nn as nn

# Channel attention module
class ChannelGate(nn.Module):
    def __init__(self, c, ratio=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c//ratio, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c//ratio, c, bias=False))
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*torch.sigmoid(y)

# Spatial attention module
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        avg = torch.mean(x,1,keepdim=True)
        mx  = torch.amax(x,1,keepdim=True)
        y = torch.cat([avg,mx],1)
        y = self.conv(y)
        return x*torch.sigmoid(y)
    
# Combine channel and spatial
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cg=ChannelGate(channels)
        self.sg=SpatialGate()
    def forward(self,x):
        return self.sg(self.cg(x)) 