import torch
from efficientunet import *

class efficientunetb2(nn.Module):

    def __init__(self, args):
        super(efficientunetb2, self).__init__()
        self.args = args

        self.channel_conv = nn.Sequential(
            nn.Conv2d(args.n_channels, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.effuet = get_efficientunet_b2(out_channels=args.n_class, concat_input=args.concat_input, pretrained=True)

    def forward(self, x):
        
        x = self.channel_conv(x)
        x = self.effuet(x)

        return x
