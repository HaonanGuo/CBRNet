from unet.unet_parts import *
from torchvision import models

class CBRNet(nn.Module):
    def __init__(self):
        super(CBRNet, self).__init__()
        bilinear = True
        state=[64,128,256,512,1024]
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 64
        self.down2 = vgg16_bn.features[12:22]  # 64
        self.down3 = vgg16_bn.features[22:32]  # 64
        self.down4 = vgg16_bn.features[32:42]  # 64
        del vgg16_bn
        self.bcecriterion = nn.BCEWithLogitsLoss()
        self.edgecriterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.85))
        self.cecriterion=nn.CrossEntropyLoss(ignore_index=-1)
        factor = 2 if bilinear else 1
        self.classifier1=nn.Sequential(nn.Conv2d(512,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.up1 = Up(state[4],state[3] // factor, bilinear)
        self.classifier2=nn.Sequential(nn.Conv2d(256+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.classifier2_2=nn.Sequential(nn.Conv2d(256,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.up2 = Up(state[3],state[2] // factor, bilinear)
        self.classifier3=nn.Sequential(nn.Conv2d(128+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.classifier3_2=nn.Sequential(nn.Conv2d(128,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.up3 = Up(state[2],state[1] // factor, bilinear)
        self.classifier4=nn.Sequential(nn.Conv2d(64+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.classifier4_2=nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.up4 = Up(state[1],state[0], bilinear)
        self.classifier5=nn.Sequential(nn.Conv2d(64+1,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.classifier5_2=nn.Sequential(nn.Conv2d(64,64,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,1))
        self.interpo=nn.Upsample(scale_factor=2, mode='bilinear')

        self.fixer=fix_seg()
    def forward(self, x,gts=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        seg5=self.classifier1(x5)
        x = self.up1(x5, x4)
        edge4=self.classifier2_2(x)
        seg4=self.classifier2(torch.cat((x,self.interpo(seg5)),1))
        x = self.up2(x,  x3)
        edge3=self.classifier3_2(x)
        seg3=self.classifier3(torch.cat((x,self.interpo(seg4)),1))
        x = self.up3(x, x2)
        edge2=self.classifier4_2(x)
        seg2=self.classifier4(torch.cat((x,self.interpo(seg3)),1))
        x = self.up4(x, x1)
        edge1=self.classifier5_2(x)
        seg1=self.classifier5(torch.cat((x,self.interpo(seg2)),1))
        direction=self.dir_head(x)
        r_x=self.fixer(direction,seg1,edge1)
        if gts is not None:
            true_masks,edge_masks,dir_masks=gts['mask'],gts['edge'],gts['direction']
            loss=0.25*self.bcecriterion(seg2.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(256,256)).squeeze().float())+ \
                  0.25*self.bcecriterion(seg3.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(128,128)).squeeze().float())+ \
                  0.25*self.bcecriterion(seg4.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(64,64)).squeeze().float())+ \
                  0.25*self.bcecriterion(seg5.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(32,32)).squeeze().float())+ \
                  0.25*self.edgecriterion(edge2.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(256,256)).squeeze().float())+ \
                  0.25*self.edgecriterion(edge3.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(128,128)).squeeze().float())+ \
                  0.25*self.edgecriterion(edge4.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(64,64)).squeeze().float())
            return loss,[r_x,seg1,seg2,seg3,seg4,seg5,edge1,edge2,edge3,edge4,direction]
        else:
            return r_x,seg1,seg2,seg3,seg4,seg5,edge1,edge2,edge3,edge4,direction
