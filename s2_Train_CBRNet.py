import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import matplotlib
matplotlib.use('tkagg')
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from eval.eval import eval_net
from utils.dataset import SegfixDataset
from torch.utils.data import DataLoader
from utils.sync_batchnorm.batchnorm import convert_model
from unet.unet_model import CBRNet
lr=1e-3
batchsize=8
epochs=99999
num_workers=24
read_name=''
save_name='CBR_Inria'
Dataset='Inria'
assert Dataset in ['WHU_BUILDING','Inria','Mass']
print(save_name)
net=CBRNet()
print(sum(p.numel() for p in net.parameters()))
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True):
    traindataset = SegfixDataset(traindir_img, traindir_mask,get_edge=True,training=True)
    valdataset = SegfixDataset(valdir_img, valdir_mask,training=False)
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(traindataset)}
        Validation size: {len(valdataset)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')
    optimizer=optim.Adam(net.module.parameters(),lr=lr,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer,1,0.7)
    bcecriterion = nn.BCEWithLogitsLoss()
    edgecriterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.85))
    cecriterion=nn.CrossEntropyLoss(ignore_index=-1)
    print('Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
    if os.path.exists(os.path.join('checkpoints', read_name + '.pth')):
        best_val_score =eval_net(net, val_loader, device,savename=Dataset+'_'+read_name)#
        print('Best iou:',best_val_score)
        no_optim=0
    else:
        print('Training new model....')
        best_val_score=-1
    for epoch in range(epochs):
        net.train()
        net.module.fixer.eval()
        net.module.fixer.requires_grad_(False)
        epoch_loss = 0
        with tqdm(total=len(traindataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for num,batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = batch['mask']>0
                dir_masks = batch['direction_map']
                dis_masks = batch['distance_map']
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)
                dir_masks = dir_masks.to(device=device)
                edge_masks = (dis_masks<5).to(device=device).float()
                dir_masks[edge_masks==0]=-1
                remasks_pred,masks_pred,pred2,pred3,pred4,pred5,edge1,edge2,edge3,edge4,direction = net(imgs)
                loss =bcecriterion(true_masks.squeeze(), torch.sigmoid(remasks_pred).squeeze().float())+ \
                      bcecriterion(remasks_pred.squeeze(), true_masks.squeeze().float())+ \
                      bcecriterion(masks_pred.squeeze(), true_masks.squeeze().float())+ \
                      edgecriterion(edge1.squeeze(), edge_masks.squeeze().float())+ \
                      cecriterion(direction,dir_masks.long())+ \
                      0.25*bcecriterion(pred2.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(256,256)).squeeze().float())+ \
                      0.25*bcecriterion(pred3.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(128,128)).squeeze().float())+ \
                      0.25*bcecriterion(pred4.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(64,64)).squeeze().float())+ \
                      0.25*bcecriterion(pred5.squeeze(), F.interpolate(true_masks,mode='bilinear',size=(32,32)).squeeze().float())+ \
                      0.25*edgecriterion(edge2.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(256,256)).squeeze().float())+ \
                      0.25*edgecriterion(edge3.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(128,128)).squeeze().float())+ \
                      0.25*edgecriterion(edge4.squeeze(), F.interpolate(edge_masks.unsqueeze(1),mode='bilinear',size=(64,64)).squeeze().float())

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
            val_score = eval_net(net, val_loader, device)
            if val_score>best_val_score:
                best_val_score=val_score
                torch.save(net.module.state_dict(),
                           dir_checkpoint +save_name+'_best.pth')
                logging.info(f'Checkpoint {save_name} saved !')
                no_optim=0
            else:
                no_optim=no_optim+1
            if no_optim>3:
                net.module.load_state_dict(torch.load(dir_checkpoint +save_name+'_best.pth'))
                scheduler.step()
                print('Scheduler step!')
                print('Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
                no_optim=0
            if optimizer.state_dict()['param_groups'][0]['lr']<1e-7:
                break
traindir_img = '../../'+Dataset+'/train/image/'
traindir_mask = '../../'+Dataset+'/train/label/'
valdir_img = '../../'+Dataset+'/val/image/'
valdir_mask = '../../'+Dataset+'/val/label/'
testdir_img = '../../'+Dataset+'/test/image/'
testdir_mask = '../../'+Dataset+'/test/label/'
dir_checkpoint = 'checkpoints/'
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if read_name!='':
        net_state_dict=net.state_dict()
        state_dict=torch.load(dir_checkpoint+read_name+'.pth', map_location=device)
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict)
        logging.info(f'Model loaded from '+read_name+'.pth')

    net = convert_model(net)
    net = torch.nn.parallel.DataParallel(net.to(device))
    torch.backends.cudnn.benchmark = True
    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              device=device)