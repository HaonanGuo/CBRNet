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
batchsize=8
num_workers=24
read_name='CBR_Inria_best'
Dataset='Inria'
assert Dataset in ['WHU_BUILDING','Inria','Mass512']
net=CBRNet()
print(sum(p.numel() for p in net.parameters()))
def eval_net(net,
              device,
              batch_size):
    testdataset = SegfixDataset(testdir_img, testdir_mask,training=False)
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    best_score =eval_net(net, test_loader, device,savename=Dataset+'_'+read_name)#
    print('Best iou:',best_score)


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
    eval_net(net=net,
              batch_size=batchsize,
              device=device)