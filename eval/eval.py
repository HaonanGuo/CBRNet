import torch
import numpy as np
from PIL import Image
import shutil,os
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
def eval_net(net, loader, device,savename=''):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    hist=0
    hist1=0
    for num,batch in enumerate(loader):
        if num%20==19:
            print('Validating ',num+1,'/',len(loader))
        imgs, true_masks,name = batch['image'], batch['mask'],batch['name']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = (true_masks>0).to(device=device, dtype=mask_type)
        with torch.no_grad():
            pred = net(imgs)
        pred1 = (pred[1] > 0).float()
        pred2 = (pred[0] > 0).float()
        hist+=fast_hist(pred1.flatten().cpu().detach().int().numpy(),true_masks.flatten().cpu().int().numpy(),2)
        hist1+=fast_hist(pred2.flatten().cpu().detach().int().numpy(),true_masks.flatten().cpu().int().numpy(),2)
    IOU=(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1]
    IOU2=(np.diag(hist1) / (hist1.sum(axis=1) + hist1.sum(axis=0) - np.diag(hist1)))[-1]
    print('IOU:', IOU)
    print('Refined IOU:', IOU2)
    print(hist1)
    return IOU
