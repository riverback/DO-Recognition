from typing import List, Optional

import os
import cv2
import random
import numpy as np

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from train_model import test_and_generate_cams
from resnet_models import BasicNet
from get_dataloader import get_loader

import pydensecrf.densecrf as dcrf

def iou_pytorch(outputs:torch.Tensor, labels:torch.Tensor, smooth=1e-6):
    # iou = tp / (fp+tp+fn)
    tp = ((outputs==1).long() + (labels==1).long()) == 2
    fp = ((outputs==1).long() + (labels==0).long()) == 2
    fn = ((outputs==0).long() + (labels==1).long()) == 2
    
    iou = (tp.sum()+smooth) / (tp.sum()+fp.sum()+fn.sum()+smooth)
    return iou

def iou_numpy(outputs:np.ndarray, labels:np.ndarray, smooth=1e-6):
    # iou = tp / (fp+tp+fn)
    tp = ((outputs==1).astype(np.uint8) + (labels==1).astype(np.uint8)) == 2
    fp = ((outputs==1).astype(np.uint8) + (labels==0).astype(np.uint8)) == 2
    fn = ((outputs==0).astype(np.uint8) + (labels==1).astype(np.uint8)) == 2
    
    iou = (tp.sum()+smooth) / (tp.sum()+fp.sum()+fn.sum()+smooth)
    
    return iou

def visulization_cams(model:BasicNet,
                      device:torch.device,
                      test_loader:DataLoader,
                      num_classes:int,
                      checkpoint_path:str,
                      vis_folder:str,
                      return_cams=True,
                      target_layer:Optional[str or None]='layer3'):
    # generate cams
    cams = test_and_generate_cams(model, device, test_loader, num_classes, checkpoint_path, return_cams, target_layer).cpu() # Size (N, 3, H, W) min-max: 0.-1.
    '''print(cams.size())
    torch.save(cams, os.path.join(vis_folder, 'olf_cams.pt'))
    return 0.'''
    img_nums = cams.size(0)
    
    mIoU = 0.
    threshold = 0.5
    
    for i in range(img_nums):
        img_i = cams[i][0].numpy()
        mask_i = cams[i][1].numpy()
        cam_i = cams[i][2].numpy()
        
        # add threshold before blur
        cam_i = np.where(cam_i>0.4, cam_i, 0)
        
        # add blur on layer1 cam
        if target_layer == 'layer1':
            cam_i = cv2.blur(cam_i, (5,5))
        
        # add threshold after blur
        cam_i = np.where(cam_i>threshold, cam_i, 0)
        
        # cal iou
        mIoU += iou_numpy(np.where(cam_i>0, 1, 0), mask_i)
        
        # generate heat map
        heat_img = cam_i*255
        heat_img = heat_img.astype(np.uint8)
        
        heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

        img_i = np.expand_dims(img_i, axis=2)
        img_i = (np.concatenate((img_i, img_i, img_i), axis=2) * 255).astype(np.uint8)
        
        mask_i = np.expand_dims(mask_i, axis=2)
        mask_i = (np.concatenate((mask_i, mask_i, mask_i), axis=2) * 255).astype(np.uint8)
        
        img_add = cv2.addWeighted(img_i, 0.5, heat_img, 0.5, 0)
        
        # plt.imsave(os.path.join(vis_folder, '{:04d}_img.png'.format(i)), cv2.resize(img_i, (256, 256)))
        # plt.imsave(os.path.join(vis_folder, '{:04d}_mask.png'.format(i)), cv2.resize(mask_i, (256, 256)))
        # plt.imsave(os.path.join(vis_folder, '{:04d}_heatmap.png'.format(i)), cv2.resize(heat_img, (256, 256)))
        # plt.imsave(os.path.join(vis_folder, '{:04d}_img_add.png'.format(i)), cv2.resize(img_add, (256, 256)))
        plt.imsave(os.path.join(vis_folder, '{:04d}_all.png'.format(i)), 
                    np.concatenate((cv2.resize(img_i, (64, 64)), cv2.resize(heat_img, (64, 64)), cv2.resize(mask_i, (64, 64)), cv2.resize(img_add, (64, 64))), axis=1))

    mIoU = mIoU / img_nums
    
    return mIoU

def suppressCMAs(vis_folder:str,
                 cam_root:str,
                 threshold:float,
                 coe:List[float]=[0.8,0.4,0.2],
                 target_layers:Optional[List[str] or None]=['layer1', 'layer2', 'layer3']):
    """
    vis_folder: folder for visulization results
    cam_root: cam root
    coe: weight for cam suppression
    """
    
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
    vis_folder = os.path.join(vis_folder, 'threshold-{}'.format(threshold))
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
        
    def weighted_sum(coe, cam_list):
        suppressCAMs = None
        for i in range(len(coe)):
            if suppressCAMs is None:
                suppressCAMs = coe[i] * cam_list[i]
            else:
                suppressCAMs += coe[i] * cam_list[i]
        return suppressCAMs
    
    assert len(coe) == len(target_layers), f'len(coe) != len(target_layer)'
    coe = [c/sum(coe) for c in coe]
    cam_list = [torch.load(os.path.join(cam_root, layer, 'olf_cams.pt')) for layer in target_layers]
    
    masks = torch.load('saved_checkpoints/masks.pt')
    images = torch.load('saved_checkpoints/images.pt')
    suppressed_cam = weighted_sum(coe, cam_list)
    
    # cal iou
    img_nums = suppressed_cam.size(0)
    suppressed_cam_threshold = torch.where(suppressed_cam>threshold, torch.ones(1), torch.zeros(1))
    mIoU = iou_pytorch(suppressed_cam_threshold, masks)
    print(mIoU)
    with open(os.path.join(vis_folder, 'iou.txt'), 'a+') as f:
        '''f.write(coe)
        f.write('\n')
        f.write(target_layers)
        f.write('\n')'''
        f.write('miou: {}'.format(mIoU))
    # val cams
    for i in range(img_nums):
        img_i = images[i].squeeze().numpy()
        mask_i = masks[i].squeeze().numpy()
        cam_i = suppressed_cam[i].squeeze().numpy()
        cam_threshold_i = np.expand_dims(suppressed_cam_threshold[i].squeeze().numpy(), axis=2).astype(np.uint8)
        
        heat_img = cam_i*255
        heat_img = heat_img.astype(np.uint8)
        
        heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

        img_i = np.expand_dims(img_i, axis=2)
        img_i = (np.concatenate((img_i, img_i, img_i), axis=2) * 255).astype(np.uint8)
        
        mask_i = np.expand_dims(mask_i, axis=2)
        cam_threshold_i = (np.concatenate((mask_i, cam_threshold_i, mask_i), axis=2) * 255).astype(np.uint8)
        mask_i = (np.concatenate((mask_i, mask_i, mask_i), axis=2) * 255).astype(np.uint8)
        
        img_add = cv2.addWeighted(img_i, 0.5, heat_img, 0.5, 0)

        # plt.imsave(os.path.join(vis_folder, '{:04d}_img.png'.format(i)), cv2.resize(img_i, (256, 256)))
        # plt.imsave(os.path.join(vis_folder, '{:04d}_mask.png'.format(i)), cv2.resize(mask_i, (256, 256)))
        # plt.imsave(os.path.join(vis_folder, '{:04d}_heatmap.png'.format(i)), cv2.resize(heat_img, (256, 256)))
        # plt.imsave(os.path.join(vis_folder, '{:04d}_img_add.png'.format(i)), cv2.resize(img_add, (256, 256)))
        plt.imsave(os.path.join(vis_folder, '{:04d}_all.png'.format(i)), 
                    np.concatenate((cv2.resize(img_i, (64, 64)), cv2.resize(heat_img, (64, 64)), cv2.resize(cam_threshold_i, (64, 64)), cv2.resize(mask_i, (64, 64)), cv2.resize(img_add, (64, 64))), axis=1))
 
def main(target_layer:str, checkpoint_path:str, device_idx:int, num_classes:int):
    
    num_classes = num_classes
    
    vis_folder = checkpoint_path.split('/')[0] + '/' + checkpoint_path.split('/')[1] + '/' + 'cams_vis' + '/{}'.format(target_layer)
    iou_record = checkpoint_path.split('/')[0] + '/' + checkpoint_path.split('/')[1] + '/' + 'cams_vis' + '/iou-{}.txt'.format(target_layer)
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
    
    # set random seed
    seed = 4096
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(device_idx)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = BasicNet(1, num_classes, False).to(device)
    
    test_loader = get_loader('Data/OLF_spines/OLF_test_camOLF.txt', 'test', 0.5, 128, 16, 'OLF_Dataset')
    
    mIoU = visulization_cams(model, device, test_loader, num_classes, checkpoint_path, vis_folder, True, target_layer)
    with open(iou_record, 'w+') as f:
        f.write('mIoU: {:.4f}'.format(mIoU))

if __name__ == '__main__':
    
    num_classes = 3
    target_layers = ['layer1', 'layer2', 'layer3', 'layer4']    
    checkpoint_path = 'saved_checkpoints/class_3_train_model-1e-3/val-best-28.pt'
    device_idx = 1
    
    
    '''for target_layer in target_layers:
        print('target layer: {}'.format(target_layer))
        main(target_layer, checkpoint_path, device_idx, num_classes)'''

    threshold_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    for threshold in threshold_list:
        print(f"\nthreshold: {threshold}")
        suppressCMAs(vis_folder='saved_checkpoints/class_3_train_model-1e-3/supress_vis_l1.012l1_1112',
                    cam_root='saved_checkpoints/class_3_train_model-1e-3/cams',
                    threshold=threshold,
                    coe=[5,5,5,10],
                    target_layers=['layer1.0', 'layer1.1', 'layer1.2', 'layer1'])