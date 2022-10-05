from bdb import Breakpoint
import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd
import torch
import torch.nn as nn
import munch

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def generate_vis(p, img):
    # All the input should be numpy.array 
    # img should be 0-255 uint8

    C = 1
    H, W = p.shape

    prob = p

    prob[prob<=0] = 1e-7

    def ColorCAM(prob, img):
        C = 1
        H, W = prob.shape
        colorlist = []
        colorlist.append(color_pro(prob,img=img,mode='chw'))
        CAM = np.array(colorlist)/255.0
        return CAM

    #print(prob.shape, img.shape)
    CAM = ColorCAM(prob, img)
    #print(CAM.shape)
    return CAM[0, :, :, :]

def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))	
	return color

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, extractor, classifier, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None, wsol_method='cam'):
        self.extractor = extractor
        self.classifier = classifier
        self.extractor.eval()
        self.classifier.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        self.wsol_method = wsol_method

    def train_compute_and_cams(self):
        print("computing cams for erase.")
        assert self.split == 'train', f"self.split == 'train' expected but got {self.split}"
        for images, targets, image_ids in self.loader:
            ### Erase
            return_mask = mch()
            
            image_size = images.shape[2:]
            images = images.cuda()
        
            pixel_features = self.extractor(images)
            cams = self.classifier(pixel_features, targets, return_cam=True) # 0-1 mask
            
            cams = t2n(cams)

            image_features = nn.AdaptiveAvgPool2d(1)(pixel_features)
            logits = self.classifier(image_features)

            predicts = torch.argmax(logits, dim=1)
            predicts = t2n(predicts.view(predicts.shape[0], predicts.shape[1]))
            targets = t2n(targets)

            for cam, target, predict, image, image_id in zip(cams, targets, predicts, images, image_ids):
                
                cam_resized = cv2.resize(cam, image_size,
                                        interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                # print(type(cam_normalized))
                if image_id in return_mask.keys():
                    return_mask[image_id] = normalize_scoremap(return_mask[image_id] + cam_normalized)
                else:
                    return_mask[image_id] = cam_normalized

        
        return return_mask



    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        for images, targets, image_ids in self.loader:
            ### Erase
            return_mask = mch()
            
            image_size = images.shape[2:]
            images = images.cuda()
        
            pixel_features = self.extractor(images)
            cams = self.classifier(pixel_features, targets, return_cam=True) # 0-1 mask
            
            '''print("查看这里是否是0-1的mask")
            print(type(cams))
            print(cams.min(), cams.max())
            raise Breakpoint'''

            cams = t2n(cams)

            image_features = nn.AdaptiveAvgPool2d(1)(pixel_features)
            logits = self.classifier(image_features)

            predicts = torch.argmax(logits, dim=1)
            predicts = t2n(predicts.view(predicts.shape[0], predicts.shape[1]))
            targets = t2n(targets)

            for cam, target, predict, image, image_id in zip(cams, targets, predicts, images, image_ids):
                ### debug
                # print(image.shape)
                # print(image_size)
                
                cam_resized = cv2.resize(cam, image_size,
                                        interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                # print(type(cam_normalized))

                ###
                if self.split == 'trian':
                    if image_id in self.mask.keys():
                        return_mask[image_id] += cam_normalized
                        cam_normalized = self.mask[image_id]
                    else:
                        return_mask[image_id] = cam_normalized

                    continue
                ### debug
                # print(cam_normalized.shape) # ndarray
                # print(cam_normalized.max(), cam_normalized.min())

                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    pred_path = ospj(self.log_folder, 'predicts', image_id)
                    gt_path = ospj(self.log_folder, 'image_gts', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    if not os.path.exists(ospd(pred_path)):
                        os.makedirs(ospd(pred_path))
                    if not os.path.exists(ospd(gt_path)):
                        os.makedirs(ospd(gt_path))
                    np.save(ospj(cam_path), cam_normalized)
                    np.save(ospj(pred_path), predict)
                    np.save(ospj(gt_path), target)
                    
                    '''###
                    vis_image = image.cpu().data * np.array(_IMAGENET_STDDEV).reshape([3, 1, 1]) + np.array(_IMAGENET_MEAN).reshape([3, 1, 1])
                    vis_image = np.int64(vis_image * 255)
                    vis_image[vis_image > 255] = 255
                    vis_image[vis_image < 0] = 0
                    vis_image = np.uint8(vis_image)
                    vis_path = ospj(self.log_folder, 'vis', image_id)
                    if not os.path.exists(ospd(vis_path)):
                        os.makedirs(ospd(vis_path))
                    plt.imsave(ospj(vis_path)+".png", generate_vis(cam_normalized, vis_image).transpose(1, 2, 0))
                    ###'''
                    
                if self.dataset_name == "OpenImages" or self.dataset_name == "ISIC":
                    self.evaluator_mask.accumulate(cam_normalized, image_id)
                elif self.dataset_name == "CUB":
                    if self.split == "test":
                        self.evaluator_mask.accumulate(cam_normalized, image_id)
                    self.evaluator_boxes.accumulate(cam_normalized, image_id)
                else:
                    self.evaluator_boxes.accumulate(cam_normalized, image_id)

            ###
            if self.split == 'train':
                return return_mask

            #np.save(ospj(cam_path), cam_normalized)
            performance={}
            if self.dataset_name == "OpenImages" or self.dataset_name == "ISIC":
                pxap, iou = self.evaluator_mask.compute()
                performance['pxap'] = pxap
                performance['iou'] = iou

            elif self.dataset_name == "CUB":
                if self.split == "test":
                    pxap, iou = self.evaluator_mask.compute()
                    performance['pxap'] = pxap
                    performance['iou'] = iou

                gt_known = self.evaluator_boxes.compute()
                top_1 = self.evaluator_boxes.compute_top1()
                performance['gt_known'] = gt_known
                performance['top_1'] = top_1
            else:
                gt_known = self.evaluator_boxes.compute()
                top_1 = self.evaluator_boxes.compute_top1()
                performance['gt_known'] = gt_known
                performance['top_1'] = top_1

        # print(self.mask.keys())


        return performance 

