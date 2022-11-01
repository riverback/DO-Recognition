from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from cams.basic import _BaseWrapper


class GradCAM(_BaseWrapper):

    def __init__(self, extractor, classifier, candidate_layers=None):
        super(GradCAM, self).__init__(extractor, classifier)
        self.fmap_pool = {} # feature map dict
        self.grad_pool = {} # gradients dict
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)


        return gcam


class BagCAMs(_BaseWrapper):

    def __init__(self, extractor, classifier):
        self.fmap_pool = {}
        self.fmap_pool_in = {}
        self.grad_pool = {}
        self.grad_pool_in = {}
    
        super(BagCAMs, self).__init__(extractor, classifier)
        
        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

                if isinstance(module, nn.ReLU):
                    return (F.relu(grad_in[0]),)
                        
            return backward_hook
        
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()
                self.fmap_pool_in[key] = input[0].detach()

            return forward_hook

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_forward_hook(save_fmaps(module[0])))
            self.handlers.append(module[1].register_backward_hook(save_grads(module[0])))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        self.ids = ids
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        ##
        self.phi = torch.zeros(self.logits.shape[0], 1).cuda()
        for i in range(0, self.logits.shape[0]):
            self.phi[i] = self.logits[i, ids[i]]
        # self.logits[:, ids]
        ##
        #print(one_hot)
        self.logits.log().backward(gradient=one_hot, retain_graph=True)

    def generate(self, target_layer):

        ##obtain the gradient
        grads = self._find(self.grad_pool, target_layer)

        ##obtain the feature map
        features = self._find(self.fmap_pool, target_layer)

        ##Calculate BagCAMs
        term_2 = grads*features
        # term_1 = grads*features + 1 # old old版本相当于先对所有分类器求平均，在与特征相乘的时候只用了对应位置的分类器
        # term_1 = F.adaptive_avg_pool2d(term_1, 1) #sum_m # old
        
        #grads: B * C * H * W # 相当于在特征图的每个Channel上有 H * W 个分类器 如果先做了avg_pool 那么相当于每个通道
        #feature: B * C * H * W
        
        # term_1 = F.adaptive_avg_pool2d()
        
        term_1 = F.adaptive_avg_pool2d( (F.adaptive_avg_pool2d(grads, 1)*features + 1) , 1) # 尝试改成新版本，之前是只用了对应位置的分类器，现在希望对分类器加一个权重相加，目前写在这里的相当于是平均加权
        bagcams = torch.relu(torch.mul(term_1, term_2)).sum(dim=1, keepdim=True) #sum_c

        

        ##Upsampling to Original Size of Images
        bagcams = F.interpolate(
            bagcams, self.image_shape, mode="bilinear", align_corners=False
        )
        
        ##Normalized the localization Maps
        B, C, H, W = bagcams.shape
        bagcams = bagcams.view(B, -1)
        bagcams -= bagcams.min(dim=1, keepdim=True)[0]
        bagcams /= bagcams.max(dim=1, keepdim=True)[0]
        bagcams = bagcams.view(B, C, H, W)

        return bagcams