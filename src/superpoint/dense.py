import numpy as np
import cv2
import torch
import torch.nn.functional as F
from superpoint import SuperPointFrontend

class SuperPoint(object):
    def __init__(self, no_grad=True):
        self.frontend = SuperPointFrontend()
        
    def __call__(self, img, keypoints = None):
        H,W = img.shape[-2:]
        outs = self.frontend.net.forward(img)
        semi, coarse_desc = outs[0], outs[1]
        D = coarse_desc.shape[1]

        if(keypoints is not None):
            kps = (keypoints/torch.tensor([[W/2.,H/2.]]).float() - 1.).to(img.device).contiguous().view(1, 1, -1, 2)
            desc = F.grid_sample(coarse_desc, kps).view(D,-1)
            desc /= torch.norm(desc, dim=0)
            desc = desc.T
            return desc

        dense_desc = F.interpolate(coarse_desc, [H+1,W+1],
            mode='bilinear', align_corners=True)
        dense_desc /= torch.norm(dense_desc, dim=1)
        return dense_desc[:,:,:H,:W]
