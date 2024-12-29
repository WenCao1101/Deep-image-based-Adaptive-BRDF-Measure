import os
import sys

import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset
import numpy as np
third_party_path = '/mnt/symphony/wen/spectral_brdfs/nerfactor/third_party'  # "third_party" 目录的路径
sys.path.append(third_party_path)
from third_party.xiuminglib import xiuminglib as xm
class CustomMerlDataset(Dataset):
    def __init__(self, rootPath, brdfname):
        """Initialize the dataset."""
        with open(rootPath + r'/{}'.format(brdfname), 'r') as f:
            self.brdfnames = f.read().strip().split('\n')
       
        self.brdfnum = len(self.brdfnames)
        self.brdfCube = np.loadtxt(rootPath + r'/brdfcube.txt').reshape((self.brdfnum, 2))
        self.rootPath = rootPath
        

    def __len__(self):
        return len(self.brdfnames)

    def __getitem__(self, idx):
        alpha = self.brdfCube[idx, 1:]
        ind_brdfs = self.brdfCube[idx, 0]
       
           #s_idx = self.dataList[idx]
        brdfname = self.brdfnames[idx]
       # image= xm.io.exr.read(self.rootPath + r'/{}_{}.exr'.format(idx,brdfname)) 
        image=mpimg.imread(self.rootPath + r'/{}_{}.png'.format(idx,brdfname))
        image = torch.from_numpy(image)
        image=image/torch.max(image)
       # brdf = z_brdfs[idx].reshape((3))
   
        label = torch.from_numpy(alpha)
        return image, label
    
    def __getbrdf__(self, idx):
        alpha = self.brdfCube[idx, 1:]
        ind_brdfs = self.brdfCube[idx, 0]
        test_root=r'/mnt/symphony/wen/spectral_brdfs/Merl_BRDF_database/BRDFDatabase/brdfs_test'
           #s_idx = self.dataList[idx]
        brdfname = self.brdfnames[idx]
      #  image= xm.io.exr.read(self.rootPath + r'/{}_{}.exr'.format(idx,brdfname)) 
        image_ref=mpimg.imread(self.rootPath + r'/{}_{}.png'.format(idx,brdfname))
        path=test_root + r'/{}.binary'.format(brdfname)
        
        image = torch.from_numpy(image_ref)
        image=image/torch.max(image)
       # brdf = z_brdfs[idx].reshape((3))
   
        label = torch.from_numpy(alpha)
        return image, label,path,image_ref
