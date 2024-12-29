#Read BRDF
import numpy as np
import os.path as path

filename = "/mnt/symphony/wen/spectral_brdfs/Merl_BRDF_database/BRDFDatabase/brdfs/alum-bronze.binary"
"""Reads a MERL-type .binary file, containing a densely sampled BRDF
    
Returns a 4-dimensional array (phi_d, theta_h, theta_d, channel)"""
print("Loading MERL-BRDF: ", filename)

f = open(filename, "rb")
dims = np.fromfile(f,np.int32,3)
vals = np.fromfile(f,np.float64,-1)
f.close()
    
BRDFVals = np.swapaxes(np.reshape(vals,(dims[2], dims[1], dims[0], 3),'F'),1,2)
BRDFVals *= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
BRDFVals[BRDFVals<0] = -1
