from gpu_plugin import *
from mitsuba_plugin import *
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)
import drjit as dr
import mitsuba as mi
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mi.set_variant('cuda_ad_rgb')
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
third_party_path = '/mnt/symphony/wen/spectral_brdfs/nerfactor/third_party'  # "third_party" 目录的路径
sys.path.append(third_party_path)
from third_party.xiuminglib import xiuminglib as xm
from utils_wen import *
from optimize_torch_net import *
from utils_wen import *
import bsdf


def main():
    test_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light/test_pointlight_mi'
    outfolder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light/test_resample/test1'
    if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
    aCnt_t = 5
     
    rCnt_t = 5
    imgListFile2 = 'test_full.txt'
    testdata = CustomImageDataset(rootPath=test_folder, imgListFile=imgListFile2, aCnt=aCnt_t, sCnt=aCnt_t, rCnt=rCnt_t)
    jj=11
    image_gt,labels=testdata.__getitem__(jj)
    image=image_gt.unsqueeze(0)
    image=image.permute(0,3,1,2)
    image=image.to(device)
    labels=labels.numpy()
   
    batchsize = 4
    image_height = 256
    image_width = 256
    color_channels = 3
    
    n_vali_batches = 1
    batch_if = True

    model_path = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/HomoNet_pytorch_poitnlight_mi'
    model=HomoNetModel().to(device)
    model.load_state_dict(torch.load('{0}/model.pth'.format(model_path)))
    print(model)
    pr= model(image)
    pr=pr.cpu().detach().numpy()


    file_out = r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light/test_resample/0902ggx_bsdf/4_16_32_32_uniform.npz'
    file_out1 = r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light/test_resample/test1/measure_4_4_4.npz'
    roughness_pr=float(pr[0,1])
    diffuse_pr=float(pr[0,0])
    diffuse_pr=np.array([diffuse_pr,diffuse_pr,diffuse_pr])
  #  mybsdf=load_bsdf(diffuseReflectance=diffuse_pr, specularReflectance=(1-diffuse_pr), al=roughness_pr)
    si = dr.zeros(mi.SurfaceInteraction3f)
 # sampler = mi.load_dict({
 #   'type': 'independent',
#})
#importance sampling
#  sampler.seed(0, wavefront_size=int(1e7))
    x = np.linspace(0, 1, 16)
    y= np.linspace(0, 1, 16)
    xx, yy = np.meshgrid(x, y)
    xy = np.dstack((xx, yy))
    xx = np.linspace(0, 1, 32)
    xx=np.round(xx,3)
    yy= np.linspace(0, 1, 32)
    yy=np.round(yy,3)
    xxx, yyy = np.meshgrid(xx, yy)
    xyy = np.dstack((xxx, yyy))
 # xy=np.reshape(xy,(-1,2))
 # xy=torch.from_numpy(xy).float().to(device)
    #plot_meshgrid(xy, 'Scatter Plot of Mesh Points', 'x', 'y')
 # sampler_out = mi.Point2f(xy[...,0],xy[...,1])
 # wo = mi.warp.square_to_cosine_hemisphere(sampler_out)
    
  #xyy=np.reshape(xyy,(-1,2))
  #xyy=torch.from_numpy(xyy).float().to(device)
  #sampler_in = mi.Point2f(xy[...,0],xy[...,1])
  #si.wi = mi.warp.square_to_cosine_hemisphere(sampler_in)
  #values_ref = mybsdf.eval(mi.BSDFContext(), si, wo)
  #values_ref=values_ref.numpy()
    mybsdf=load_bsdf_cus(diffuseReflectance=diffuse_pr, specularReflectance=(1-diffuse_pr), al=roughness_pr) 
    measure_brdf=np.zeros([4,16,3,32,32])
 #   params_v = np.zeros([1,8,2])
    theta_i=np.zeros([16])
    phi_i=np.zeros([4])
    phi_in=np.round(np.linspace(0, 90, 4),5)
    theta_in=np.round(np.linspace(0, 90, 16),5)
    for i in range(4):
        phi_i[i]=phi_in[i]/180.0*dr.pi
        for j in range(16):
            
           # params_v[i,j]=[phi_in[i]/180.0*dr.pi,theta_in[j]/180.0*dr.pi]
            theta_i[j]=theta_in[j]/180.0*dr.pi
            
            for ii in range(32):
                  for jj in range(32):
                        si.wi = np.array([dr.sin(theta_in[j]/180.0*dr.pi)*dr.cos(phi_in[i]/180.0*dr.pi),dr.sin(theta_in[j]/180.0*dr.pi)*dr.sin(phi_in[i]/180.0*dr.pi),dr.cos(theta_in[j]/180.0*dr.pi)])
                     #   sampler_in = mi.Point2f(xy[i,j,0],xy[i,j,1])
                     #   si.wi = mi.warp.square_to_cosine_hemisphere(sampler_in)
                        sample1=mi.Float(xyy[ii,jj,0])
                        sampler_o = mi.Point2f(xyy[ii,jj,0],xyy[ii,jj,1])
                      #  print(sampler_o)
                        woo = mi.warp.square_to_cosine_hemisphere(sampler_o)
                       # bs,value=mybsdf.sample(mi.BSDFContext(), si, sample1,sampler_o)
                       # wo=[bs.wo[0,0],bs.wo[1,0],bs.wo[2,0]]
                       # woo=mybsdf.importance_sample(si.wi, sampler_o)
                        wo=[woo[0,0],woo[1,0],woo[2,0]]
                      #  uu=mybsdf.sample_reverse(si.wi,wo)
                     #   print(uu)
                        values_ref = mybsdf.eval(mi.BSDFContext(), si, wo)
                        measure_brdf[i,j,:,ii,jj]=[values_ref[0],values_ref[0],values_ref[0]]
                       # print(values_ref) 
                       # measure_brdf[i,j,:,ii,jj]=values_ref.numpy()
    np.savez(file_out,phi_in=phi_i,theta_in=theta_i , diffuse=diffuse_pr, roughness=roughness_pr,measure_brdf=measure_brdf)
    print('Measurement done...\n')
    1
    1
    1
    
    mybsdf=load_bsdf_cus(diffuseReflectance=diffuse_pr, specularReflectance=(1-diffuse_pr), al=roughness_pr) 
    measure_brdf_py=np.zeros([1,8,3,4,4])
    params_v = np.zeros([1,4,2])
    #plot_meshgrid(xyy, 'Scatter Plot of Mesh Points', 'x', 'y')
    phi_in=[90]
    theta_in=np.round(np.linspace(0, 90, 8))
  #  theta_in=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180]

  #  in_dir=np.zeros([8,8,2])
  #  o_dir=np.zeros([32,32,2])
    for i in range(1):
        for j in range(4):
       #     sampler_in = mi.Point2f(xyy[i,j,0],xyy[i,j,1])
       #     si.wi = mi.warp.square_to_cosine_hemisphere(sampler_in)
            si.wi = np.array([dr.sin(theta_in[j]/180.0*dr.pi)*dr.cos(phi_in[i]/180.0*dr.pi),dr.sin(theta_in[j]/180.0*dr.pi)*dr.sin(phi_in[i]/180.0*dr.pi),dr.cos(theta_in[j]/180.0*dr.pi)])
       #     theta_in = dr.safe_acos(si.wi[2])*180.0/dr.pi
       #     phi_in = dr.atan2(si.wi[1],si.wi[0])*180.0/dr.pi
      #      in_dir[i,j,:]=[phi_in[0],theta_in[0]]
            params_v[i,j]=[phi_in[i]/180.0*dr.pi,theta_in[j]/180.0*dr.pi]
           
            
            for ii in range(4):
                  for jj in range(4):  
                         sampler_o = mi.Point2f(xy[ii,jj,0],xy[ii,jj,1])
                      #   wo = mi.warp.square_to_cosine_hemisphere(sampler_o)
                      #   theta_o = dr.safe_acos(wo[2])*180.0/dr.pi
                      #   phi_o= dr.atan2(wo[1],wo[0])*180.0/dr.pi
                      #   o_dir[ii,jj,:]=[phi_o[0],theta_o[0]]
                         bs=mybsdf.importance_sample(mi.BSDFContext(), si, sampler_o)
                         wo=[bs.wo[0,0],bs.wo[1,0],bs.wo[2,0]]
                     #   wo = np.array([dr.sin(theta_o[jj]/180.0*dr.pi)*dr.cos(phi_o[ii]/180.0*dr.pi),dr.sin(theta_o[jj]/180.0*dr.pi)*dr.sin(phi_o[ii]/180.0*dr.pi),dr.cos(theta_o[jj]/180.0*dr.pi)])
                         values_ref = mybsdf.eval(mi.BSDFContext(), si, wo)
                         measure_brdf_py[i,j,:,ii,jj]=[values_ref[0,0],values_ref[1,0],values_ref[2,0]]
    np.savez(file_out1, measure_brdf=measure_brdf_py,params_v=params_v, diffuse=diffuse_pr, roughness=roughness_pr)
    print('Measurement done...\n')
  #  plot_tensor(measure_brdf_py)
   
  #  data = np.load('measurements_folder/measurements.npz')
 #   measure_brdf = data['measure_brdf']
   # measure_wiwo = data['measure_wiwo']  


def important_sampling(sample2,wi,alpha=0.1):
    #importance sampling with ward brdf
    chooseSpecular = True
    wi=wi/np.linalg.norm(wi)    
    wo=wi
  
    if(chooseSpecular):
           # specular sampling
               phiH = 2.0*dr.pi*sample2[1]
               cosPhiH=dr.cos(phiH)
             #  sinPhiH=dr.safe_sqrt(1-cosPhiH*cosPhiH)
               thetaH = dr.atan(alpha*dr.safe_sqrt(-dr.log(dr.maximum(sample2[0],1e-8))))   
               
               H=np.array([dr.sin(thetaH)*dr.cos(phiH),dr.sin(thetaH)*dr.sin(phiH),dr.cos(thetaH)]) 
               H=np.array([H[0,0],H[1,0],H[2,0]])
               wii=np.array([wi[0,0],wi[1,0],wi[2,0]]) 
               woo = 2.0*np.dot(wii,H)*H-wii
               if(woo[2]<=0):
                   woo=np.array([0,0,0])
                   
    return wo   

if __name__ == '__main__':
    main()
