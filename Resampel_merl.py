import sys,os
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)
import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import sys, os
from utils_wen import *
import mitsuba as mi
from merl_mitsuba import *
from optimize_merl_net import *
import torch
from pytorch_merl_data import *
import bsdf



def uniformDiskToSquareConcentric(p):
    r = np.sqrt(p[0]**2 + p[1]**2)
    phi = np.arctan2(p[1], p[0])  # Use numpy's arctan2 for angle calculation

    # Ensure phi is in the range [-pi/4, 7pi/4]
    if phi < -np.pi / 4:
        phi += 2 * np.pi

    # Map phi into different ranges and compute 'a' and 'b'
    if phi < np.pi / 4:
        a = r
        b = phi * a / (np.pi / 4)
    elif phi < 3 * np.pi / 4:
        b = r
        a = -(phi - np.pi / 2) * b / (np.pi / 4)
    elif phi < 5 * np.pi / 4:
        a = -r
        b = (phi - np.pi) * a / (np.pi / 4)
    else:
        b = -r
        a = -(phi - 3 * np.pi / 2) * b / (np.pi / 4)

    # Return the transformed point in the unit square
    return np.array([0.5 * (a + 1), 0.5 * (b + 1)])



def main():
    
    outfolder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/sample_number'
    if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
    test_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/test_pointlight_merl2'

    modelfolder1 = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/torch_poitnlight_ggx/toge_test'
    modelfolder = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/torch_poitnlight_merl/merl_finetune_5pram'
    batchsize = 1
    @dr.wrap_ad(source='torch', target='drjit')
    def render_texture(albedo,alpha, specular,spp=32, seed=1):
      albedo=dr.ravel(albedo)
      alpha=dr.ravel(alpha)
      specular=dr.ravel(specular)
      albedo_pr1=albedo[0]
      albedo_pr2=albedo[1]
      albedo_pr3=albedo[2]
      alpha_pr=alpha[0]
      specular_pr=specular[0]
      scene = load_scene_ggx(load_sensor(spp=spp),albedo_pr1,albedo_pr2,albedo_pr3,alpha_pr,specular_pr)
      #scene = load_scene_ggx(new_emitter,load_sensor(spp=spp),albedo_pr1,albedo_pr2,albedo_pr3,alpha_pr)
      image = mi.render(scene,seed=seed)
        
      return image
    albedo_scale=0.77
    albedo_bias=0.03
    imgListFile1 = 'test_full.txt'
    brdfname = 'brdfnames.txt'    
    testdata=CustomMerlDataset(test_folder, brdfname)
    test_data_loader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=True)

    paraModel = ParaModel1()
    paraModel.classifier0 = nn.Sequential(
            nn.Linear(512, 5),
            nn.Sigmoid()
        )
    paraModel.to(device) 
 #   paraModel.load_state_dict(torch.load('{0}/model_499.pth'.format(modelfolder)))
    paraModel.load_state_dict(torch.load('{0}/bestmodel/model_599_5para_newtog.pth'.format(modelfolder)))
    test_losses = []
    image_test_losses = []
    alpha_test_losses = []
    loss_accum1 = 0
    image_loss1 = 0
    spp = 2
    alpha_loss1 = 0 
    image_loss=[]
    
    j=5
    image,labels1,path,image_reff=testdata.__getbrdf__(j)
    image_re=image.unsqueeze(0)
    albedo_in=albedo_extract(image_re)

    albedo=np.array([albedo_in[0,0],albedo_in[0,1],albedo_in[0,2]])
    brdf_name=merl.parse_name(path)
    print('adaptive brdf_name:',brdf_name)
    outfolder=join(outfolder + r'/{}'.format(brdf_name))
    if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
    image1=image.unsqueeze(0)
   # for j, (image,labels1) in enumerate(test_data_loader):
           #optimizer.zero_grad()
         #   size_t=len(labels1)
    
    image1=image1.permute(0,3,1,2)
    image1=image1.to(device)
    para= paraModel(image1)
          #  albedo1=albedo_scale * albedo1 + albedo_bias # [bias, scale + bias]
   # alpha1=para[:,3].unsqueeze(1)
    #albedo1=para[:,0:3]
    para=para.detach().cpu().numpy()
    alpha1=para[0,3]
    alpha1=np.round(alpha1,6)
    spec=1.0
    spec=para[0,4]
    
    spec=np.round(spec,4)
    labels1=labels1.numpy()[0]
    
    loss_alpha1=np.mean(alpha1-labels1)**2
   
    loss22=[]
        
                  
   # tmp=render_texture(albedo1[0,:],alpha1[0,:],specular[0,:] ,spp=spp, seed=j)
    image_g=image.to(device)

  #  tmp=tmp.detach().cpu().numpy()
  #  plt.imshow(tmp** (1.0 / 2.2))
    image_g=image_g.detach().cpu().numpy()
   # image_g=image_g.to(torch.float32)/image_g.max()
   
    plt.imshow(image_g)
    
    print("Test Done!")

    
    print('Rendering Test data...\n')
    si = dr.zeros(mi.SurfaceInteraction3f)
    xx = np.linspace(0.15,0.85, 6)
    xx=np.round(xx,3)
    yy= np.linspace(0.15,0.85, 6)
    yy=np.round(yy,3)
    xxx, yyy = np.meshgrid(yy, xx)
    xyy = np.dstack((xxx, yyy))

    print('Rendering Test data...\n')
    mybsdf=load_merl_cus(path) 
    alpha=mybsdf.merl_alpha
    alpha=np.round(alpha,6)
  #  spec=1.0-alpha
    measure_brdf=np.zeros([1,8,3,6,6])
    phi_i=np.zeros([1])
   
    theta_i=np.zeros([8])
    
  #  phi_in=np.round(np.linspace(10, 80, 1),5)
    phi_in=np.array([45])
    roughness=0.0
   # theta_in=np.round(np.linspace(0, 90, 16),5)
    theta_in=np.array([5,30,45,55,64,70,80,89])
   
    file_out = join(outfolder + r'/{}_{}_{}_{}_ada.npz'.format(measure_brdf.shape[0],measure_brdf.shape[1],measure_brdf.shape[3],measure_brdf.shape[4]))
    for i in range(1):
        phi_i[i]=phi_in[i]/180.0*dr.pi
       
        for j in range(8):
           
           # params_v[i,j]=[phi_in[i]/180.0*dr.pi,theta_in[j]/180.0*dr.pi]
            theta_i[j]=theta_in[j]/180.0*dr.pi
            
            for ii in range(6):
                  for jj in range(6):
                        si.wi = np.array([dr.sin(theta_in[j]/180.0*dr.pi)*dr.cos(phi_in[i]/180.0*dr.pi),dr.sin(theta_in[j]/180.0*dr.pi)*dr.sin(phi_in[i]/180.0*dr.pi),dr.cos(theta_in[j]/180.0*dr.pi)])
                  
                        sampler_in = mi.Point2f(xyy[ii,jj,0],xyy[ii,jj,1])
                      #  woo = mi.warp.square_to_cosine_hemisphere(sampler_in)
                      #  uu=mi.warp.cosine_hemisphere_to_square(woo) 
                      #  uux=uniformDiskToSquareConcentric(np.array([woo[0,0],woo[1,0]]))
                       # woo=[woo[0,0],woo[1,0],woo[2,0]]
                       
                        woo=mybsdf.imprtance_sample(si.wi, sampler_in,alpha=alpha)
                        woo=[woo[0,0],woo[1,0],woo[2,0]]
                      
                      #  u1,u2=mybsdf.sample_reverse(si.wi,woo)
                      #  si.wi=np.array([woo[0,0],woo[1,0],woo[2,0]])
                        
                        values_ref = mybsdf.eval_m(mi.BSDFContext(), si, woo)
                        measure_brdf[i,j,:,ii,jj]=[values_ref[0,0],values_ref[1,0],values_ref[2,0]]
                       # print(values_ref) 
                       # measure_brdf[i,j,:,ii,jj]=values_ref.numpy()
    np.savez(file_out,phi_in=phi_i,theta_in=theta_i ,diffuse=albedo, roughness=alpha,measure_brdf=measure_brdf)
    print('Measurement done...\n')

if __name__ == '__main__':   
    main()   
