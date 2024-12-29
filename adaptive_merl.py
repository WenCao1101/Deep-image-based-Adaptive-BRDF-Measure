import sys,os
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)
import mitsuba as mi
import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import sys, os
import torch
from merl_mitsuba import *
from pytorch_merl_data import *
from optimize_merl_net import *
from Merl_measure import *
import cv2
import flip





def main():
    outfolder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/merl_model_adaptive'
    if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
    test_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/test_pointlight_merl2'

    modelfolder1 = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/torch_poitnlight_ggx/toge_test'
    modelfolder = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/torch_poitnlight_merl/merl_finetune'
    batchsize = 1
    @dr.wrap_ad(source='torch', target='drjit')
    def render_texture(path,spp=2, seed=1):

      scene = load_scene_measure(load_sensor_meas(spp=spp),path)
      #scene = load_scene_ggx(new_emitter,load_sensor(spp=spp),albedo_pr1,albedo_pr2,albedo_pr3,alpha_pr)
      image = mi.render(scene,seed=seed) 
      return image
    
    brdfname = 'brdfnames.txt'    
    testdata=CustomMerlDataset(test_folder, brdfname)
    test_data_loader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=True)

    paraModel = ParaModel().to(device)
    paraModel.classifier0 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        ) 
    paraModel.load_state_dict(torch.load('{0}/model_49_ll2.pth'.format(modelfolder)))
    j=0
    image,labels1,path_binary,image_ref=testdata.__getbrdf__(j)
    brdf_name=merl.parse_name(path_binary)
    print('adaptive brdf_name:',brdf_name)
    outfolder=join(outfolder + r'/{}'.format(brdf_name))
    if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
    
    image1=image.unsqueeze(0)
    image1=image1.permute(0,3,1,2)
    image1=image1.to(device)
    paraModel.to(device)
    para= paraModel(image1)
    alpha1=para
    alpha1=alpha1[0,0].detach().cpu().numpy()
    labels1=labels1[0].numpy()
    albedo1=np.array([0.77,0.77,0.77])

    loss_alpha1=np.mean(alpha1-labels1)**2
    loss_alpha1=np.round(loss_alpha1,5)
    loss_alpha=[]
    loss_alpha.append(loss_alpha1.item())
    npz_file = join(outfolder + r'/ada.npz')
    opt = mi.ad.Adam(lr=0.05)
    iteration_count = 50
    opt_size = 8
    opt = opt_size
    errors = []
    plt.imshow(image_ref**(1/2.2))
    for it in range(17,32):
        it=int(it)
        sample_merl(4,16,it,it,path_binary,labels1,albedo1,npz_file)
        tmp1=render_texture(npz_file,spp=2, seed=1)
        mi.util.write_bitmap(join(outfolder + r'/adaptive_{}.png'.format(it)),tmp1)
        tmp11=plt.imread(join(outfolder + r'/adaptive_{}.png'.format(it)))
      
        tmp11 =cv2.resize(tmp11, (256,256), interpolation=cv2.INTER_AREA)
        flipErrorMap, meanFLIPError, parameters = flip.evaluate(image_ref, tmp11, "LDR")
        plt.imshow(tmp11**(1/2.2))
        
       # loss=np.mean((tmp1-image_ref)**2)    
        print('loss_image:',meanFLIPError)
        
        errors.append(meanFLIPError)
        if(meanFLIPError<0.01):
            break
    print('apdaptive sample:',opt)
    print('loss_image:',meanFLIPError)

        

    
    
def mse(image, image_ref):
    return dr.mean(dr.sqr(image - image_ref))

def sample_merl(phi_size,theta_size,u1_size,u2_size,path,alpha,albedo_pr,file_out):
    mybsdf=load_merl_cus(path) 
    si = dr.zeros(mi.SurfaceInteraction3f)
    xx = np.linspace(0, 1, u1_size)
    xx=np.round(xx,3)
    yy= np.linspace(0, 1, u2_size)
    yy=np.round(yy,3)
    xxx, yyy = np.meshgrid(xx, yy)
    xyy = np.dstack((xxx, yyy))
    measure_brdf=np.zeros([phi_size,theta_size,3,u1_size,u2_size])
    phi_i=np.zeros([phi_size])
    theta_i=np.zeros([theta_size])
    
    phi_in=np.round(np.linspace(1, 89, phi_size),5)
  
    theta_in=np.round(np.linspace(1, 89, theta_size),5)

   
    for i in range(phi_size):
        phi_i[i]=phi_in[i]/180.0*dr.pi
        for j in range(theta_size):
           
           # params_v[i,j]=[phi_in[i]/180.0*dr.pi,theta_in[j]/180.0*dr.pi]
            theta_i[j]=theta_in[j]/180.0*dr.pi
            
            for ii in range(u1_size):
                  for jj in range(u2_size):
                        si.wi = np.array([dr.sin(theta_in[j]/180.0*dr.pi)*dr.cos(phi_in[i]/180.0*dr.pi),dr.sin(theta_in[j]/180.0*dr.pi)*dr.sin(phi_in[i]/180.0*dr.pi),dr.cos(theta_in[j]/180.0*dr.pi)])
                  
                        sampler_in = mi.Point2f(xyy[ii,jj,0],xyy[ii,jj,1])
                      #  woo = mi.warp.square_to_cosine_hemisphere(sampler_in)
                      #  uu=mi.warp.cosine_hemisphere_to_square(woo) 
                      #  uux=uniformDiskToSquareConcentric(np.array([woo[0,0],woo[1,0]]))
                       # woo=[woo[0,0],woo[1,0],woo[2,0]]
                       
                        woo=mybsdf.imprtance_sample(si.wi, sampler_in,alpha=alpha)
                        woo=[woo[0,0],woo[1,0],woo[2,0]]
                      
                        u1,u2=mybsdf.sample_reverse(si.wi,woo)
                      #  si.wi=np.array([woo[0,0],woo[1,0],woo[2,0]])
                        
                        values_ref = mybsdf.eval_m(mi.BSDFContext(), si, woo)
                        measure_brdf[i,j,:,ii,jj]=[values_ref[0,0],values_ref[1,0],values_ref[2,0]]
                       # print(values_ref) 
                       # measure_brdf[i,j,:,ii,jj]=values_ref.numpy()
    np.savez(file_out,phi_in=phi_i,theta_in=theta_i , diffuse=albedo_pr, roughness=alpha,measure_brdf=measure_brdf)





if __name__ == '__main__':   
    main()  
