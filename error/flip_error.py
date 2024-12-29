  ##############################flip error############################################
import flip
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os

if __name__ == '__main__':
    
    samples_alum=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]
    
    rmse_list_alum=[]
    psnr_list_alum=[]
    meanFLIPError_list_alum=[]
    routfolder ="/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/green-metallic-paint"
    test = "/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/white-diffuse-ball/dtu_32.png"
    ref = "/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/white-diffuse-ball/env_gt_100.png"
    for sample in samples_alum:
        test=join(routfolder + r'/1_8_{}_{}_env.png'.format(sample,sample))
        flipErrorMap, meanFLIPError, parameters = flip.evaluate(ref, test, "LDR")
        img1 = plt.imread(test)
     #   img1=img1/255
        img2 = plt.imread(ref)
      #  img2=img2/255
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mse = np.mean((img1 - img2) ** 2)
        rmse = np.sqrt(mse)
        psnr = 10 * np.log10((1.0 ** 2) / mse)
        print("rmse:", rmse)
        print("psnr:", psnr)
        rmse_list_alum.append(rmse)
        psnr_list_alum.append(psnr)
        meanFLIPError_list_alum.append(meanFLIPError)
    
    samples_alum=[32,64,256,512]
    
    rmse_list_alum=[]
    psnr_list_alum=[]
    meanFLIPError_list_alum=[]
    routfolder ="/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/render_data/liu2023learning_supplemental"
    test = "/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/render_data/liu2023learning_supplemental/cooktorrance-ours/32/green-metallic-paint.jpg"
    ref = "/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/render_data/liu2023learning_supplemental/reference/green-metallic-paint.jpg"
    for sample in samples_alum:
        test=join(routfolder + r'/cooktorrance-ours/{}/white-diffuse-bball.jpg'.format(sample))
        flipErrorMap, meanFLIPError, parameters = flip.evaluate(ref, test, "LDR")
        img1 = plt.imread(test)
        img1=img1/255
        img2 = plt.imread(ref)
        img2=img2/255
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mse = np.mean((img1 - img2) ** 2)
        rmse = np.sqrt(mse)
        psnr = 10 * np.log10((1.0 ** 2) / mse)
        print("rmse:", rmse)
        print("psnr:", psnr)
        rmse_list_alum.append(rmse)
        psnr_list_alum.append(psnr)
        meanFLIPError_list_alum.append(meanFLIPError)
    
   # plt.imshow(flipErrorMap)
   # plt.show()
   # plt.axis('off')  # Turn off the axis
  #  plt.savefig('/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/alum-bronze/sample_counts/flip_error_uniform.png', bbox_inches='tight', pad_inches=0)
    
    np.save(join(routfolder, 'rmse.npy'), rmse_list_alum)
    np.save(join(routfolder, 'psnr.npy'), psnr_list_alum)
    np.save(join(routfolder, 'flip.npy'), meanFLIPError_list_alum)
    np.save(join(routfolder, 'samples.npy'), samples_alum)
   # error = np.concatenate(errors, axis=-1)
   # plt.hist(error, bins=20, color='blue', edgecolor='black')
    
    routfolder = '/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/sample_number'
   # samples_alum=[4,8,16,32]
   # samples_green=[1,2,4,8,16,32,64]
  
    rmse_alum=np.load('/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/red-fabric2/23/rmse.npy')
    flip_alum=np.load('/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/red-fabric2/23/flip.npy')
    psnr_dark=np.load('/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/red-fabric2/23/psnr.npy')
   # flip_dark=np.load('/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/dark-specular-fabric/35/flip.npy')
    psnr_green=np.load('/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/render_data/liu2023learning_supplemental/cooktorrance-ours/psnr_red-fabric2.npy')
    rmse_green=np.load('/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/render_data/liu2023learning_supplemental/cooktorrance-ours/rmse_red-fabric2.npy')
    plt.plot(samples_alum,rmse_alum ,linestyle='--', color='r', marker='s',label='green-metallic-paint')
    
    plt.title('RMSE', fontweight='bold')
    plt.xlabel('Samples', fontweight='bold')
    plt.ylabel('RMSE', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend()
    output_path = os.path.join(routfolder, 'all rmse VS samples.png')
    plt.savefig(output_path,bbox_inches='tight', dpi=300)

    plt.plot(samples_alum,flip_alum,linestyle='--', color='r', marker='s',label='alum-bronze')
    plt.title('FLIP', fontweight='bold')
    plt.xlabel('Samples', fontweight='bold')
    plt.ylabel('FLIP', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend()
    output_path = os.path.join(routfolder, 'FLIP VS samples.png')
    plt.savefig(output_path,bbox_inches='tight', dpi=300)