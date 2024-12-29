
from os.path import join
import sys,os
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)
import mitsuba as mi
#from merl_mitsuba import *
mi.set_variant('cuda_ad_rgb')
from pytorch_merl_data import *
from coordinateFunctions import *
from Render_ggx_data import *
from utils_wen import *
from matplotlib import pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimize_ggx_net import *
from torchvision import transforms
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ParaModel1(nn.Module):
    def __init__(self):
        super(ParaModel1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(), 
        )
        
        self.classifier0 = nn.Sequential(
            nn.Linear(512, 5),
            nn.Sigmoid()
        )
        
       
    def forward(self, x):
        x = self.features(x)
        x0 = self.classifier0(x)
 
        return x0


# Example usage
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)\
   
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
def load_rusink():
    x=np.linspace(0, 90, num=90, dtype=int)
    y=np.linspace(0, 90, num=90, dtype=int)
    z=np.linspace(0, 180, num=180, dtype=int)
    xyz=np.array([np.arange(90), np.arange(90), np.arange(180)], dtype=object)
    rusink =MERLToRusink(xyz)
    return rusink    

def main():
   
   train_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/train_pointlight_merl2'
   test_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/test_pointlight_merl2'
   outfolder = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/torch_poitnlight_merl/merl_finetune2_5pram'
   modelfolder = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/torch_poitnlight_ggx/5_param'
   if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
   image_height = 256
   image_width = 256
   color_channels = 3
  
   batchsize = 1
   @dr.wrap_ad(source='torch', target='drjit')
   def render_texture(albedo,alpha,specular=1.0 ,spp=2, seed=1):
      albedo=dr.ravel(albedo)
      alpha=dr.ravel(alpha)
      specular=dr.ravel(specular)
      albedo_pr1=albedo[0]
      albedo_pr2=albedo[1]
      albedo_pr3=albedo[2]
      alpha_pr=alpha[0]
      specular_pr=specular[0]
      scene = load_scene_merl_test(load_sensor(spp=spp),albedo_pr1,albedo_pr2,albedo_pr3,alpha_pr,specular_pr)
      #scene = load_scene_ggx(new_emitter,load_sensor(spp=spp),albedo_pr1,albedo_pr2,albedo_pr3,alpha_pr)
      image = mi.render(scene,seed=seed)
        
      return image
   imgListFile = 'train_full.txt'
   brdfname = 'brdfnames.txt'    
   traindata=CustomMerlDataset(train_folder, brdfname)
   img,label=traindata.__getitem__(3)
   train_data_loader = torch.utils.data.DataLoader(traindata, batch_size=batchsize, shuffle=True)

   imgListFile1 = 'test_full.txt'
   testdata=CustomMerlDataset(test_folder, brdfname)
   test_data_loader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=True)
   
   #data,label=traindata.__getitem__(3)
 #  data,label=traindata._load_data()
 #  imgListFile2 = 'test_full.txt'
 #  testdata = Dataset(batchsize,test_folder, brdfname,image_height, image_width, color_channels)
   
 #  datapipe_test = testdata.build_pipeline(batch_if=batchsize) 
 #  vali_batches = datapipe_test.take(n_vali_batches)
   model = ParaModel1().to(device)
   model.load_state_dict(torch.load('{0}/ParaModel_300_norm.pth'.format(modelfolder)))
   
   for param in model.features.parameters():
            param.requires_grad = False
 #  for idx in range(24, len(model.features)):
 #   layer = model.features[idx]
 #   if hasattr(layer, 'weight') and layer.weight.requires_grad is False:
 #       for param in layer.parameters():
 #           param.requires_grad = True                
 #  model.classifier0 = nn.Sequential(
 #           nn.Linear(512, 2),
 #           nn.Sigmoid()
 #       ) 
         
   print(model)
   
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
   for param in model.parameters():
    if torch.isnan(param).any():
        print("模型参数包含NaNs")
    if torch.isinf(param).any():
        print("模型参数包含Infs")

   loss_fn = nn.L1Loss()
   mse_loss = nn.MSELoss()

# Optimization hyper-parameters
   iteration_count =1000
   spp = 2
   
   #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
  
   scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
   
   model.train(mode=True).to(device)
   #pr=torch.tensor([0.5,0.5])
   #rendered_img = render_texture(pr.numpy(), spp=spp, seed=1)  
   train_losses = []
   image_train_losses = []
   alpha_train_losses = []
   test_losses = []
   image_test_losses = []
   alpha_test_losses = []
  
   best_val_loss = 0.0
   patience = 100 # 提前停止的容忍度
   patience_counter = 0
   specular_in=0.6
 #  specular.requires_grad = True
   for i in range(iteration_count):
         print(f"Epoch {i+1}\n-------------------------------")
        # trainingLog = open('{0}/trainingLog_{1}.txt'.format(outfolder, i), 'w')
         loss_accum = 0
         image_loss = 0
         diff_loss = 0  
         alpha_loss = 0
         
         for j, (image,labels) in enumerate(train_data_loader):
       #      plt.imshow(image[0].numpy())
             albedo_int=albedo_extract(image)
             albedo_int=torch.tensor(albedo_int).to(device)
             size_tr=len(labels)
             optimizer.zero_grad()
             image2=image.permute(0,3,1,2)
             
             image2=image2.to(device)
             mask =(image2[0] != 0).any(dim=0).unsqueeze(0).expand_as(image2[0])
             para= model(image2)
            # alpha=para
             albedo=para[:,0:3]
             alpha=para[:,3].unsqueeze(1)
             specualr=para[:,4].unsqueeze(1)
        
             
             labels=labels.to(device).float()
           #  specular_in=1.0-labels
             specular_in=torch.tensor(specular_in).to(device)
            # loss=loss_fn(torch.log(alpha+1), torch.log(labels+1))
             loss_alpha=mse_loss(alpha, labels)
           #  loss=loss_fn(alpha, labels)
             loss_albedo=mse_loss(albedo, albedo_int)
             loss_specular=mse_loss(specualr, specular_in)
             loss2=[]
             
            # log_tmp1=torch.zeros((size_tr,3,256,256),dtype=torch.float32).to(device)
            # log_img1=torch.zeros((size_tr,3,256,256),dtype=torch.float32).to(device)   \
             image=image.to(device)
             for ii in range(size_tr):
                  
                  tmp1=render_texture(albedo[ii,:],alpha[ii,:],specualr[ii,:] ,spp=spp, seed=ii*len(train_data_loader)+j)
                 
                  loss2.append(loss_fn(torch.abs((((((tmp1/tmp1.max()))*255).to(torch.uint8)).to(torch.float32) / 255).to(device)), image[ii]))
                 # loss2.append(loss_fn(torch.abs((((((tmp/tmp.max())** (1.0 / 2.2))*255).to(torch.uint8)).to(torch.float32) / 255).to(device)), torch.abs( (((((image[ii]/image[ii].max())** (1.0 / 2.2))*255).to(torch.uint8)).to(torch.float32) / 255).to(device))))

            # loss2=loss_fn(torch.abs(torch.log(rendered_img.pow(1.0/2.2)+eps)), torch.abs(torch.log(image.pow(1.0/2.2)+eps)))
             loss22=torch.mean(torch.tensor(loss2,dtype=torch.float32))
             loss=loss_alpha*100+loss22+loss_specular*50
           #   loss=loss2*100
            # loss=torch.mean(torch.tensor(loss2,dtype=torch.float32))*100
         
             loss.backward() 
             loss_accum += loss.item()
             image_loss += torch.mean(torch.tensor(loss2,dtype=torch.float32)).item()*100
            # diff_loss += loss_diff.item()*100
             alpha_loss += loss_alpha.item()*100
             #print(f'image{j}',end='\r')
            # print(f'Images {j+1}/{len(train_data_loader)} ,Total Loss: {loss.item}', end='\r')
            # print(f'Images {j+1}/{len(train_data_loader)} ,Diffuse Loss: {loss_diff.item()*100}', end='\r')
           #  print(f'Images {j+1}/{len(train_data_loader)} ,Rough Loss: {loss_rough.item()*100}', end='\r')
           #  print(f'Images {j+1}/{len(train_data_loader)} ,Image Loss: {torch.mean(torch.tensor(loss2,dtype=torch.float32)).item()*100}', end='\r')
             optimizer.step()
         
      #   trainingLog.close()
         train_losses.append(loss_accum/len(train_data_loader))
         image_train_losses.append(image_loss/len(train_data_loader))
        # diff_train_losses.append(diff_loss/len(train_data_loader))
         alpha_train_losses.append(alpha_loss/len(train_data_loader))
         print(f'Iteration {i+1}/{iteration_count}, Total Loss: {train_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Image Loss: {image_train_losses[-1]}')
       #  print(f'Iteration {i+1}/{iteration_count}, Diffuse Loss: {diff_train_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Alpha Loss: {alpha_train_losses[-1]}', end='\r')
         if (i +1)% 20 == 0:     
            scheduler.step()
         model.eval() 
         loss_accum1 = 0
         image_loss1 = 0

         alpha_loss1 = 0 
         for j, (image1,labels1) in enumerate(test_data_loader):
            albedo1_in=albedo_extract(image1)
            albedo1_in=torch.tensor(albedo1_in).to(device)
         #   optimizer.zero_grad()
            size_t=len(labels1)
            image11=image1.permute(0,3,1,2)
            image11=image11.to(device)
            param= model(image11)
            alpha1=param[:,3].unsqueeze(1)
            albedo1=param[:,0:3]
            specualr1=param[:,4].unsqueeze(1)
          #  albedo1=albedo_scale * albedo1 + albedo_bias # [bias, scale + bias]
          #  alpha1=param[:,3].unsqueeze(1)
             #pr=pr.clamp(0,1)
    
            labels1=labels1.to(device)

           # loss_alpha1=loss_fn(torch.log(alpha1+1), torch.log(labels1+1))
            loss_alpha1=mse_loss(alpha1, labels1)
            loss_albedo1=mse_loss(albedo1, albedo1_in)
           #  loss_alpha1=loss_fn(alpha1, labels1)
            image_g1=image1.to(device)
            loss22=[]
            for ii in range(size_t):
                
                  tmp11=render_texture(albedo1[ii,:],alpha1[ii,:],specualr1[ii,:] ,spp=spp, seed=ii*len(train_data_loader)+j)
                 
                  
                #  tmp1=tmp1.permute(2,0,1)
                #  tmp11=tmp11.to(torch.float32)/tmp11.max()
               # tmp=tmp.permute(2,0,1)
                  tmp111=torch.abs((((((tmp11/tmp11.max())** (1.0 / 2.2))*255).to(torch.uint8)).to(torch.float32) / 255).to(device))
                 
                  loss22.append(loss_fn(tmp111, image_g1))
            loss33=torch.mean(torch.tensor(loss22,dtype=torch.float32)) 
            loss11=loss_alpha1*100+loss33*10+loss_albedo1*50  
            loss_accum1 += loss11.item()
            image_loss1 += torch.mean(torch.tensor(loss22,dtype=torch.float32)).item()*100
    
            
            alpha_loss1 += loss_alpha1.item()*100
         test_losses.append(loss_accum1/len(test_data_loader))
         image_test_losses.append(image_loss1/len(test_data_loader))
        # diff_train_losses.append(diff_loss/len(train_data_loader))
         alpha_test_losses.append(alpha_loss1/len(test_data_loader))
         print(f'Iteration {i+1}/{iteration_count}, Total Test Loss: {test_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Image Test Loss: {image_test_losses[-1]}')
       #  print(f'Iteration {i+1}/{iteration_count}, Diffuse Loss: {diff_train_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Alpha Test Loss: {alpha_test_losses[-1]}', end='\r')   
         if (alpha_test_losses[-1]< best_val_loss):
             best_val_loss = alpha_test_losses[-1]
          #   patience_counter = 0
        # 可选：保存最优模型
             torch.save(model.state_dict(), '{0}/best_model_{1}_l2.pth'.format(outfolder, i))
             print("提前停止")
             break
        
   

   torch.save(model.state_dict(), '{0}/model_{1}_5para_norm.pth'.format(outfolder, i) )
        
 #  np.save('{0}/albedoError.npy'.format(outfolder), train_losses)
   np.save('{0}/imageError.npy'.format(outfolder), image_train_losses)
  # np.save('{0}/diffuseError.npy'.format(outfolder), diff_train_losses)
   np.save('{0}/roughnessError.npy'.format(outfolder), alpha_train_losses)
   epochs = np.arange(1,  iteration_count+ 1)
   plt.plot(epochs,train_losses, color='blue', label="Total Loss")
   plt.plot(epochs,image_train_losses, color='red', label="Image Train Loss")
 #  plt.plot(epochs,diff_train_losses, color='green', label="Diff Train Loss")
   plt.plot(epochs,alpha_train_losses, color='purple', label="Alpha Train Loss")
   plt.plot(epochs,test_losses, color='blue', label="Total Test Loss")
   plt.plot(epochs,image_test_losses, color='red', label="Image Test Loss")
 #  plt.plot(epochs,diff_train_losses, color='green', label="Diff Train Loss")
   plt.plot(epochs,alpha_test_losses, color='purple', label="Alpha Test Loss")
   plt.title("Training Losses Over Epochs") 
   plt.xlabel("Epochs")
   plt.ylabel("Loss")
   plt.show()
   plt.legend()
   output_path = os.path.join(outfolder, 'training_losses.png')
   plt.savefig(output_path)
   model.train(mode=False)
   torch.save(model.state_dict(), '{0}/model.pth'.format(outfolder))
   print("Training Done!  Saved PyTorch Model State to model.pth")
   
   
   del albedo
   del alpha

   del image
   '''  
   print("Tesing the model") 
   loss_accum = 0
   image_loss = 0

   alpha_loss = 0

   model.load_state_dict(torch.load('{0}/model.pth'.format(outfolder)))
   model.to(device)
   for j, (image,labels) in enumerate(test_data_loader):
       optimizer.zero_grad()
       image=image.permute(0,3,1,2)
       image=image.to(device)
       albedo,alpha= model(image)
       albedo=albedo_scale * albedo + albedo_bias # [bias, scale + bias]
           
             #pr=pr.clamp(0,1)
    
       labels=labels.to(device)

       loss_alpha=loss_fn(torch.log(alpha+1), torch.log(labels+1))
   
       loss2=[]
       for ii in range(batchsize):
           mask =(image[ii] != 0).any(dim=0).unsqueeze(0).expand_as(image[ii])
           tmp=render_texture(albedo[ii,:],alpha[ii,:], spp=spp, seed=ii*len(train_data_loader)+j)
           tmp=tmp.permute(2,0,1)
           log_tmp=torch.where(mask, torch.log(tmp + 1), 0)
           log_img=torch.where(mask, torch.log(image[ii] + 1), 0)
           loss2.append(loss_fn(log_tmp, log_img))
       loss3=torch.mean(torch.tensor(loss2,dtype=torch.float32)) 
       loss=loss_alpha+loss3  
       loss_accum += loss.item()
       image_loss += torch.mean(torch.tensor(loss2,dtype=torch.float32)).item()
      # diff_loss += loss_diff.item()*100
       alpha_loss += loss_alpha.item()
       print(f'Images {j+1}/{len(test_data_loader)} ,Total Loss: {loss.item()}')
     #  print(f'Images {j+1}/{len(test_data_loader)} ,Diffuse Loss: {loss_diff.item()*100}')
       print(f'Images {j+1}/{len(test_data_loader)} ,Alpha Loss: {loss_alpha.item()}')
       print(f'Images {j+1}/{len(test_data_loader)} ,Image Loss: {torch.mean(torch.tensor(loss2,dtype=torch.float32)).item()}', end='\r')
    ''' 
   np.save('{0}/TestError.npy'.format(outfolder), loss_accum/len(test_data_loader))
   np.save('{0}/imageTestError.npy'.format(outfolder), image_loss/len(test_data_loader))
#   np.save('{0}/diffuseTestError.npy'.format(outfolder), diff_loss/len(test_data_loader))
   np.save('{0}/alphaTestError.npy'.format(outfolder), alpha_loss/len(test_data_loader))    
   print("Test Done!")

   del albedo
   del alpha
  
   del image
   del model
   torch.cuda.empty_cache()

   print("Script finished and resources cleaned up.")

if __name__ == '__main__':
    main()   