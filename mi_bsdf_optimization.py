from gpu_plugin import *
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_wen import *
from matplotlib import pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_data import *
from torchvision import transforms
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HomoNetModel(nn.Module):
    def __init__(self):
        super(HomoNetModel, self).__init__()
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
            nn.BatchNorm2d(256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        
            
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
batch_size = 2
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

def main():
   eps = 1e-6
   train_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light/train_pointlight_mi'
   test_folder=r'/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light/test_pointlight_mi'
   outfolder = r'/home/wen/wen/spectral_BRDF/spectral_brdf_images/deeplearng/net_results/HomoNet_pytorch_poitnlight_mi'
   if(os.path.exists(outfolder) == False):
        os.makedirs(outfolder)  
   image_height = 256
   image_width = 256
   color_channels = 3
   aCnt = 19
   
   rCnt = 40
   aCnt_t = 5
   
   rCnt_t = 5     
   new_emitter = load_emitter()
   batchsize = 1
    
    
   @dr.wrap_ad(source='torch', target='drjit')
   def render_texture(brdf_pr, spp=2, seed=1):
      brdf_pr=dr.ravel(brdf_pr)

      diffuse_pr = brdf_pr[0]
      specular_pr = 1-diffuse_pr
      #roughness_pr = torch.exp(1e-2*brdf_pr[1])
      roughness_pr = brdf_pr[1]
      scene = load_scene(new_emitter,load_sensor(spp=spp),diffuseReflectance=diffuse_pr, specularReflectance=(1-diffuse_pr), al=roughness_pr)
      image= mi.render(scene, spp=spp, seed=seed)
      return image
   imgListFile = 'train_full.txt'
   traindata=CustomImageDataset(rootPath=train_folder, imgListFile=imgListFile, aCnt=aCnt, sCnt=aCnt, rCnt=rCnt)
   data,label=traindata.__getitem__(3)
   imgListFile2 = 'test_full.txt'
   testdata = CustomImageDataset(rootPath=test_folder, imgListFile=imgListFile2, aCnt=aCnt_t, sCnt=aCnt_t, rCnt=rCnt_t)
   train_data_loader = torch.utils.data.DataLoader(traindata, batch_size=batchsize, shuffle=True)
   test_data_loader = torch.utils.data.DataLoader(testdata, batch_size=batchsize, shuffle=False)
   model = HomoNetModel()
   print(model)
 
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   for param in model.parameters():
    if torch.isnan(param).any():
        print("模型参数包含NaNs")
    if torch.isinf(param).any():
        print("模型参数包含Infs")

   loss_fn = nn.L1Loss()

# Optimization hyper-parameters
   iteration_count = 1500
   spp = 2
   
   scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
   resize_transform = transforms.Resize((image_height, image_width)) 
   model.train(mode=True).to(device)
   #pr=torch.tensor([0.5,0.5])
   #rendered_img = render_texture(pr.numpy(), spp=spp, seed=1)  
   train_losses = []
   image_train_losses = []
   diff_train_losses = []
   rough_train_losses = []
   for i in range(iteration_count):
         print(f"Epoch {i+1}\n-------------------------------")
         loss_accum = 0
         image_loss = 0
         diff_loss = 0  
         rough_loss = 0
         
         
         for j, (image,labels) in enumerate(train_data_loader):
             optimizer.zero_grad()
             image=image.permute(0,3,1,2)
             image=image.to(device)
             pr= model(image)
             pr=pr.to(device)
             #pr=pr.clamp(0,1)
             specPred=pr[:,0].clone().to(device)
             roughPred=pr[:,1].clone().to(device)
             specGT=labels[:,0].clone().to(device)
             roughGT=labels[:,1].clone().to(device)
             
             loss_diff=loss_fn(specPred, specGT)
             if loss_diff.isnan():loss_diff=eps
             loss_rough=loss_fn(torch.abs(torch.log(roughPred+eps)), torch.abs(torch.log(roughGT+eps)))
             if loss_rough.isnan():loss_rough=eps
             loss1=loss_diff+loss_rough
             image = resize_transform(image).to(device)
            
             loss2=[]
             for ii in range(batchsize):
                  tmp=render_texture(pr[ii,:], spp=spp, seed=i*len(train_data_loader)+j)
                 # tmp=torch.stack(tmp)
                  tmp=tmp.permute(2,0,1)
                  mask =(image[ii] != 0).any(dim=0).unsqueeze(0).expand_as(image[ii])
                #  tmp=torch.where(mask, tmp, torch.zeros_like(tmp))
                #  loss2.append(loss_fn(tmp*mask, image[ii]*mask))
                 # loss2.append(loss_fn(torch.abs((((((tmp)** (1.0 / 2.2))*255).to(torch.uint8)).to(torch.float32) / 255).to(device)), torch.abs( (((((image[ii])** (1.0 / 2.2))*255).to(torch.uint8)).to(torch.float32) / 255).to(device))))
                  loss2.append(loss_fn(torch.abs(torch.where(mask,torch.log(tmp+eps),torch.zeros_like(tmp))), torch.abs(torch.where(mask,torch.log(image[ii]+eps),torch.zeros_like(image[ii])))))
             
             image.requires_grad = True
             
            # loss2=loss_fn(torch.abs(torch.log(rendered_img.pow(1.0/2.2)+eps)), torch.abs(torch.log(image.pow(1.0/2.2)+eps)))
           #  print(f'{loss2}',end='\r')
             loss=loss1*100+torch.mean(torch.tensor(loss2,dtype=torch.float32))*10
             
             loss.backward() 
             loss_accum += loss.item()
             image_loss += torch.mean(torch.tensor(loss2,dtype=torch.float32)).item()*10
             diff_loss += loss_diff.item()*100
             rough_loss += loss_rough.item()*100
          #   print(f'image{j}',end='\r')
             optimizer.step()
         scheduler.step()
         train_losses.append(loss_accum/len(train_data_loader))
         image_train_losses.append(image_loss/len(train_data_loader))
         diff_train_losses.append(diff_loss/len(train_data_loader))
         rough_train_losses.append(rough_loss/len(train_data_loader))
         print(f'Iteration {i+1}/{iteration_count}, Total Loss: {train_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Image Loss: {image_train_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Diffuse Loss: {diff_train_losses[-1]}')
         print(f'Iteration {i+1}/{iteration_count}, Roughness Loss: {rough_train_losses[-1]}', end='\r')
#         test_loop(test_data_loader, model, loss_fn)
   print("Done!")     
   model.train(mode=False)


   torch.save(model.state_dict(), "model_1500_log.pth")
   print("Saved PyTorch Model State to model.pth")
   del image
   del tmp
   del specPred
   del roughPred
   del specGT
   del roughGT
   del model
#   torch.cuda.empty_cache()

print("Script finished and resources cleaned up.")

if __name__ == '__main__':
    main()   