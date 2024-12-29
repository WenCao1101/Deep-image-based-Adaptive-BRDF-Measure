import sys,os
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
#from model import *
import torch
import merlFunctions as merl
from os.path import basename
mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
#dr.set_flag(dr.JitFlag.LoopRecord, False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360


MeasuredBRDF_global = None
def rotate_vector_tensor(vector, axis, angle):
    out = vector * torch.cos(angle).unsqueeze(1)
    
    #temp = vector * axis
    temp = torch.sum(vector * axis, dim=1, keepdim=True)
    
    out += axis * temp * (1 - torch.cos(angle).unsqueeze(1))
    
    axis = axis.unsqueeze(0)

    cross = torch.cross(axis,vector, dim=1)

    out += cross * torch.sin(angle).unsqueeze(1)
    
    return out
def xyztothetaphi(xyz):
    """Converts a unit vector to spherical coordinates (theta, phi)."""
    
    mask_pos = xyz[..., 2] > 0.99999
    mask_neg = xyz[..., 2] < -0.99999
    
    theta = torch.arccos(xyz[..., 2])
    phi= torch.arctan2(xyz[..., 1], xyz[..., 0])
  
    theta[mask_pos] = 0
    phi[mask_pos] = 0
    theta[mask_neg] = np.pi
    phi[mask_neg] = 0  
   
    return theta, phi
def rotate_vector_tensor2(x, axis, angle):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    out = cos_angle.unsqueeze(1) * x
  
    tmp1 = torch.sum(x * axis, dim=1, keepdim=True)
    tmp2 = tmp1 * (1.0 - cos_angle.unsqueeze(1))
    out += axis * tmp2
    axis = axis.unsqueeze(0)
    out += sin_angle.unsqueeze(1) * torch.cross(axis, x, dim=1)
    
    return out

def std_to_half_diff(wi, wo):
    half =0.5*(wi + wo)
    half=dr.normalize(half)
    half=half.torch().to(device)
    wi=wi.torch().to(device)
   # theta_h = torch.arccos(half[..., 2])
    #phi_h = torch.arctan2(half[..., 1], half[..., 0])
    theta_h, phi_h = xyztothetaphi(half)
    bi_normal = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)

    tmp = rotate_vector_tensor2(wi, normal, -phi_h)
    tmp = tmp / torch.norm(tmp, dim=1, keepdim=True)
    diff = rotate_vector_tensor2(tmp, bi_normal, -theta_h)
    diff = diff / torch.norm(diff, dim=1, keepdim=True)
    
    
  #  theta_d = torch.arccos(diff[..., 2])
  #  phi_d = torch.arctan2(diff[..., 1], diff[..., 0])
    theta_d, phi_d = xyztothetaphi(diff)
    return theta_h, theta_d, phi_d


class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.path=props["path"]
        self.filename = basename(self.path)[:-len('.binary')]
        
        global MeasuredBRDF_global
        
      
        MeasuredBRDF_global = merl.readMERLBRDF(self.path) # (phi_d, theta_h, theta_d, ch)
        self.merl_alpha,self.m_ax,self.m_ay,self.m_rho = merl.fit_ggx_parameters_t(MeasuredBRDF_global)
        self.m_n=mi.Vector3f(0,0,1)
        self.m_sqrt_one_minus_rho_sqr= dr.safe_sqrt(1.0-self.m_rho**2)
        self.m_tx_n=0
        self.m_ty_n=0


        if self.merl_alpha<0.0 or self.merl_alpha>1.0:
            raise "out of range"
        MeasuredBRDF_global = torch.from_numpy(MeasuredBRDF_global).float().to(device)
        #    merl_alpha = torch.tensor(merl_alpha).to(device)
     #   self.m_ggx=mi.MicrofacetDistribution(mi.MicrofacetType.GGX, self.merl_alpha,True)
     
       # self.alpha = merl_alpha
     #   m_alpha = torch.from_numpy(m_alpha).float().to(device)
        # Set the BSDF flags
        reflection_flags = (
            mi.BSDFFlags.GlossyReflection
            | mi.BSDFFlags.FrontSide
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags
        
    def half_diff_look_up_brdf(self, theta_h, theta_d, phi_d):
        theta_jalf_mask=theta_h<=0.0
        theta_h[theta_jalf_mask]=0.0
        theta_half_deg = theta_h / (torch.pi * 0.5) * BRDF_SAMPLING_RES_THETA_H
        
        id_theta_h = torch.clip(
            (torch.sqrt(theta_half_deg * BRDF_SAMPLING_RES_THETA_H)).int(),
            0,
            BRDF_SAMPLING_RES_THETA_H - 1,
        )
        id_theta_d = torch.clip(
            (theta_d / (torch.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D).int(),
            0,
            BRDF_SAMPLING_RES_THETA_D - 1,
        )
        phi_mask=phi_d<0
        phi_d[phi_mask]=phi_d[phi_mask]+torch.pi
        
        id_phi_d = torch.clip(
            (phi_d / torch.pi * BRDF_SAMPLING_RES_PHI_D / 2).int(),
            0,
            BRDF_SAMPLING_RES_PHI_D / 2-1,
        )
        
        # return the value with nearest index value, officially used in the Merl BRDF
        id_theta_h = (id_theta_h).int()
        id_theta_d = (id_theta_d).int()
        id_phi_d = (id_phi_d).int()
        # print(id_theta_d, id_phi_d, id_theta_h)
        return torch.clip(MeasuredBRDF_global[id_phi_d,id_theta_h,id_theta_d,:], 0, None)
    
    def sample(self, ctx, si, sample1, sample2, active=True):
        # Compute Fresnel terms
        bs = mi.BSDFSample3f()
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

        active &= cos_theta_i > 0
      #  bs.wo=mi.warp.square_to_cosine_hemisphere(sample2)
       
     
      #  result_ggx=self.m_ggx.sample(si.wi, sample2)
      #  bs.wo=mi.reflect(si.wi, result_ggx[0])
        #########sample #########
        o=mi.Vector3f(si.wi.x,si.wi.y,si.wi.z)
        u1=sample2.x*0.99998+0.00001
        u2=sample2.y*0.99998+0.00001
        a=o.x*self.m_ax+o.y*self.m_ay*self.m_rho
        b=o.y*self.m_ay*self.m_sqrt_one_minus_rho_sqr
        c=o.z-o.x*self.m_tx_n-o.y*self.m_ty_n
        o_std=mi.Vector3f(a,b,c)
        o_std=dr.normalize(o_std)
    
        tx_m,ty_m=self.sample_vp22_std(u1,u2,o_std)
        tx_h=self.m_ax*tx_m+self.m_tx_n
        choleski=self.m_rho*tx_m+self.m_sqrt_one_minus_rho_sqr*ty_m
        ty_h=self.m_ay*choleski+self.m_ty_n
        h=mi.Vector3f(-tx_h,-ty_h,1.0)
        h=dr.normalize(h)
        bs.wo=dr.select(o_std.z>0.0,mi.reflect(o,h),mi.Vector3f(0.0,0.0,1.0))
         #########sample #########
        
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)   
        
        bs.eta = 1.0

        # theta_h = torch.clip(theta_h, 0, torch.pi / 2)
        # theta_d = torch.clip(theta_d, 0, torch.pi / 2)
        # phi_d = torch.clip(phi_d, 0, torch.pi)

        cos_theta_wo = mi.Frame3f.cos_theta(bs.wo)
        active &= cos_theta_wo > 0.0
        
        bs.pdf = self.pdf(ctx, si, bs.wo, active)

        active1=mi.Mask=True
        value = self.eval(ctx, si, bs.wo, active1)
        return (bs, dr.select(active & (bs.pdf > 0.0), value/bs.pdf, mi.Vector3f(0.0)))
    def imprtance_sample(self, ctx, si, sample2,alpha):
        phi_a=0.0
        cos_phi_a=np.cos(phi_a)
        sin_phi_a=np.sin(phi_a)
        cos_phi_a_sq=2.0*cos_phi_a*cos_phi_a-1.0
        a1_sqr=alpha*alpha
        a2_sqr=alpha*alpha
        tmp1=a1_sqr+a2_sqr
        tmp2=a1_sqr-a2_sqr
        m_ax=np.sqrt(0.5*(tmp1+tmp2*cos_phi_a_sq))
        m_ay=np.sqrt(0.5*(tmp1-tmp2*cos_phi_a_sq))
        m_rho=(a2_sqr-a1_sqr)*sin_phi_a*cos_phi_a/(m_ax*m_ay)
        m_n=mi.Vector3f(0,0,1)
        m_sqrt_one_minus_rho_sqr= dr.safe_sqrt(1.0-m_rho**2)
        m_tx_n=0
        m_ty_n=0
        bs = mi.BSDFSample3f()
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)

      #  bs.wo=mi.warp.square_to_cosine_hemisphere(sample2)

      #  result_ggx=self.m_ggx.sample(si.wi, sample2)
      #  bs.wo=mi.reflect(si.wi, result_ggx[0])
        #########sample #########
        o=mi.Vector3f(si.wi.x,si.wi.y,si.wi.z)
        u1=sample2.x*0.99998+0.00001
        u2=sample2.y*0.99998+0.00001
        a=o.x*m_ax+o.y*m_ay*m_rho
        b=o.y*m_ay*m_sqrt_one_minus_rho_sqr
        c=o.z-o.x*m_tx_n-o.y*m_ty_n
        o_std=mi.Vector3f(a,b,c)
        o_std=dr.normalize(o_std)
    
        tx_m,ty_m=self.sample_vp22_std(u1,u2,o_std)
        tx_h=m_ax*tx_m+m_tx_n
        choleski=m_rho*tx_m+m_sqrt_one_minus_rho_sqr*ty_m
        ty_h=m_ay*choleski+m_ty_n
        h=mi.Vector3f(-tx_h,-ty_h,1.0)
        h=dr.normalize(h)
        bs.wo=dr.select(o_std.z>0.0,mi.reflect(o,h),mi.Vector3f(0.0,0.0,1.0))
            #########sample #########
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)   
        
        bs.eta = 1.0   
        return bs 
    def sample_vp22_std(self,u1,u2,o_std):
        cos_theta_k=o_std.z
        sin_theta_k=dr.select(cos_theta_k<1.0,dr.safe_sqrt(1.0-cos_theta_k*cos_theta_k),mi.Float(0.0))
        tx=self.qf2_radial(u1,cos_theta_k,sin_theta_k)
        ty=self.q3_radial(u2,tx)

        xslope1=tx
        yslope1=ty

        nrm=1.0/dr.safe_sqrt(o_std.x*o_std.x+o_std.y*o_std.y)
        cos_phi_k=o_std.x*nrm
        sin_phi_k=o_std.y*nrm
        xslope2=cos_phi_k*tx-sin_phi_k*ty
        yslope2=sin_phi_k*tx+cos_phi_k*ty

        xslope=dr.select(sin_theta_k==0.0,xslope1,xslope2)
        yslope=dr.select(sin_theta_k==0.0,yslope1,yslope2)
      
        return xslope,yslope
    def qf2_radial(self,u,cos_theta_k,sin_theta_k):
        sin_theta=u*(1.0+cos_theta_k)-1.0
        cos_theta=dr.safe_sqrt(1.0-sin_theta*sin_theta)
        
        tan_theta=sin_theta/cos_theta
        tan_theta_k=sin_theta_k/cos_theta_k
        res1=-(tan_theta+tan_theta_k)/(1.0-tan_theta*tan_theta_k)
        
        cot_theta_k2=cos_theta_k/sin_theta_k
        res2=(1.0+tan_theta*cot_theta_k2)/(tan_theta-cot_theta_k2)
        value1=dr.select(sin_theta_k<0.707107,res1,res2)

        cot_theta=cos_theta/sin_theta
        tan_theta_k2=sin_theta_k/cos_theta_k
        res3=(1.0+tan_theta_k2*cot_theta)/(tan_theta_k2-cot_theta)

        cot_theta_k=cos_theta_k/sin_theta_k
        res4=(cot_theta+cot_theta_k)/(1.0-cot_theta*cot_theta_k)
        value2=dr.select(sin_theta_k<0.707107,res3,res4)

        value=dr.select(cos_theta>0.707107,value1,value2)
        return value
    def q3_radial(self,u,qf2):
        alpha=dr.safe_sqrt(1.0+qf2*qf2)
        S=0
     
        u1=2.0*(0.5-u)
        S1=-1.0

        u2=2.0*(u-0.5)
        S2=1.0
        u=dr.select(u<0.5,u1,u2)
        S=dr.select(u<0.5,S1,S2)
        p=u * (u * (u * (-0.365728915865723) 
	          + 0.790235037209296) - 0.424965825137544) + 0.000152998850436920
        q=u * (u * (u * (u * 0.169507819808272 - 0.397203533833404) 
	          - 0.232500544458471) + 1) - 0.539825872510702
        value=S*alpha*(p/q)
        return value    
    def eval(self, ctx, si, wo, active=True):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        wi = si.wi

        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)

      #  theta_h = torch.clip(theta_h, 0, torch.pi / 2)
      #  theta_d = torch.clip(theta_d, 0, torch.pi / 2)
        mask_phi_d=phi_d[...]<0  
        phi_d[mask_phi_d]=phi_d[mask_phi_d]+torch.pi
      #  phi_d = torch.clip(phi_d, 0, None)
       

        value = self.half_diff_look_up_brdf(theta_h, theta_d,phi_d)
       # value = value.detach().cpu().numpy()
        value = mi.Vector3f(value[..., 0], value[..., 1], value[..., 2])
     #   value=mi.Vector3f(0.1,0.5,0.7)*dr.inv_pi
        return dr.select(
            (cos_theta_i >0.0) & (cos_theta_o >0.0), value*cos_theta_o, mi.Vector3f(0)
        )
    def eval_m(self, ctx, si, wo, active=True):
      #  cos_theta_i = mi.Frame3f.cos_theta(si.wi)
     #   cos_theta_o = mi.Frame3f.cos_theta(wo)

        wi = si.wi

        theta_h, theta_d, phi_d = std_to_half_diff(wi, wo)

      #  theta_h = torch.clip(theta_h, 0, torch.pi / 2)
      #  theta_d = torch.clip(theta_d, 0, torch.pi / 2)
#        phi_d = torch.clip(phi_d, 0, None)


        value = self.half_diff_look_up_brdf(theta_h, theta_d,phi_d)
       # value = value.detach().cpu().numpy()
        value = mi.Vector3f(value[..., 0], value[..., 1], value[..., 2])
     #   value=mi.Vector3f(0.1,0.5,0.7)*dr.inv_pi
      #  return dr.select(
      #      (cos_theta_i >0.0) & (cos_theta_o >0.0), value*cos_theta_o, mi.Vector3f(0)
      #  )
        return value
    def sigam(self,wh):
        a=wh.x*self.m_ax+wh.y*self.m_ay*self.m_rho
        b=wh.y*self.m_ay*self.m_sqrt_one_minus_rho_sqr
        c=wh.z-wh.x*self.m_tx_n-wh.y*self.m_ty_n
        nrm=dr.safe_sqrt(a*a+b*b+c*c)
        ttt=mi.Vector3f(a,b,c)/nrm

        result=nrm*((1.0+ttt.z)/2.0)
  
        return result
    def g1(self, wi, wh):
       g1_loacal=dr.dot(wh,self.m_n)
    
       ss=self.sigam(wh)
       result=wh.z/ss
       return dr.select(g1_loacal>0.0,result,mi.Float(0.0))
    def pdf(self, ctx, si, wo, active=True):
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
       
        h=dr.normalize(si.wi+wo)
        kh=dr.dot(h,wo)
    #    pdf1=self.m_ggx.pdf(si.wi,h)/ (4.0 * dr.dot(wo, h))
    #    pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo)
      ######gaf-G##########
        g1_o=self.g1(h,wo)
        
      #  result1=self.m_ggx.smith_g1(h, wo)
        g1_i=self.g1(h,si.wi)
        tmp=g1_o*g1_i
     #   dr.mask=active&tmp>0.0
        active=mi.Mask
        active=tmp>0.0
        G=tmp/(g1_i+g1_o-tmp)
        active &=G>0.0
        active &=kh>0.0
        active &=h.z>0.0

      ##########vndf##########
        cos_therta_h_sqr=h.z*h.z
        cos_theta_h_sqr_sqr=cos_therta_h_sqr*cos_therta_h_sqr
        xslope=-h.x/h.z
        yslope=-h.y/h.z
        nrm_p22=self.m_ax*self.m_ay*self.m_sqrt_one_minus_rho_sqr
        x=xslope-self.m_tx_n
        y=yslope-self.m_ty_n
        x_=x/self.m_ax
        tmp1=self.m_ax*y-self.m_rho*self.m_ay*x
        tmp2=self.m_ax*self.m_ay*self.m_sqrt_one_minus_rho_sqr
        y_=tmp1/tmp2
        r_sqr=x_*x_+y_*y_
        tmp=1.0+r_sqr
        p22_std=(1.0/(np.pi*tmp*tmp))/nrm_p22/cos_theta_h_sqr_sqr
        vndf=p22_std*kh/self.sigam(wo)
        pdf=vndf/(4.0*dr.dot(si.wi,h))
        
        return dr.select((cos_theta_i >0.0) & (cos_theta_o >0.0)&active , pdf, mi.Float(0.0))
    
    

    def eval_pdf(self, ctx, si, wo, active=True):
        f = self.eval(ctx,si,wo,active)
        pdf = self.pdf(ctx,si,wo,active)
        return f,pdf

    def to_string(self):
        return "MyBSDF[\n" "  filename = \"%s\"\n" "]" % self.filename  # noqa: E131
mi.register_bsdf("merl_measure", lambda props: MyBSDF(props))

def load_scene( mysensor,path):
    scene_dict = {
        'type': 'scene',
        'sensor': mysensor,
        "integrator": {
        "type": "direct"},
         'sphere' : {
        'type': 'sphere',
        'mybsdff':{
          "type": "merlbsdf",
          "path": path,
          }},
       
        "myemitter":{
     'type': 'point',
     'position':(0, 0, 4),
     'intensity': {
         'type': 'rgb',
         'value': 50.0,
     }},
    }
    
    return mi.load_dict(scene_dict)

def load_sensor(spp=2):
    return mi.load_dict({
               "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 100.0,
        "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0.1, 3.5],
                                                 target=[0, 0, 0],
                                                 up=[0, 0, 1]),
        "myfilm": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "box"
            },
            "width": 256,
            "height": 256,
        }, "mysampler": {
            "type": "independent",
            "sample_count": spp,
        }
    })

def load_emitter(light_int=20,rotateX=0,rotateY=0):
    emitter = {
        'type': 'point',
    #    'position': light_pos,
        'position':mi.ScalarTransform4f.rotate(angle=rotateX, axis=[1, 0, 0]).rotate(angle=rotateY, axis=[0, 1, 0])@mi.ScalarPoint3f([0,0,4]),
        'intensity': {
                'type': 'spectrum',
                'value': light_int,
            },
    }
    return mi.load_dict(emitter)
def load_emitter1(light_pos,light_int=20,rotateX=0,rotateY=0):
    emitter = {
        'type': 'point',
      #  'position': [0,0,4],
        'to_world':mi.ScalarTransform4f.translate([0,0,4]).rotate(angle=rotateX, axis=[1, 0, 0]).rotate(angle=rotateY, axis=[0, 1, 0]),
        'intensity': {
                'type': 'spectrum',
                'value': light_int,
            },
    }
    return mi.load_dict(emitter)
def load_merl_cus(path):
    return mi.load_dict({
          "type": "merlbsdf",
          "path":path,
          
})

def load_shape(shape_type="sphere", mybsdf=None):
    return mi.load_dict({
        "type": shape_type,
        "something": mybsdf,
    })



scene11 = mi.load_dict({
    "type": "scene",
    "myintegrator": {
        "type": "direct",
    },
    "mysensor": {
        "type": "perspective",
        "near_clip": 1.0,
        "far_clip": 100.0,
        "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0.1, 3.5],
                                                 target=[0, 0, 0],
                                                 up=[0, 0, 1]),
        "myfilm": {
            "type": "hdrfilm",
            "rfilter": {
                "type": "box"
            },
            "width": 512,
            "height": 512,
        }, "mysampler": {
            "type": "independent",
            "sample_count":1024,
        },
    },

      "myemitter":{
      'type': 'point',
      'position':(0, 0, 4),
      'intensity': {
          'type': 'rgb',
          'value': 50.0,
      }},

  #  "myemitter":{
  #   'type': 'envmap',
  #   'filename': '/home/wen/wen/spectral_BRDF/mitsuba/scene/sssdragon/envmap.exr',
  #   },

    "myshape": {
        "type": "sphere",
      #  "radius": 0.5,

        "mybsdf": {
            "type": "merl_measure",
            "path": "/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/green-acrylic/reconstruction_green-metallic-paint_32.binary",
            }
              }
})

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    import time

    start_time = time.time()

    seed = 0
    image = mi.render(scene11, spp=2, seed=seed).torch()
    plt.imshow(image.cpu().pow(1/2.2).clamp(0,1))
    plt.show()
    

    mi.util.write_bitmap("merl_measured2.png", image)
    end_time = time.time()

    print("Render time: " + str(end_time - start_time) + " seconds")