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
from utils_wen import *
from os.path import basename
mi.set_variant("cuda_ad_rgb")
dr.set_flag(dr.JitFlag.VCallRecord, False)
#dr.set_flag(dr.JitFlag.LoopRecord, False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def p_func(u):
    return -0.365728915865723 * u**3 + 0.790235037209296 * u**2 - 0.424965825137544 * u + 0.000152998850436920

def q_func(u):
    return 0.169507819808272 * u**4 - 0.397203533833404 * u**3 - 0.232500544458471 * u**2 + u - 0.539825872510702

def derive_f_func(u, pq):
    return (-1.097186747597169 * u**2 + 1.580470074418592 * u - 0.424965825137544 
            - 0.678031279233088 * pq * u**3 + 1.191610601500212 * pq * u**2 
            + 0.465001088916942 * pq * u - pq)

def newton_rhapson(pq, u):
    eps = 1e-6
    h = p_func(u) - pq * q_func(u) / derive_f_func(u, pq)

    for i in range(dr.shape(h)[0]):
        while np.abs(h[i]) >= eps:
           h[i] = (p_func(u) - pq[i] * q_func(u)) / derive_f_func(u, pq[i])
           u = u - h[i]
   
    return u


class MyBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.path=props["path"]
        
  ##########################################################
        self.filename = basename(self.path)[:-len('.npz')]
        MeasuredBRDF =readMeasure(self.path) # (phi_d, theta_h, theta_d, ch)
        self.m_diffuseReflectance =MeasuredBRDF['diffuse']
        self.m_specularReflectance = 1.0-self.m_diffuseReflectance
        self.alpha = MeasuredBRDF['roughness']
        self.m_n=mi.Vector3f(0,0,1)
        
        self.m_tx_n=0
        self.m_ty_n=0
        self.m_ax,self.m_ay,self.m_rho =self.parameters_changed()
        self.m_sqrt_one_minus_rho_sqr= dr.safe_sqrt(1.0-self.m_rho**2)
        self.Measuredbrdf =MeasuredBRDF['measure_brdf']
        self.theta_in=MeasuredBRDF['theta_in']
        self.phi_in=MeasuredBRDF['phi_in']
        ##################################################################3
        # Set the BSDF flags
        reflection_flags = (
            mi.BSDFFlags.GlossyReflection
            | mi.BSDFFlags.FrontSide
        )
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags
        
    
    def parameters_changed(self):
        phi_a=0.0
        cos_phi_a=np.cos(phi_a)
        sin_phi_a=np.sin(phi_a)
        cos_phi_a_sq=2.0*cos_phi_a*cos_phi_a-1.0
        a1_sqr=self.alpha*self.alpha
        a2_sqr=self.alpha*self.alpha
        tmp1=a1_sqr+a2_sqr
        tmp2=a1_sqr-a2_sqr
        ax=np.sqrt(0.5*(tmp1+tmp2*cos_phi_a_sq))
        ay=np.sqrt(0.5*(tmp1-tmp2*cos_phi_a_sq))
        rho=(a2_sqr-a1_sqr)*sin_phi_a*cos_phi_a/(ax*ay)
        return ax,ay,rho   
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
    def sample_reverse(self, wi, wo):
        # Compute Fresnel terms
        h=dr.normalize(wi+wo)
        tx_h = -h.x / h.z
        ty_h = -h.y / h.z
        tx_m = (tx_h - self.m_tx_n) / self.m_ax
        choleski = (ty_h - self.m_ty_n) / self.m_ay
        ty_m = (choleski - self.m_rho * tx_m) / self.m_sqrt_one_minus_rho_sqr 
        o=mi.Vector3f(wi.x,wi.y,wi.z)
        a=o.x*self.m_ax+o.y*self.m_ay*self.m_rho
        b=o.y*self.m_ay*self.m_sqrt_one_minus_rho_sqr
        c=o.z-o.x*self.m_tx_n-o.y*self.m_ty_n
        o_std=mi.Vector3f(a,b,c)
        o_std=dr.normalize(o_std)
        active=mi.Mask=True
        active &= o_std.z > 0.0
        #if o_std.z <= 0.0:
        #    return mi.Vector3f(0.0,0.0)
        cos_theta_k=o_std.z
        sin_theta_k=dr.select(cos_theta_k<1.0,dr.safe_sqrt(1.0-cos_theta_k*cos_theta_k),mi.Float(0.000001))
        tttm=dr.safe_sqrt(o_std.x*o_std.x+o_std.y*o_std.y)
        ttm=dr.select(tttm>0.0,tttm,mi.Float(0.000001))
        nrm=1.0/ttm
        cos_phi_k=o_std.x*nrm
        sin_phi_k=o_std.y*nrm
        tx1=tx_m
        ty1=ty_m
        tx2=cos_phi_k*tx_m+sin_phi_k*ty_m
        ty2=cos_phi_k*ty_m-sin_phi_k*tx_m
        tx=dr.select(sin_theta_k==0.0,tx1,tx2)
        ty=dr.select(sin_theta_k==0.0,ty1,ty2)
         
        tan_theta_k=sin_theta_k/cos_theta_k
        tan_theta1=(tan_theta_k+tx)/(tan_theta_k*tx-1.0) 
        cot_theta_k=cos_theta_k/sin_theta_k
        tan_theta2=(1.0+tx*cot_theta_k)/(tx-cot_theta_k)
        tan_theta=dr.select(sin_theta_k<0.707107,tan_theta1,tan_theta2)

        cos_theta1=1.0/dr.safe_sqrt(1.0+tan_theta*tan_theta)
        tan_theta_k2=sin_theta_k/cos_theta_k
        cot_theta1=(-1.0+tx*tan_theta_k2)/(tx+tan_theta_k2)
        cot_theta_k2=cos_theta_k/sin_theta_k
        cot_theta2=(tx-cot_theta_k2)/(1.0+tx*cot_theta_k2)
        cot_theta=dr.select(sin_theta_k<0.707107,cot_theta1,cot_theta2)
        
        cos_theta2=cot_theta/dr.safe_sqrt(1.0+cot_theta*cot_theta)
        cos_theta2=dr.abs(cos_theta2)
        cos_theta=dr.select(cos_theta1>0.707107,cos_theta1,cos_theta2)
        sin_theta=cos_theta/cot_theta
        u1=(1.0+sin_theta)/(1.0+cos_theta_k)
      #  u12=(1.0-sin_theta)/(1.0-cos_theta_k)
       # if (u11>=0.0)&(u11<=1.0):
       #     u1=u11
       # else:
        #    u1=u12
        u1=(u1 - 0.00001) / 0.99998   
        u1=dr.select(active,u1,mi.Float(0.0))
        u1=dr.clamp(u1,0.0,1.0) 
        alpha=dr.safe_sqrt(1.0+tx*tx)
        S=-1.0
 
        target1=ty/(S*alpha)
        active1=mi.Mask=True
        active1 &= target1>=0.0
       # if (target1>=0.0):
        u21=newton_rhapson(target1,0.1)
        u21=(1.0-u21)/2.0
        u21=dr.select(u21<0.0,(1.0-newton_rhapson(target1,0.8))/2.0,u21)
        u21=dr.select(active1,u21,mi.Float(0.0))
        
        S=1.0
        target2=ty/(S*alpha)
        u22=newton_rhapson(target2,0.8) 
        u22=(1.0+u22)/2.0
        u22=dr.select(u22>1.0,(1.0+newton_rhapson(target2,0.1))/2.0,u22)
        u22=dr.select(active1,mi.Float(0.0),u22)
        u2=dr.select((u21<0.5)&(u21>0.0),u21,u22)
       
        u2=(u2 - 0.00001) / 0.99998   
        u2=dr.clamp(u2,0.0,1.0)   
        u2=dr.select(active,u2,mi.Float(0.0))          
        return u1,u2            

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
        S=dr.select(u<0.5,S1,S2)
        u=dr.select(u<0.5,u1,u2)
        
        p=u * (u * (u * (-0.365728915865723) 
	          + 0.790235037209296) - 0.424965825137544) + 0.000152998850436920
        q=u * (u * (u * (u * 0.169507819808272 - 0.397203533833404) 
	          - 0.232500544458471) + 1) - 0.539825872510702
        value=S*alpha*(p/q)
        return value    
    def eval(self, ctx, si, wo, active=True):
        
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        h=dr.normalize(si.wi+wo)
     #   phi_h=dr.atan2(h[1],h[0])
       # theta_h=dr.safe_acos(h[2])
       # dr.select(phi_h<0.0,phi_h+2.0*np.pi,phi_h)
        
      #  h=mi.Vector3f(dr.cos(phi_h)*dr.sin(theta_h),dr.sin(phi_h)*dr.sin(theta_h),dr.cos(theta_h))    
      #  h=dr.normalize(h)
        result=mi.Color3f(0.0)
      #  sample_s=mi.warp.cosine_hemisphere_to_square(wo)
        u1,u2=self.sample_reverse(si.wi,wo)
        param_values = [self.phi_in, self.theta_in,np.array([0,1,2])]
        data_m=mi.MarginalDiscrete2D3(self.Measuredbrdf,param_values,False, False)
        sample_s=mi.Vector2f(u1,u2)
        
      #  theta_i=elevation(si.wi)
        theta_i=dr.safe_acos(si.wi[2])  
        phi_i=dr.atan2(si.wi[1],si.wi[0])
        for i in range(3):
            params_rgb=[phi_i,theta_i,i]
            result[i]=data_m.eval(sample_s,params_rgb)
    
        return dr.select(
            (cos_theta_i >0.0) & (cos_theta_o >0.0), result*cos_theta_o, mi.Vector3f(0)
        )
    
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

def load_scene_measure( mysensor,path):
    scene_dict = {
        'type': 'scene',
        'sensor': mysensor,
        "integrator": {
        "type": "direct"},
         'sphere' : {
        'type': 'sphere',
        'mybsdff':{
          "type": "merl_measure",
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

def load_sensor_meas(spp=2):
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
            "width": 512,
            "height": 512,
        }, "mysampler": {
            "type": "independent",
            "sample_count": spp,
        }
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
  #    'filename': '/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/white-diffuse-ball/envmap.exr',
 #     },

    "myshape": {
        "type": "sphere",
      #  "radius": 0.5,

        "mybsdf": {
            "type": "merl_measure",
            "path": "/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/white-diffuse-ball/2_8_36_36_ada.npz",
            }
              }
})

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    import time

    start_time = time.time()

    seed = 0
    image = mi.render(scene11, spp=10, seed=seed).torch()
    plt.imshow(image.cpu().pow(1/2.2).clamp(0,1))
    plt.show()
    

    mi.util.write_bitmap("/mnt/symphony/wen/spectral_brdfs/train_data_wen_point_light_merl/resample_merl/white-diffuse-ball/2_8_36_36_ada.png", image)
    end_time = time.time()

    print("Render time: " + str(end_time - start_time) + " seconds")