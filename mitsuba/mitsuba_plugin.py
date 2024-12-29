import sys
mitsuba3_p = '/home/wen/wen/spectral_BRDF/mitsuba3/build/python'  # "mitsuba3" 目录的路径
sys.path.append(mitsuba3_p)

import torch
import mitsuba 

import drjit as dr
import numpy as np
dr.set_flag(dr.JitFlag.VCallRecord, False)
import mitsuba as mi
import matplotlib.pyplot as plt
mi.set_variant('cuda_ad_rgb')
#dr.set_flag(dr.JitFlag.LoopRecord, False)
class MyBSDF(mitsuba.BSDF):
    def __init__(self, props):
        mitsuba.BSDF.__init__(self, props)
        self.m_specularReflectance =dr.mean( props['specularReflectance'])
        self.m_diffuseReflectance =dr.mean( props['diffuseReflectance'])
        self.alpha =dr.mean( props['alpha'])
        reflection_flags   = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.Glossy
        self.m_specularSamplingWeight = dr.mean(dr.abs(self.m_specularReflectance)) /dr.mean(dr.abs(self.m_specularReflectance) + dr.abs(self.m_diffuseReflectance))
       # reflection_flags   = mitsuba.BSDFFlags.SpatiallyVarying|mitsuba.BSDFFlags.DiffuseReflection|mitsuba.BSDFFlags.FrontSide | mitsuba.BSDFFlags.BackSide
        self.m_components  = [reflection_flags]
        self.m_flags = reflection_flags

    def sample(self, ctx, si, sample1, sample2, active=True):
        chooseSpecular = True
        
        bs = mitsuba.BSDFSample3f()
        wo=-si.wi
   
     
        alpha=dr.mean(self.alpha)
        
        for ii in range(dr.shape(sample2)[1]):
            if(sample2[0,ii]<self.m_specularSamplingWeight):
            #    sample2[0,ii] = sample2[0,ii]/self.m_specularSamplingWeight
                 chooseSpecular = True
            else:
            #    sample2[0,ii] = (sample2[0,ii]-self.m_specularSamplingWeight)/(1-self.m_specularSamplingWeight)
                chooseSpecular = False

            if(chooseSpecular):
           # specular sampling
               phiH = 2.0*dr.pi*sample2[1,ii]
              # if(sample2[1,ii]>0.5):
              #     phiH = phiH+dr.pi 
               cosPhiH=dr.cos(phiH)
               sinPhiH=dr.sin(phiH)
               thetaH = dr.atan(alpha*dr.safe_sqrt(-dr.log(dr.maximum(sample2[0,ii],1e-8))))   
               
               H=np.array([dr.sin(thetaH)*dr.cos(phiH),dr.sin(thetaH)*dr.sin(phiH),dr.cos(thetaH)]) 
               #if(dr.shape(H)[0]==1):
               #H=np.array([H[0,0],H[1,0],H[2,0]])
               wii=np.array([si.wi[0,ii],si.wi[1,ii],si.wi[2,ii]]) 
               woo = 2.0*np.dot(wii,H)*H-wii
               wo[0,ii]=woo[0]
               wo[1,ii]=woo[1]
               wo[2,ii]=woo[2]

              # bs.pdf[ii]=self.pdf_s(wii,woo)
               
          #  else:
           # diffuse sampling   
          #    r1=2.0*sample2[0,ii]-1.0
          #    r2=2.0*sample2[1,ii]-1.0
          #    if(r1==0.0 and r2==0.0):
          #        r=phi=0.0
          #    elif(r1*r1>r2*r2):
          #       r=r1
          #       phi=(dr.pi/4.0)*(r2/r1) 
          #    else:
          #       r=r2
          #       phi=(dr.pi/2.0)-(dr.pi/4.0)*(r1/r2)
          #    sinphi,cosphi = dr.sincos(phi)
          #    wo[0,ii]=r*cosphi
          #    wo[1,ii]=r*sinphi
          #    wo[2,ii]=dr.safe_sqrt(1.0-r*cosphi*r*cosphi-r*sinphi*r*sinphi)
                 
             # bs.pdf[ii]=mi.warp.square_to_cosine_hemisphere_pdf([bs.wo[0,ii],bs.wo[1,ii],bs.wo[2,ii]])[0]
              
        bs.sampled_component = mitsuba.UInt32(0)
     #   ctx.type_mask.set(bs.lobes, +mitsuba.BSDFFlags.DiffuseReflection)
        bs.sampled_type = mitsuba.UInt32(+mitsuba.BSDFFlags.GlossyReflection)   
     #   bs.sampled_type = mitsuba.UInt32(+mitsuba.BSDFFlags.DiffuseReflection)
        pdf=self.pdf(ctx,si,wo)
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = 1.0
       
        value = self.eval(ctx,si,bs.wo)/bs.pdf
        return (bs,value)
    def importance_sample(self, wi, sample2):
   
        alpha=dr.mean(self.alpha)
        
      #  for ii in range(dr.shape(sample2)[1]):

         #   if(chooseSpecular):
           # specular sampling
        phiH = 2.0*dr.pi*sample2[1]
              # if(sample2[1,ii]>0.5):
              #     phiH = phiH+dr.pi 
        cosPhiH=dr.cos(phiH)
        sinPhiH=dr.sin(phiH)
        thetaH = dr.atan(alpha*dr.safe_sqrt(-dr.log(dr.maximum(sample2[0],1e-8))))   
               
        H=mi.Vector3f(dr.sin(thetaH)*dr.cos(phiH),dr.sin(thetaH)*dr.sin(phiH),dr.cos(thetaH)) 
        wii=mi.Vector3f(wi.x,wi.y,wi.z) 
        
        woo=mi.reflect(wii,H)    
       
       
     
        return (woo)
    def eval(self, ctx, si, wo, active=True):
        result = mitsuba.Vector3f(0.0,0.0,0.0)
        h=0.5*(si.wi+wo)
        h=dr.normalize(h)
       # thetaH = dr.acos(h[2])
      #  phiH = dr.atan2(h[1],h[0])
        
        alpha=dr.mean(self.alpha)
        alpha2 = alpha*alpha
        factor1 = mitsuba.Float32(0.0)
        factor1 = 1.0/(4.0*dr.pi *alpha2* dr.sqrt(dr.clamp(mitsuba.Frame3f.cos_theta(wo),1e-5,1)*dr.clamp(mitsuba.Frame3f.cos_theta(si.wi),1e-5,1)))
        factor2=h[0]/alpha
        #exponent=-(2.0*factor2*factor2)/(h[2]*h[2])
        tan_thetaH=mitsuba.Frame3f.tan_theta(h)
        exponent=-(tan_thetaH*tan_thetaH)/(alpha*alpha)
        spectRef=factor1*dr.exp(exponent)
        result=self.m_specularReflectance*spectRef
      #  result=dr.maximum(result+spectRef,1e-16)
     #   result=dr.maximum(result+self.m_specularReflectance*spectRef,1e-16)
     #   result=result+self.m_diffuseReflectance*1.0/dr.pi       
        return result
     #   return result*dr.clamp(mitsuba.Frame3f.cos_theta(wo),1e-5,1)
    
    def pdf(self, ctx, si, wo, active=True):
        pdf = mitsuba.Float32(0.0)
        diffusePdf = mitsuba.Float32(0.0)
        specPdf = mitsuba.Float32(0.0)
        h=0.5*(si.wi+wo)
        h=dr.normalize(h)
        tan_thetaH=mitsuba.Frame3f.tan_theta(h)
        alpha=dr.mean(self.alpha)
        alpha2 = alpha*alpha
        factor1 = mitsuba.Float32(0,0)
        factor1 = 1.0/(4.0*dr.pi *alpha2* dr.dot(h,si.wi)*dr.clamp(dr.power(mitsuba.Frame3f.cos_theta(h),3),1e-5,1))
       # factor2=h[0]/self.alpha
       # exponent=-(2.0*factor2*factor2)/(h[2]*h[2])
       # tan_thetaH=mitsuba.Frame3f.tan_theta(h)
        exponent=-(tan_thetaH*tan_thetaH)/(alpha*alpha)
        specPdf=factor1*dr.exp(exponent)
        diffusePdf = dr.clamp(mitsuba.Frame3f.cos_theta(wo),1e-5,1)*1/dr.pi
        pdf = self.m_specularSamplingWeight*specPdf + (1-self.m_specularSamplingWeight)*diffusePdf
        return pdf
    def pdf_s(self, wii, wo):
        pdf = mitsuba.Float32(0.0)
        diffusePdf = mitsuba.Float32(0.0)
        specPdf = mitsuba.Float32(0.0)
        h=0.5*(wii+wo)
        h=h/np.linalg.norm(h)
        #h=dr.normalize(h)
        tan_thetaH=mitsuba.Frame3f.tan_theta(h)
        alpha=self.alpha
        #alpha=dr.mean(self.alpha)
        alpha2 = alpha*alpha
        factor1 = mitsuba.Float32(0,0)
        factor1 = 1.0/(4.0*dr.pi *alpha2* np.dot(h,wii)*dr.clamp(dr.power(mitsuba.Frame3f.cos_theta(h),3),1e-5,1))
       # factor2=h[0]/self.alpha
       # exponent=-(2.0*factor2*factor2)/(h[2]*h[2])
       # tan_thetaH=mitsuba.Frame3f.tan_theta(h)
        exponent=-(tan_thetaH*tan_thetaH)/(alpha*alpha)
        specPdf=factor1*dr.exp(exponent)
        return specPdf
    def sample_reverse(self,wi,wo):
    #importance sampling with ward brdf
     wi=dr.normalize(wi) 
     alpha=self.alpha
   # wo=wo/np.linalg.norm(wo)
     h=dr.normalize(0.5*(wo+wi))
     thetaH=dr.acos(h[2])
     
     phiH=dr.atan2(h[1],h[0])
     if(phiH[0]<0):
        phiH=phiH+2*dr.pi 
  
     sample2=np.zeros(2,dtype=np.float32)
 
    
           # specular sampling
     sample2[1]=phiH[0]/(2.0*dr.pi)    

     sample2[0]=dr.maximum(dr.exp(-dr.power((dr.tan(thetaH[0]))/alpha,2)),1e-8)
          

     return sample2
    def eval_pdf(self, ctx, si, wo, active=True):
        f = self.eval(ctx,si,wo,active)
        pdf = self.pdf(ctx,si,wo,active)
        return f,pdf

    def to_string(self,):
        return 'Ward_py'

mitsuba.register_bsdf("Ward_py", lambda props: MyBSDF(props))

def load_scene( mylight, mysensor,diffuseReflectance=0.2, specularReflectance=0.8, al=0.2):
    scene_dict = {
        'type': 'scene',
        'sensor': mysensor,
        "integrator": {
        "type": "direct"},
         'sphere' : {
        'type': 'sphere',
        'mybsdff':{
          "type": "ward",
          "diffuse_reflectance":diffuseReflectance ,
          "specular_reflectance":specularReflectance,
          "alpha":al,}},
        'light': mylight,
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
def load_bsdf_cus(diffuseReflectance=0.2, specularReflectance=0.8, al=0.2):
    return mi.load_dict({
          "type": "Ward_py",
          "diffuseReflectance":diffuseReflectance ,
          "specularReflectance":specularReflectance,
          "alpha":al,
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
            "width": 256,
            "height": 256,
        }, "mysampler": {
            "type": "independent",
            "sample_count":2,
        },
    },
    "myemitter":{
    'type': 'point',
    'position':[0, 0, 4],
    'intensity': {
        'type': 'spectrum',
        'value': 20.0,
    }},
    
    "myshape": {
        "type": "sphere",
        "mybsdf": {
            "type": "Ward_py",
            "diffuseReflectance":np.array(0.8) ,
            "specularReflectance":np.array(0.2),
            "alpha":np.array(0.1),
            } }
})


#img = mitsuba.render(scene11).torch()
#1
#plt.imshow(img.cpu().pow(1/2.2).clamp(0,1))
#1
#mi.util.write_bitmap('test_groudtruth.png', img)