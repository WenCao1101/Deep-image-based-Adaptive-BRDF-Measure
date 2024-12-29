#Read BRDF
import numpy as np
import os.path as path
from os.path import basename
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D = 360

def rotate_vector_tensor(vector, axis, angle):
    out=vector*np.cos(angle)

    temp = np.dot(vector, axis)
   
    
    out += axis * temp * (1 - np.cos(angle))
    

    cross = np.cross(axis, vector)

    out += cross * np.sin(angle)
    
    return out

def std_half_diff(wi, wo):
    half =0.5*(wi + wo)
    half=half/np.linalg.norm(half)
 #   theta_h=np.arccos(half[2])
 #   phi_h=np.arctan2(half[1], half[0])
    theta_h, phi_h = xyztothetaphi(half)
    bi_normal = np.array([0, 1, 0], dtype=np.float32)
    normal = np.array([0, 0, 1], dtype=np.float32)
    

    tmp = rotate_vector_tensor(wi, normal, -phi_h)
    tmp = tmp / np.linalg.norm(tmp)
    diff = rotate_vector_tensor(tmp, bi_normal, -theta_h)
    diff = diff / np.linalg.norm(diff)
    
 #   theta_d = np.arccos(diff[2])
 #   phi_d = np.arctan2(diff[1], diff[0])
    theta_d, phi_d = xyztothetaphi(diff)
    return theta_h, theta_d, phi_d
def phithera2vec(theta, phi):
    """Converts spherical coordinates (theta,phi) to a unit vector"""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    return np.array([sin_theta*cos_phi,sin_theta*sin_phi,cos_theta])
def xyztothetaphi(xyz):
    """Converts a unit vector to spherical coordinates (theta, phi)."""
    x, y, z = xyz
    if z > 0.99999:
        theta = phi = 0
    elif z < -0.99999:
        theta = np.pi
        phi = 0
    else:
        phi = np.arctan2(y, x)
        theta = np.arccos(z)
    return theta, phi
def half_diff_look_brdf(theta_h, theta_d, phi_d):

        theta_half_deg = theta_h / (np.pi * 0.5) * BRDF_SAMPLING_RES_THETA_H
        
        id_theta_h = np.clip(
            np.sqrt(theta_half_deg * BRDF_SAMPLING_RES_THETA_H),
            0,
            BRDF_SAMPLING_RES_THETA_H - 1,
        )
        id_theta_d = np.clip(
            theta_d / (np.pi * 0.5) * BRDF_SAMPLING_RES_THETA_D,
            0,
            BRDF_SAMPLING_RES_THETA_D - 1,
        )
        if phi_d < 0:
            phi_d=phi_d+np.pi
        
        id_phi_d = np.clip(
            phi_d / np.pi * BRDF_SAMPLING_RES_PHI_D / 2,
            0,
            BRDF_SAMPLING_RES_PHI_D // 2 - 1,
        )
        
        # return the value with nearest index value, officially used in the Merl BRDF
        id_theta_h = int(id_theta_h)
        id_theta_d = int(id_theta_d)
        id_phi_d = int(id_phi_d)
        # print(id_theta_d, id_phi_d, id_theta_h)
        return id_phi_d,id_theta_h,id_theta_d

def readMERLBRDF(filename):
    """Reads a MERL-type .binary file, containing a densely sampled BRDF
    
    Returns a 4-dimensional array (phi_d, theta_h, theta_d, channel)"""
    print("Loading MERL-BRDF: ", filename)
  #  dims=[int(BRDF_SAMPLING_RES_PHI_D/2),BRDF_SAMPLING_RES_THETA_H,BRDF_SAMPLING_RES_THETA_D]

    try: 
        f = open(filename, "rb")
        dims = np.fromfile(f,np.int32,3)
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print("Cannot read file: ", path.basename(filename))
        return
        
    BRDFVals = np.swapaxes(np.reshape(vals,(dims[2], dims[1], dims[0], 3),'F'),1,2)
    BRDFVals *= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    BRDFVals[BRDFVals<0] = -1
    
    return BRDFVals
def saveMERLBRDF(filename,BRDFVals,shape=(180,90,90),toneMap=True):
    "Saves a BRDF to a MERL-type .binary file"
    print("Saving MERL-BRDF: ", filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    if(BRDFVals.shape != (np.prod(shape),3) and BRDFVals.shape != shape+(3,)):
        print("Shape of BRDFVals incorrect")
        return
        
    #Do MERL tonemapping if needed
    if(toneMap):
        BRDFVals /= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    
    #Are the values not mapped in a cube?
    if(BRDFVals.shape[1] == 3):
        BRDFVals = np.reshape(BRDFVals,shape+(3,))
        
    #Vectorize:
    vec = np.reshape(np.swapaxes(BRDFVals,1,2),(-1),'F')
    shape = [shape[2],shape[1],shape[0]]
    
    try: 
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file: ", path.basename(filename))
        return
        

def latent_code(mean, std, num_samples,dim):
    return np.random.normal(mean, std, size=(num_samples, dim))
def uwrap_edge(i, edge):
    if i >= edge:
        i = edge - 1
    elif i < 0:
        i = 0
    return i
def parse_name(path):
        return basename(path)[:-len('.binary')]

def compute_p22_smith(brdf, res):
    """Computes the p22 function for a MERL BRDF"""
    cnt=int(res-1)
    dtheta=np.sqrt(0.5*np.pi)/cnt
    km = np.zeros((cnt, cnt))
    p22 = np.zeros(res)
    for i in range(cnt):
        tmp = i / cnt
        theta = tmp * np.sqrt(np.pi * 0.5)
        theta_o = theta * theta
        cos_theta_o = np.cos(theta_o)
        tan_theta_o = np.tan(theta_o)
        xyz=phithera2vec(theta_o,0)
        theta_h, theta_d, phi_d = std_half_diff(xyz, xyz)

        id_phi_d,id_theta_h,id_theta_d =half_diff_look_brdf(theta_h, theta_d,phi_d)
        fr=brdf[id_phi_d,id_theta_h,id_theta_d, :]
        fr[fr<0]=0
        intensity=0.2126*fr[0]+0.7152*fr[1]+0.0722*fr[2]
        kij_tmp=(dtheta*np.power(cos_theta_o,6.0))*(8.0*intensity)
        for j in range(cnt):
            dphi_h=np.pi/180.0
            tmp=j/cnt
            theta = tmp * np.sqrt(np.pi * 0.5)
            theta_h = theta * theta
            cos_theta_h = np.cos(theta_h)
            tan_theta_h = np.tan(theta_h)
            tan_product = tan_theta_h * tan_theta_o
            nint = 0.0
            for phi_h in np.arange(0.0, 2.0 * np.pi+dphi_h, dphi_h):  
                nint +=max(1.0,tan_product*np.cos(phi_h))
            nint=nint*dphi_h
            km[j,i]=theta*kij_tmp*nint*tan_theta_h/(cos_theta_h*cos_theta_h)

 #   eigenvalues, eigenvectors= np.linalg.eig(km)
    ev=eigenvector(4,km,cnt)
    for ii in range(ev.shape[0]):
        p22[ii]=1e-2*ev[ii]      
    p22[ev.shape[0]]=0    
        
    return p22
def eigenvector(iterations,matrix,size):
    j=0
    vec = [np.ones(size), np.zeros(size)]
    for i in range(iterations):
        transform(matrix,vec[j],vec[1-j])
        j=1-j

    return vec[j]
def transform(matrix,v,out):
    out[:]=0
    for j in range(v.shape[0]):
        for i in range(v.shape[0]):
            out[j]=out[j]+matrix[i,j]*v[i]
    
def normalize_p22(p22):
    """Normalizes the p22 function"""
    ntheta = 128
    dphi = 2.0 * np.pi
    dtheta = np.pi / ntheta
    nint = 0.0
    resultion=90
    for i in range(ntheta):
        u = i / ntheta  # in [0,1)
        theta_h = u * u * 0.5 * np.pi
        r_h = np.tan(theta_h)
        cos_theta_h = np.cos(theta_h)
        uu=np.sqrt(2.0*np.arctan(r_h)/np.pi)
        frac,intpart=np.modf(uu*resultion-uu)
        intpart=int(intpart)
        i1=uwrap_edge(int(intpart),int(resultion))
        i2=uwrap_edge(int(intpart+1),int(resultion))
       
        p1=p22[i1]
        p2=p22[i2]
       
        p22_r=p1+(p2-p1)*frac

        nint=nint+(u * p22_r * r_h) / (cos_theta_h * cos_theta_h) 
    nint*=dtheta*dphi
    nint=1.0/nint
    for j in range(p22.shape[0]):
        p22[j]=p22[j]*nint

    return p22
def fit_ggx_parameters_t(MeasuredBRDF):
    """Fits the GGX parameters to a MERL BRDF"""
    merl_alpha = 0.5
    ntheta = 128
    dtheta = np.pi / ntheta
    nint = 0.0
    resultion=90
    m_p22=compute_p22_smith(MeasuredBRDF, 90)
    m_p22=normalize_p22(m_p22)
    for i in range(ntheta):
        u=i/ntheta
        theta_h=u*u*np.pi/2

        cos_theta_h = np.cos(theta_h)
        r_h=np.tan(theta_h)
        r_h_sq=r_h*r_h
        
        uu=np.sqrt(2.0*np.arctan(r_h)/np.pi)
        frac,intpart=np.modf(uu*resultion-uu)
        intpart=int(intpart)
        i1=uwrap_edge(int(intpart),int(resultion))
        i2=uwrap_edge(int(intpart+1),int(resultion))
      
        p1=m_p22[i1]
        p2=m_p22[i2]
        p22_r=p1+(p2-p1)*frac
        nint=nint+(u*r_h_sq*p22_r)/(cos_theta_h*cos_theta_h)        
    nint=nint*dtheta*4.0
    merl_alpha=nint
    phi_a=0.0
    cos_phi_a=np.cos(phi_a)
    sin_phi_a=np.sin(phi_a)
    cos_phi_a_sq=2.0*cos_phi_a*cos_phi_a-1.0
    a1_sqr=merl_alpha*merl_alpha
    a2_sqr=merl_alpha*merl_alpha
    tmp1=a1_sqr+a2_sqr
    tmp2=a1_sqr-a2_sqr
    ax=np.sqrt(0.5*(tmp1+tmp2*cos_phi_a_sq))
    ay=np.sqrt(0.5*(tmp1-tmp2*cos_phi_a_sq))
    rho=(a2_sqr-a1_sqr)*sin_phi_a*cos_phi_a/(ax*ay)

    return merl_alpha ,ax,ay,rho   
def fit_ggx_parameters(MeasuredBRDF):
    """Fits the GGX parameters to a MERL BRDF"""
    m_alpha=0.4
    return m_alpha
def saveMERLBRDF(filename,BRDFVals,shape=(180,90,90),toneMap=True):
    "Saves a BRDF to a MERL-type .binary file"
    print("Saving MERL-BRDF: ", filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    if(BRDFVals.shape != (np.prod(shape),3) and BRDFVals.shape != shape+(3,)):
        print("Shape of BRDFVals incorrect")
        return
        
    #Do MERL tonemapping if needed
    if(toneMap):
        BRDFVals /= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    
    #Are the values not mapped in a cube?
    if(BRDFVals.shape[1] == 3):
        BRDFVals = np.reshape(BRDFVals,shape+(3,))
        
    #Vectorize:
    vec = np.reshape(np.swapaxes(BRDFVals,1,2),(-1),'F')
    shape = [shape[2],shape[1],shape[0]]
    
    try: 
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file: ", path.basename(filename))
        return
        
if __name__ == "__main__":
    #Test the functions
    BRDFVals = readMERLBRDF("/mnt/symphony/wen/spectral_brdfs/Merl_BRDF_database/BRDFDatabase/brdfs/alum-bronze.binary")
    alpha = fit_ggx_parameters_t(BRDFVals)
    p22=compute_p22_smith(BRDFVals, 90)
    1
    saveMERLBRDF("test.binary",BRDFVals)
    print("Done")