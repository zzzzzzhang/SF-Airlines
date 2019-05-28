
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
import warnings
import struct
# import struct
warnings.filterwarnings('ignore')
np.set_printoptions(suppress= True)


# In[94]:


filename = 'D:\Himawari\AHI8_OBI_4000M_NOM_20190222_2150.hdf'
path_out = 'D:\Himawari\out'
filename_out = 'AHI8_FOG_' + filename.split('_')[4] + '_' + filename.split('_')[5].split('.')[0]


# In[95]:


def get_LonLat(l,c):
    coff = 1375.5
    loff = 1375.5
    cfac = 10233128
    lfac = 10233128
    ea = 6378.137
    eb = 6356.7523
    h = 42164
    lamda_Himawari_8 = 140.68
    x = (np.pi * (c - coff))/(180 * np.exp2(-16) * cfac)
    y = (np.pi * (l - coff))/(180 * np.exp2(-16) * lfac)
    sd = np.sqrt((h * np.cos(x) * np.cos(y))**2 - (np.cos(y)**2 + ea**2/eb**2 * np.sin(y)**2) * ((h**2) - ea**2))
    sn = (h * np.cos(x) * np.cos(y) - sd)/(np.cos(y)**2 + ea**2/eb**2 * np.sin(y)**2)
    s1 = h - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1**2 + s2**2)
    lon = (180/np.pi)*np.arctan(s2/s1) + lamda_Himawari_8
    lat = (180/np.pi)*np.arctan((ea**2/eb**2) * (s3/sxy))
    return lon,lat

def get_CL(lon,lat):
    coff = 1375.5
    loff = 1375.5
    cfac = 10233128
    lfac = 10233128
    ea = 6378.137
    eb = 6356.7523
    h = 42164
    lambda_D = 140.7
    lon = lon*np.pi/180
    lat = lat*np.pi/180
    lambda_e = lon
    phi_e = np.arctan(eb**2 * np.tan(lat)/ea**2)
    re = eb/np.sqrt(1-(ea**2 - eb**2) * np.cos(phi_e)**2/ea**2)
    r1 = h - re * np.cos(phi_e) * np.cos(lambda_e - lambda_D)
    r2 = - re * np.cos(phi_e) * np.sin(lambda_e - lambda_D)
    r3 = re * np.sin(phi_e)
    rn = np.sqrt(r1**2 + r2**2 + r3**2)
    x = np.arctan(-r2/r1)*180/np.pi
    y = np.arcsin(-r3/rn)*180/np.pi
    c = coff + x * 2**(-16) * cfac
    l = loff + y * 2**(-16) * lfac
    return l,c
    
def fidx(data,idx):
    arr = np.array([data[i,j] for i,j in idx],dtype= 'float32')
    return arr.reshape((1250,1750))
    
def FDI(T,R):
    one = 299792458/(np.pi * 3.9**5)
    two = np.exp(14388 / (3.9 * T) - 1)
    return (one/two)/R

def cosR(sunzenith,R):
    return R/ np.cos(np.deg2rad(sunzenith))


# In[96]:


lons,lats = np.meshgrid(np.arange(70,140,0.04),np.arange(60.,10.,-0.04))


# In[97]:


index_l,index_c = np.round(get_CL(lons,lats))
index_l = index_l.astype('uint16')
index_c = index_c.astype('uint16')
z = list(zip(index_l.reshape(-1),index_c.reshape(-1)))


# In[98]:


# open file
data_himawari8 = h5py.File(filename,'r')

# get bt value
data_39 = data_himawari8['NOMChannelIRX0390_4000'][:] * 0.01
data_39 = fidx(data_39,z)
data_112 = data_himawari8['NOMChannelIRX1120_4000'][:]* 0.01
data_112 = fidx(data_112,z)
data_064 = data_himawari8['NOMChannelVIS0064_4000'][:] * 0.0001
data_064 = fidx(data_064,z)
sunzenith = data_himawari8['NOMSunZenith'][:]
sunzenith = fidx(sunzenith,z)

# close
data_himawari8.close()
del z,data_himawari8

mask = np.full(sunzenith.shape, True)
DCD = np.full(sunzenith.shape,999,dtype='float16')
# 边角处掩码
masked_sunzenith = np.ma.array(sunzenith, mask=np.where(sunzenith == 65535, True, False))
data_064 = cosR(masked_sunzenith,data_064)


# In[99]:


# 0 - 90 lands
index = np.where((masked_sunzenith <= 90) & (data_064 < 0.45))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 10))] = False
FDI_039 = FDI(np.ma.array(data_39,mask= mask),np.ma.array(data_064,mask= mask))
mask[np.where(FDI_039 >= 8)] = True
FDI_039 = np.ma.array(FDI_039,mask= mask)


# In[100]:


# 90 + lands
DCD = np.full(sunzenith.shape,999,dtype= 'float16') #先重置，上步骤用过了
index = np.where((masked_sunzenith > 90) & (masked_sunzenith <= 180))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 0) & (mask == True))] = False 
DCD = np.ma.array(DCD,mask= mask)


# In[101]:


# 暂时没有雾的分级，所以直接用mask绘图
fog_lands = np.where(mask == False,100,0)
fog_lands = np.ma.array(fog_lands,mask= mask)
# ma oceans
fog_lands = maskoceans(lons,lats,fog_lands,inlands= True)

lons_1 = np.ma.array(lons,mask= fog_lands.mask)
lats_1 = np.ma.array(lats,mask= fog_lands.mask)


# In[102]:


# oceans
mask = np.full(sunzenith.shape, True)
DCD = np.full(sunzenith.shape,999,dtype='float')
# 边角处掩码
masked_sunzenith = np.ma.array(sunzenith, mask=np.where(sunzenith == 65535, True, False))
# 0-10
index = np.where(masked_sunzenith <=  10)
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 3)&(DCD >= -2))] = False

# 10-15
index = np.where((masked_sunzenith > 10) & (masked_sunzenith <= 15))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 10) & (DCD >= 3))] = False

# 15_80
index = np.where((data_064 >= 0.2) & (data_064 <= 0.45) &  ( masked_sunzenith >15 ) & (masked_sunzenith <= 80))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 20) & (DCD >= 3))] = False
boxs = np.array([data_064[x_ - 1 : x_ + 2,y_ - 1 : y_ + 2].reshape(9) if x_ > 0 and x_ < 1248 and y_ > 0 and y_ < 1748 else np.ones(9) for x_,y_ in zip(index[0],index[1])])
boxs_std = boxs.std(axis= 0)
mask[np.where((boxs_std < 0.5)|(boxs_std > 3.5))] = True

# 80_90
index = np.where((masked_sunzenith > 80) & (masked_sunzenith <= 90))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 3) & (DCD >= -2))] = False

# 90 +
index = np.where((masked_sunzenith > 90) & (masked_sunzenith <= 180))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= -2))] = False


# In[103]:


# ma lands
masklands = (maskoceans(lons,lats,masked_sunzenith,inlands= True).mask == False)
masklands = masklands | mask
# 暂时没有雾的分级，所以直接用mask绘图
fog_oceans = np.where(mask == False,100,0)
fog_oceans = np.ma.array(fog_oceans,mask= masklands)

lons_2 = np.ma.array(lons,mask= fog_oceans.mask)
lats_2 = np.ma.array(lats,mask= fog_oceans.mask)


# In[104]:


# 合并
DCD = data_39 - data_112
DCD[DCD >= 8] = 200
DCD[DCD <  8] = 100

fog = np.where((fog_lands.mask == True)&(fog_oceans.mask == True), 9999, DCD)
fog = np.where((fog == 9999)&(sunzenith != 65535), 0, fog)


# In[105]:


fog = np.ma.masked_where((fog == 9999)|(fog == 0),fog)


# In[106]:


# 画图
fig = plt.figure(1,figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111)
m_1 = Basemap( llcrnrlon = 70, llcrnrlat = 10, urcrnrlon = 140, urcrnrlat = 60, projection = 'cyl')
# m_1 = Basemap( projection = 'ortho', lat_0 = 0, lon_0 = 140 )
m_1.drawcoastlines(linewidth= 1)
m_1.drawmeridians(np.arange(70.,141.,10.),labels=[0, 0, 0, 1],fontsize=10)
m_1.drawparallels(np.arange(10.,61.,10.),labels=[1, 0, 0, 0],fontsize=10)

X,Y = m_1(lons,lats)
cs = m_1.contourf(X,Y,fog)

# fig.savefig(path_out + '\\' + filename_out + '.jpg')


# In[72]:


with open(path_out + '\\' + filename_out + '.bin','wb') as f:
    s = 'filename:' + filename_out + 'time:' + filename_out[9:] + 'lon:70E-140E' + 'lat:10N-60N' + 'resolution:0.04*0.04'
    for i in range(len(s)):
        s_bin = struct.pack('c',s[i].encode('ASCII'))
        f.write(s_bin)
    fog.data.tofile(f)

