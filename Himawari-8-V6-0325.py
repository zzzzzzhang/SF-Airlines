
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans


# In[2]:


filename = 'D:\工作\AHI8_OBI_4000M_NOM_20190223_0020.hdf'
path_out = 'D:\工作'


# In[3]:


def get_LonLat(l,c):
    coff = 1375.5
    loff = 1375.5
    cfac = 10233128
    lfac = 10233128
    ea = 6378.137
    eb = 6356.7523
    h = 42164
    lamda_Himawari_8 = 140.7
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

def FDI(T,R):
    one = 299792458/(np.pi * 3.9**5)
    two = np.exp(14388 / (3.9 * T) - 1)
    return (one/two)/R

def cosR(sunzenith,R):
    return R/ np.cos(np.deg2rad(sunzenith))


# In[4]:


# open file
data_himawari8 = h5py.File(filename,'r')

# get bt value
data_39 = data_himawari8['NOMChannelIRX0390_4000'][:] * 0.01
data_112 = data_himawari8['NOMChannelIRX1120_4000'][:] * 0.01
data_064 = data_himawari8['NOMChannelVIS0064_4000'][:] * 0.0001
sunzenith = data_himawari8['NOMSunZenith'][:]

# close
data_himawari8.close()

mask = np.full(sunzenith.shape, True)
DCD = np.full(sunzenith.shape,999,dtype='float')
# 边角处掩码
masked_sunzenith = np.ma.array(sunzenith, mask=np.where(sunzenith == 65535, True, False))
# data_064 = cosR(masked_sunzenith,data_064)


# In[5]:


# 0 - 90 lands
index = np.where((masked_sunzenith < 90) & (data_064 < 0.1))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 10))] = False
FDI_039 = FDI(np.ma.array(data_39,mask= mask),np.ma.array(data_064,mask= mask))
mask[np.where(FDI_039 >= 8)] = True
FDI_039 = np.ma.array(FDI_039,mask= mask)

# 90 + lands
DCD = np.full(sunzenith.shape,999,dtype= 'float') #先重置，上步骤用过了
index = np.where((masked_sunzenith > 90) & (masked_sunzenith <= 180))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 0) & (mask == True))] = False 
DCD = np.ma.array(DCD,mask= mask)


# In[6]:


# 暂时没有雾的分级，所以直接用mask绘图
fog_lands = np.where(mask == False,100,0)
fog_lands = np.ma.array(fog_lands,mask= mask)
# lon,lat
lat, lon = np.meshgrid(range(sunzenith.shape[0]), range(sunzenith.shape[1]))
lat = np.ma.array(lat,mask= mask)
lon = np.ma.array(lon,mask= mask)
lons_1, lats_1 = get_LonLat(lon,lat)
lons_1[lons_1 > 180] -= 360
# ma oceans
fog_lands = maskoceans(lons_1,lats_1,fog_lands,inlands= True)

lons_1 = np.ma.array(lons_1,mask= fog_lands.mask)
lats_1 = np.ma.array(lats_1,mask= fog_lands.mask)


# In[7]:


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
mask[np.where((DCD <= 20) & (DCD >= 3))] = False

# 15_80
index = np.where((data_064 >= 0.2) & (data_064 <= 0.45) &  ( masked_sunzenith >15 ) & (masked_sunzenith <= 80))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 20) & (DCD >= 3))] = False
boxs = np.array([data_064[x_ - 1 : x_ + 2,y_ - 1 : y_ + 2].reshape(9) for x_,y_ in zip(index[0],index[1])])
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


# In[8]:


# lon,lat
lat, lon = np.meshgrid(range(sunzenith.shape[0]), range(sunzenith.shape[1]))
lat = np.ma.array(lat,mask= mask)
lon = np.ma.array(lon,mask= mask)
lons_2, lats_2 = get_LonLat(lon,lat)
lons_2[lons_2 > 180] -= 360
# ma lands
masklands = (maskoceans(lons_2,lats_2,masked_sunzenith,inlands= True).mask == False)
masklands = masklands | mask
# 暂时没有雾的分级，所以直接用mask绘图
fog_oceans = np.where(mask == False,100,0)
fog_oceans = np.ma.array(fog_oceans,mask= masklands)

lons_2 = np.ma.array(lons_2,mask= fog_oceans.mask)
lats_2 = np.ma.array(lats_2,mask= fog_oceans.mask)


# In[9]:


# 画图
fig = plt.figure(1,figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111)
m_1 = Basemap( projection = 'ortho', lat_0 = 0, lon_0 = 140 )
m_1.drawcoastlines(linewidth= 0.5)

x_lands,y_lands = m_1(lons_1,lats_1)
m_1.contourf(x_lands,y_lands,fog_lands)
x_oceans,y_oceans = m_1(lons_2,lats_2)
m_1.contourf(x_oceans,y_oceans,fog_oceans)
# draw lat/lon grid lines every 30 degrees.
m_1.drawmeridians(np.arange(0,360,30))
m_1.drawparallels(np.arange(-90,90,30))
fig.savefig(path_out + '\\fog.jpg')
# plt.show()


# In[10]:


lat, lon = np.meshgrid(range(sunzenith.shape[0]), range(sunzenith.shape[1]))
lons,lats = get_LonLat(lon,lat)
f_fog = h5py.File(path_out+'\\fog.hdf','w')
f_fog['fog_lands'] = fog_lands
f_fog['fog_oceans'] = fog_oceans
f_fog['lons'] = lons
f_fog['lats'] = lats
f_fog.close() 

