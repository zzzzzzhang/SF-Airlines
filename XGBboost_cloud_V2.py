
# coding: utf-8

# In[27]:


import numpy as np
import scipy.io as scio
import h5py
from sklearn.externals import joblib
import warnings
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.basemap import maskoceans

np.set_printoptions(suppress= True)
warnings.filterwarnings('ignore')

# load models
xgb_clf = joblib.load('model/xgb_clf_0.9083.m')
xgb_r = joblib.load('model/xgb_r_1386.7171.m')


# In[28]:


# path
filename = r'D:\Himawari\AHI8_OBI_4000M_NOM_20190223_0020.hdf'
path_out = r'D:\Himawari\out'


# In[29]:


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


# In[30]:


# readfile
data_hdf = h5py.File(filename,'r')

# assign 16 features
features = np.ones((2750,2750,16),dtype = 'float32')
features[:,:,0] = data_hdf['NOMChannelVIS0064_4000'][:].astype('float16')/10000
features[:,:,1] = data_hdf['NOMChannelVIS0086_4000'][:].astype('float16')/10000
features[:,:,2] = data_hdf['NOMChannelVIS0160_4000'][:].astype('float16')/10000
features[:,:,3] = data_hdf['NOMChannelVIS0230_4000'][:].astype('float16')/10000
features[:,:,4] = data_hdf['NOMChannelIRX0390_4000'][:].astype('float16')/100
features[:,:,5] = data_hdf['NOMChannelIRX0620_4000'][:].astype('float16')/100
features[:,:,6] = data_hdf['NOMChannelIRX0700_4000'][:].astype('float16')/100
features[:,:,7] = data_hdf['NOMChannelIRX0730_4000'][:].astype('float16')/100
features[:,:,8] = data_hdf['NOMChannelIRX0860_4000'][:].astype('float16')/100
features[:,:,9] = data_hdf['NOMChannelIRX0960_4000'][:].astype('float16')/100
features[:,:,10] = data_hdf['NOMChannelIRX1040_4000'][:].astype('float16')/100
features[:,:,11] = data_hdf['NOMChannelIRX1120_4000'][:].astype('float16')/100
features[:,:,12] = data_hdf['NOMChannelIRX1230_4000'][:].astype('float16')/100
features[:,:,13] = data_hdf['NOMChannelIRX1330_4000'][:].astype('float16')/100
features[:,:,14] = data_hdf['NOMSatelliteZenith'][:].astype('float16')/100
features[:,:,15] = data_hdf['NOMSunZenith'][:].astype('float16')/100
features[:,:,0] = features[:,:,0]/np.cos(np.deg2rad(features[:,:,15]))

# close & clear
data_hdf.close()
del data_hdf


# In[31]:


# do some mask
mask = np.ones((2750,2750),dtype= 'bool')
mask = np.where(features[:,:,0]< 0.01,True,False)
for i in range(16):
    features[:,:,i] = np.ma.masked_array(features[:,:,i],mask)


# In[32]:


# classify
features_reshaped = features.reshape(-1,16)
cloud_clfed = xgb_clf.predict(features_reshaped)
index = np.where(cloud_clfed == 1)


# In[33]:


# regress
cloud_clfed[index] = xgb_r.predict(features_reshaped[index])


# In[50]:


# reshape
cloud_clfed = cloud_clfed.reshape((2750,2750))
cloud_clfed = np.where(cloud_clfed < 0, 0, cloud_clfed)
# cloud_clfed = np.ma.masked_array(cloud_clfed,mask)


# In[35]:


# lon,lat,mask 掉反射率小于1%的
lat, lon = np.meshgrid(range(2750), range(2750))
lat = np.ma.array(lat,mask= mask)
lon = np.ma.array(lon,mask= mask)
# index_lonlat = np.where(mask == False)
lons, lats = get_LonLat(lon,lat)
lons[lons > 180] -= 360


# In[54]:


# 画图
fig = plt.figure(1,figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111)
m_1 = Basemap( projection = 'ortho', lat_0 = 0, lon_0 = 140 )
m_1.drawcoastlines(linewidth= 0.5)
m_1.drawmeridians(np.arange(0,360,15))
m_1.drawparallels(np.arange(-90,90,15))

X,Y = m_1(lons,lats)
cloud_clfed = np.ma.masked_array(cloud_clfed, mask = X.mask)
m_1.contourf(X,Y,cloud_clfed)

plt.colorbar(ax = ax)
# m_1.imshow(cloud_clfed[::-1])
fig.savefig(path_out + '\\cloud.jpg')
plt.show()


# In[ ]:


# lat, lon = np.meshgrid(range(sunzenith.shape[0]), range(sunzenith.shape[1]))
# lons,lats = get_LonLat(lon,lat)
f_cloud = h5py.File(path_out+'\\cloud.hdf','w')
f_cloud['cloud_clfed'] = cloud_clfed
f_cloud['lons'] = lons
f_cloud['lats'] = lats
f_cloud.close() 

