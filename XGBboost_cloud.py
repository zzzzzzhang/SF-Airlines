
# coding: utf-8

# In[12]:


import numpy as np
import scipy.io as scio
import h5py
from sklearn.externals import joblib
import warnings
# from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.basemap import interp
# from mpl_toolkits.basemap import maskoceans

np.set_printoptions(suppress= True)
warnings.filterwarnings('ignore')


# In[13]:


# load models
xgb_clf = joblib.load('model/xgb_clf_0.9083.m')
xgb_r = joblib.load('model/xgb_r_1386.7171.m')
# path
filename = r'D:\Himawari\AHI8_OBI_4000M_NOM_20190223_0020.hdf'
path_out = r'D:\Himawari\out'
filename_out = 'AHI8_TOP_' + filename.split('_')[4] + '_' + filename.split('_')[5].split('.')[0]


# In[14]:


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


# In[15]:


# readfile
data_hdf = h5py.File(filename,'r')

# assign 16 features
up = 110
down = 1120
left = 0
right = 1370
features = np.ones((down - up,right - left,16),dtype = 'float32')
features[:,:,0] = data_hdf['NOMChannelVIS0064_4000'][up:down,left:right].astype('float16')/10000
features[:,:,1] = data_hdf['NOMChannelVIS0086_4000'][up:down,left:right].astype('float16')/10000
features[:,:,2] = data_hdf['NOMChannelVIS0160_4000'][up:down,left:right].astype('float16')/10000
features[:,:,3] = data_hdf['NOMChannelVIS0230_4000'][up:down,left:right].astype('float16')/10000
features[:,:,4] = data_hdf['NOMChannelIRX0390_4000'][up:down,left:right].astype('float16')/100
features[:,:,5] = data_hdf['NOMChannelIRX0620_4000'][up:down,left:right].astype('float16')/100
features[:,:,6] = data_hdf['NOMChannelIRX0700_4000'][up:down,left:right].astype('float16')/100
features[:,:,7] = data_hdf['NOMChannelIRX0730_4000'][up:down,left:right].astype('float16')/100
features[:,:,8] = data_hdf['NOMChannelIRX0860_4000'][up:down,left:right].astype('float16')/100
features[:,:,9] = data_hdf['NOMChannelIRX0960_4000'][up:down,left:right].astype('float16')/100
features[:,:,10] = data_hdf['NOMChannelIRX1040_4000'][up:down,left:right].astype('float16')/100
features[:,:,11] = data_hdf['NOMChannelIRX1120_4000'][up:down,left:right].astype('float16')/100
features[:,:,12] = data_hdf['NOMChannelIRX1230_4000'][up:down,left:right].astype('float16')/100
features[:,:,13] = data_hdf['NOMChannelIRX1330_4000'][up:down,left:right].astype('float16')/100
features[:,:,14] = data_hdf['NOMSatelliteZenith'][up:down,left:right].astype('float16')/100
features[:,:,15] = data_hdf['NOMSunZenith'][up:down,left:right].astype('float16')/100
features[:,:,0] = features[:,:,0]/np.cos(np.deg2rad(features[:,:,15]))

# close & clear
# data_hdf.close()
# del data_hdf


# In[16]:


# do some mask
mask = np.ones(features.shape,dtype= 'bool')
mask = np.where(features[:,:,0]< 0.01,True,False)
for i in range(16):
    features[:,:,i] = np.ma.masked_array(features[:,:,i],mask)


# In[17]:


# classify
features_reshaped = features.reshape(-1,16)
cloud_clfed = xgb_clf.predict(features_reshaped)
index = np.where(cloud_clfed == 1)


# In[18]:


# regress
cloud_clfed[index] = xgb_r.predict(features_reshaped[index])


# In[19]:


# reshape
cloud_clfed = cloud_clfed.reshape(features.shape[0:2])
cloud_clfed = np.where(cloud_clfed < 0, 0, cloud_clfed)
# cloud_clfed = np.ma.masked_array(cloud_clfed,mask)


# In[20]:


# lon,lat,mask 掉反射率小于1%的
lat, lon = np.meshgrid(range(left,right),range(up,down))
lat = np.ma.array(lat,mask= mask)
lon = np.ma.array(lon,mask= mask)
# index_lonlat = np.where(mask == False)
lons, lats = get_LonLat(lon,lat)
lons[lons > 180] -= 360


# 等经纬度插值(失败)
# lat = np.arange(10,60.01,0.04)
# lon = np.arange(70,140.01,0.04)

# lons_out, lats_out = np.meshgrid(lon,lat)
# interpFun = interpolate.interp2d(lats[500:600,500:600],lons[500:600,500:600],cloud_clfed[500:600,500:600],kind= 'cubic')
# cloud_clfed = interp(cloud_clfed,lons,lats,lons_out,lats_out)


# In[21]:


# 画图
fig = plt.figure(1,figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111)
m_1 = Basemap( llcrnrlon = 70, llcrnrlat = 10, urcrnrlon = 140, urcrnrlat = 60, 
               projection = 'cyl')
# m_1 = Basemap( projection = 'ortho', lat_0 = 0, lon_0 = 140 )
m_1.drawcoastlines(linewidth= 0.5)
m_1.drawmeridians(np.arange(70.,141.,10.),labels=[0, 0, 0, 1],fontsize=10)
m_1.drawparallels(np.arange(10.,61.,10.),labels=[1, 0, 0, 0],fontsize=10)

X,Y = m_1(lons,lats)
cloud_clfed = np.ma.masked_array(cloud_clfed, mask = X.mask)
cs = m_1.contourf(X,Y,cloud_clfed)

cbar  = m_1.colorbar(cs,'bottom',pad = '5%')
cbar.set_label('H of cloud /m')
# plt.colorbar(ax = ax)
# m_1.imshow(cloud_clfed[::-1])
fig.savefig(path_out + '\\' + filename_out + '.jpg')
plt.show()


# In[25]:


f_cloud = h5py.File(path_out + '\\' + filename_out + '.hdf','w')
f_cloud['cloud_clfed'] = cloud_clfed
f_cloud['lons'] = lons
f_cloud['lats'] = lats
f_cloud.close() 

