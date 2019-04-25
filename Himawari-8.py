
# coding: utf-8

# In[1]:


import h5py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
import warnings
# import struct
warnings.filterwarnings('ignore')
np.set_printoptions(suppress= True)


# In[2]:


filename = 'D:\Himawari\AHI8_OBI_4000M_NOM_20190223_0020.hdf'
path_out = 'D:\Himawari\out'
filename_out = 'AHI8_FOG_' + filename.split('_')[4] + '_' + filename.split('_')[5].split('.')[0]


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
up = 110
down = 1120
left = 0
right = 1370
data_39 = data_himawari8['NOMChannelIRX0390_4000'][up:down,left:right] * 0.01
data_112 = data_himawari8['NOMChannelIRX1120_4000'][up:down,left:right]* 0.01
data_064 = data_himawari8['NOMChannelVIS0064_4000'][up:down,left:right] * 0.0001
sunzenith = data_himawari8['NOMSunZenith'][up:down,left:right]

# close
data_himawari8.close()

mask = np.full(sunzenith.shape, True)
DCD = np.full(sunzenith.shape,999,dtype='float16')
# 边角处掩码
masked_sunzenith = np.ma.array(sunzenith, mask=np.where(sunzenith == 65535, True, False))
data_064 = cosR(masked_sunzenith,data_064)


# In[5]:


# 0 - 90 lands
index = np.where((masked_sunzenith <= 90) & (data_064 < 0.45))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 10))] = False
FDI_039 = FDI(np.ma.array(data_39,mask= mask),np.ma.array(data_064,mask= mask))
mask[np.where(FDI_039 >= 8)] = True
FDI_039 = np.ma.array(FDI_039,mask= mask)

# 90 + lands
DCD = np.full(sunzenith.shape,999,dtype= 'float16') #先重置，上步骤用过了
index = np.where((masked_sunzenith > 90) & (masked_sunzenith <= 180))
DCD[index] = data_39[index] - data_112[index]
mask[np.where((DCD <= 0) & (mask == True))] = False 
DCD = np.ma.array(DCD,mask= mask)


# In[6]:


# 暂时没有雾的分级，所以直接用mask绘图
fog_lands = np.where(mask == False,100,0)
fog_lands = np.ma.array(fog_lands,mask= mask)
# lon,lat
lat, lon = np.meshgrid(range(left,right),range(up,down))
lat = np.ma.array(lat,mask= mask)
lon = np.ma.array(lon,mask= mask)
lons_1, lats_1 = get_LonLat(lon,lat)
lons_1[lons_1 > 180] -= 360
# ma oceans
fog_lands = maskoceans(lons_1,lats_1,fog_lands,inlands= True)

lons_1 = np.ma.array(lons_1,mask= fog_lands.mask)
lats_1 = np.ma.array(lats_1,mask= fog_lands.mask)


# In[25]:


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
boxs = np.array([data_064[x_ - 1 : x_ + 2,y_ - 1 : y_ + 2].reshape(9) if x_ > 0 and x_ < 1009 and y_ > 0 and y_ < 1369 else np.ones(9) for x_,y_ in zip(index[0],index[1])])
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


# In[26]:


# lon,lat
lat, lon = np.meshgrid(range(left,right),range(up,down))
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


# In[27]:


# 合并
DCD = data_39 - data_112
DCD[DCD >= 8] = 200
DCD[DCD <  8] = 100

fog = np.where((fog_lands.mask == True)&(fog_oceans.mask == True), 9999, DCD)
fog = np.where((fog == 9999)&(sunzenith != 65535), 0, fog)


# In[36]:


with open(path_out + '\\' + filename_out + '.bin','wb') as f:
    np.array('test').tofile(f)
    fog.data.tofile(f)


# In[29]:


fog = np.ma.masked_where((fog == 9999)|(fog == 0),fog)

lat, lon = np.meshgrid(range(left,right),range(up,down))
lat = np.ma.array(lat,mask= fog.mask)
lon = np.ma.array(lon,mask= fog.mask)
lons, lats = get_LonLat(lon,lat)


# In[30]:


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

fig.savefig(path_out + '\\' + filename_out + '.jpg')
plt.show()


# In[87]:


# f_fog = h5py.File(path_out + '\\' + filename_out + '.hdf','w')
# f_fog['fog'] = fog
# f_fog['lons'] = lons
# f_fog['lats'] = lats
# f_fog.close() 

