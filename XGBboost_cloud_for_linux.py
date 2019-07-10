import numpy as np
import h5py
import glob
from sklearn.externals import joblib
import warnings
import struct
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

np.set_printoptions(suppress= True)
warnings.filterwarnings('ignore')


# load models
xgb_clf = joblib.load('model/xgb_clf_087.m')
xgb_r = joblib.load('model/xgb_reg_1730.m')

# load config
with open('config','r') as f:
    text = f.readline()
    StartLon = float(text.split('=')[1][:-1])
    text = f.readline()
    StartLat = float(text.split('=')[1][:-1])
    text = f.readline()
    XReso = float(text.split('=')[1][:-1])
    text = f.readline()
    EndLon = float(text.split('=')[1][:-1])
    text = f.readline()
    EndLat = float(text.split('=')[1][:-1])
    text = f.readline()
    YReso = float(text.split('=')[1][:-1])
XNumGrids = round((EndLon - StartLon)/XReso)
YNumGrids = round((EndLat - StartLat)/YReso)

# path
datapath='/data/hd8'
robs_path_list = glob.glob(datapath + "/*.hdf")
robs_path_list.sort()
robs_path_list=robs_path_list[::-1]
filename=robs_path_list[0]
path_out = '/data/himawari_cloud_top/out/'
filename_out = 'AHI8_TOP_' + filename.split('_')[4] + '_' + filename.split('_')[5].split('.')[0]

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
    return arr.reshape((YNumGrids,XNumGrids))

lons,lats = np.meshgrid(np.arange(StartLon,EndLon,XReso),np.arange(StartLat,EndLat,YReso))

index_l,index_c = np.round(get_CL(lons,lats))
index_l = index_l.astype('uint16')
index_c = index_c.astype('uint16')
z = list(zip(index_l.reshape(-1),index_c.reshape(-1)))

# readfile
data_hdf = h5py.File(filename,'r')

# assign 16 features
features = np.ones((YNumGrids,XNumGrids,14),dtype = 'float16')
features[:,:,0] = fidx(data_hdf['NOMChannelVIS0064_4000'][:],z)
features[:,:,1] = fidx(data_hdf['NOMChannelVIS0086_4000'][:],z)
features[:,:,2] = fidx(data_hdf['NOMChannelVIS0160_4000'][:],z)
features[:,:,3] = fidx(data_hdf['NOMChannelVIS0230_4000'][:],z)
features[:,:,4] = fidx(data_hdf['NOMChannelIRX0390_4000'][:],z)
features[:,:,5] = fidx(data_hdf['NOMChannelIRX0620_4000'][:],z)
features[:,:,6] = fidx(data_hdf['NOMChannelIRX0700_4000'][:],z)
features[:,:,7] = fidx(data_hdf['NOMChannelIRX0730_4000'][:],z)
features[:,:,8] = fidx(data_hdf['NOMChannelIRX0860_4000'][:],z)
features[:,:,9] = fidx(data_hdf['NOMChannelIRX0960_4000'][:],z)
features[:,:,10] = fidx(data_hdf['NOMChannelIRX1040_4000'][:],z)
features[:,:,11] = fidx(data_hdf['NOMChannelIRX1120_4000'][:],z)
features[:,:,12] = fidx(data_hdf['NOMChannelIRX1230_4000'][:],z)
features[:,:,13] = fidx(data_hdf['NOMChannelIRX1330_4000'][:],z)
# features[:,:,14] = fidx(data_hdf['NOMSatelliteZenith'][:],z)
sunZenith = fidx(data_hdf['NOMSunZenith'][:],z)
features[:,:,0] = features[:,:,0]/np.cos(np.deg2rad(sunZenith))

# close & clear
data_hdf.close()
del data_hdf

# do some mask
mask = np.ones(features[:,:,3].shape,dtype= 'bool')
mask = np.where(sunZenith == 65535,True,False)
for i in range(14):
    features[:,:,i] = np.ma.masked_array(features[:,:,i],mask)


# classify
features_reshaped = features.reshape(-1,14)
cloud_clfed = xgb_clf.predict(features_reshaped)
index = np.where(cloud_clfed == 1)


# regress
cloud_clfed[index] = xgb_r.predict(features_reshaped[index])


# reshape
cloud_clfed = cloud_clfed.reshape(features.shape[0:2])
cloud_clfed = np.where(cloud_clfed < 0, 0, cloud_clfed)
cloud_clfed = cloud_clfed.astype('float32')
# cloud_clfed = np.ma.masked_where(cloud_clfed == 0,cloud_clfed)

# figure
fig = plt.figure(1,figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111)
m_1 = Basemap( llcrnrlon = StartLon, llcrnrlat = EndLat, urcrnrlon = EndLon, urcrnrlat = StartLat, 
               projection = 'cyl')
# m_1 = Basemap( projection = 'ortho', lat_0 = 0, lon_0 = 140 )
m_1.drawcoastlines(linewidth= 0.5)
m_1.drawmeridians(np.arange(StartLon,EndLon + 1.,10.),labels=[0, 0, 0, 1],fontsize=10)
m_1.drawparallels(np.arange(EndLat,StartLat + 1.,10.),labels=[1, 0, 0, 0],fontsize=10)

X,Y = m_1(lons,lats)
# cloud_clfed = np.ma.masked_array(cloud_clfed, mask = mask)
cs = m_1.contourf(X,Y,cloud_clfed)
cbar  = m_1.colorbar(cs,'bottom',pad = '5%')
cbar.set_label('H of cloud /m')
# m_1.imshow(cloud_clfed[::-1])
fig.savefig(path_out + '/figure/' + filename_out + '.jpg')
#plt.show()

with open(path_out + '/bin/' + filename_out + '.bin','wb') as f:
    ss = 'Bottom={},Coordination=GEO,DataType=3,Day={},\
	Dimension=1,Forecast=000,Height={},Hour={},Invalid=0.0000,\
	Left={},LevelUnit=,LevelValue=,Levels=1,Machine=LittleEnding,\
	Minute={},Month={},NorthtoSouth=True,Offset=0,ProductId=CLOUDTOP,\
	Ratio=1.0000,ResolutionCell=0,Resolution_x={},Resolution_y={},\
	Right={},Secod=00,SiteCode=SAT,Storage=CFORMAT,Top={},Width={},Year={}'.format('%.4f'%EndLat,
                                                                                   '%02d'%Day,
                                                                                   '%04d'%YNumGrids,
                                                                                   '%02d'%Hour,
                                                                                   '%.4f'%StartLon,
                                                                                   '%02d'%Minute,
                                                                                   '%02d'%Month,
                                                                                   '%.4f'%0.04,
                                                                                   '%.4f'%-0.04,
                                                                                   '%.4f'%EndLon,
                                                                                   '%.4f'%StartLat,
                                                                                   '%04d'%XNumGrids,
                                                                                   '%04d'%Year)
    sss = 'Nowcasting  {}     '.format(len(ss)+len(str(len(ss))) + 17) + ss
    sss_b = sss.encode('ASCII')
    cloud_clfed.tofile(f)

