import os
import numpy as np
from netCDF4 import Dataset
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

#path
def sort_file_by_time(file_path):
    files = os.listdir(file_path)
    if not files:
        return
    else:
		#对files进行排序.x是files的元素,:后面的是排序的依据. x只是文件名,所以要带上join
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return files[-1]
		
path_in = r'/data/liuting/tangjie/hd8data/B03'
path_out = '/data/himawari_cloud_top/out'

#filename
filename = os.path.join(path_in,sort_file_by_time(path_in))
Year = int(filename.split('/')[-1].split('_')[2][:4])
Month = int(filename.split('/')[-1].split('_')[2][4:6])
Day = int(filename.split('/')[-1].split('_')[2][6:])
Hour = int(filename.split('/')[-1].split('_')[3][:2])
Minute = int(filename.split('/')[-1].split('_')[3][2:4])
filename_out = 'AHI8_TOP_' + filename.split('/')[-1].split('_')[2] + '_' + filename.split('/')[-1].split('_')[3]


data = Dataset(filename)
StartLon = data.variables['longitude'][:].data[0]
StartLat = data.variables['latitude'][:].data[0]
EndLon = data.variables['longitude'][:].data[-1]
EndLat = data.variables['latitude'][:].data[-1]
XNumGrids = data.variables['longitude'].shape[0]
YNumGrids = data.variables['latitude'].shape[0]
XReso =  (EndLon - StartLon)/(XNumGrids - 1)
YReso =  (StartLat - EndLat)/(YNumGrids - 1)
lons = data.variables['longitude'][:].data
lats = data.variables['latitude'][:].data
lons,lats = np.meshgrid(lons,lats)
# assign features
features = np.ones((YNumGrids,XNumGrids,14),dtype = 'float32')
features[:,:,0] = data.variables['albedo'][:] if 'albedo' in data.variables.keys() else  data.variables['tbb'][:]


dirs_needed = ['B04','B05','B06','B07','B08','B09','B10','B12','B13','B14','B15','B16']
path_in = r'/data/liuting/tangjie/hd8data/'
# read one file and assign 14 features
for i,dir_needed in enumerate(dirs_needed):
    filepath = os.path.join(path_in,dir_needed)
    filename = os.path.join(filepath,sort_file_by_time(filepath))
    data = Dataset(filename)
    features[:,:,i+1] = data.variables['albedo'][:] if 'albedo' in data.variables.keys() else  data.variables['tbb'][:]
features *= 100

# do some mask
mask = np.where(features[:,:,0] <= 0,True,False)
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

# 画图
fig = plt.figure(1,figsize=(8,8),dpi = 100)

ax = fig.add_subplot(111)
m_1 = Basemap( llcrnrlon = StartLon, llcrnrlat = EndLat, urcrnrlon = EndLon, urcrnrlat = StartLat, 
               projection = 'cyl')
m_1.drawcoastlines(linewidth= 0.5)
m_1.drawmeridians(np.arange(StartLon,EndLon + 0.1,10.),labels=[0, 0, 0, 1],fontsize=10)
m_1.drawparallels(np.arange(EndLat,StartLat + 0.1,10.),labels=[1, 0, 0, 0],fontsize=10)

X,Y = m_1(lons,lats)
cs = m_1.contourf(X,Y,cloud_clfed)
cbar  = m_1.colorbar(cs,'bottom',pad = '5%')
cbar.set_label('H of cloud /m')
fig.savefig(path_out + '/figure/' + filename_out + '.jpg')

with open(path_out + '/bin/' + filename_out + '.bin','wb') as f:
    ss = 'Bottom={},Coordination=GEO,DataType=3,Day={},Dimension=1,Forecast=000,Height={},Hour={},Invalid=0.0000,Left={},LevelUnit=,LevelValue=,Levels=1,Machine=LittleEnding,Minute={},Month={},NorthtoSouth=True,Offset=0,ProductId=CLOUDTOP,Ratio=1.0000,ResolutionCell=0,Resolution_x={},Resolution_y={},Right={},Secod=00,SiteCode=SAT,Storage=CFORMAT,Top={},Width={},Year={}'.format('%.4f'%EndLat,
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
    f.write(sss_b)
    cloud_clfed.tofile(f)
