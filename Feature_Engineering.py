
# coding: utf-8

# In[ ]:


import numpy as np
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore',module='sklearn*')
np.set_printoptions(suppress= True)


# In[ ]:


data = np.loadtxt(r'D:\Himawari\ahi-agri.txt')


# In[ ]:


class cloud():
    data = 0
    samples_clf = 0
    samples_reg = 0
    def __init__(self,data):
        self.data = data
        self.samples_clf = 0
        self.samples_reg = 0
    def clf(self):
        samples_clf = self.data[:,[13,14,15,16,3,4,5,6,7,8,9,10,11,12,20,18,2]].copy()
        samples_clf[:,:-1] *= 100
        samples_clf[:,-1] *= 1000
        index = np.where((samples_clf == 65534.99799999999)| (np.isnan(samples_clf)))[0]
        index = list(set(index))
        samples_clf = np.delete(samples_clf,index,axis= 0)
        np.random.shuffle(samples_clf)
        samples_clf = samples_clf[:100000]
        samples_clf[np.where(samples_clf[:,-1]!= 0),-1] = 1
        self.samples_clf = samples_clf
        return self.samples_clf
    def reg(self):
        samples_reg = self.data[:,[13,14,15,16,3,4,5,6,7,8,9,10,11,12,20,18,2]].copy()
        samples_reg[:,:-1] *= 100
        samples_reg[:,-1] *= 1000
        index = np.where((samples_reg == 65534.99799999999)| (np.isnan(samples_reg)))[0]
        index = list(set(index))
        samples_reg = np.delete(samples_reg,index,axis= 0)
        index = np.where(samples_reg[:,-1] == 0)[0]
        samples_reg = np.delete(samples_reg,index,axis= 0)
        np.random.shuffle(samples_reg)
        samples_reg = samples_reg[:50000]
        self.samples_reg = samples_reg
        return self.samples_reg


# In[ ]:


cloud_himawari8 = cloud(data)


# In[ ]:


samples_clf = cloud_himawari8.clf()


# In[ ]:


samples_reg = cloud_himawari8.reg()


# In[ ]:


#交叉验证
x_train,x_test,y_train,y_test = train_test_split(samples_clf[:,:-1],samples_clf[:,-1],test_size = 0.25,random_state = 10)


# In[ ]:


# 微调训练
xgb_clf = XGBClassifier(max_depth= 13, 
                        learning_rate= 0.1, 
                        n_estimators= 100, 
                        n_jobs= -1, 
                        gamma= 0.11, 
                        reg_alpha= 0, 
                        reg_lambda= 3,
                        min_child_weight = 10,
                        subsample = 0.7,
                        colsample_bytree = 1,
                        seed= 10)
xgb_clf.fit(x_train,y_train)
# 预测,测试集
y_pred = xgb_clf.predict(x_test)
accuracy_test = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy(test) : %.4f" %accuracy_test)
# 预测,训练集
y_pred = xgb_clf.predict(x_train)
accuracy_train = metrics.accuracy_score(y_train, y_pred)
print ("Accuracy(train) : %.4f" %accuracy_train)


# In[ ]:


#交叉验证
x_train,x_test,y_train,y_test = train_test_split(samples_reg[:,:-1],samples_reg[:,-1],test_size = 0.25,random_state = 10)


# In[ ]:


# 微调训练
xgb_reg = XGBRegressor(max_depth= 12, 
                        learning_rate= 0.1, 
                        n_estimators= 100, 
                        n_jobs= -1, 
                        gamma= 0.11, 
                        reg_alpha= 0, 
                        reg_lambda= 3,
                        min_child_weight = 20,
                        subsample = 1,
                        colsample_bytree = 1,
                        seed= 10)
xgb_reg.fit(x_train,y_train)
# 预测,测试集
y_pred = xgb_reg.predict(x_test)
accuracy_test = metrics.mean_squared_error(y_test, y_pred)**0.5
print ("Accuracy(test) : %.4f" %accuracy_test)
# 预测,训练集
y_pred = xgb_reg.predict(x_train)
accuracy_train = metrics.mean_squared_error(y_train, y_pred)**0.5
print ("Accuracy(train) : %.4f" %accuracy_train)


# In[ ]:


joblib.dump(xgb_clf,r'model_d&n/xgb_clf_089.m')
joblib.dump(xgb_reg,r'model_d&n/xgb_reg_1517.m')

