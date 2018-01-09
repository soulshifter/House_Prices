# $ 0 U l $ h ! f T 3 r

# IMPORT BASIC LIBRARIES

import numpy as np

import pandas as pd
pd.set_option('display.height', 2000)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', -1)

import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# DATA LOAD

train_set = pd.read_csv("/home/vic/Desktop/Kaggle/House Prices/train.csv")
target = train_set.SalePrice
test_set = pd.read_csv("/home/vic/Desktop/Kaggle/House Prices/test.csv")

# DATA CHECK

#print(train_set.head())
#print(train_set.tail())
#print(train_set.shape)
#print(train_set.describe())
#print(train_set.columns)
#print(test_set.head())
#print(test_set.tail())
#print(test_set.shape)
#print(test_set.describe())
#print(test_set.columns)
#print(train_set.dtypes)


# CHECKING CORRELATION OF FEATURES WITH SALEPRICE AND TAKING ONLY IMPORTANT FEATURES
# PLOTTING HEATMAP

"""corrmat = train_set.corr()
top_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
sns.set(font_scale=0.5)
plt.figure(figsize=(100,100))
g = sns.heatmap(train_set[top_features].corr(),cmap="RdYlGn",annot=True)
#plt.show()"""


# CHECKING FEATURES CO-RELATION

corr = train_set.corr()
corr.sort_values(["SalePrice"],ascending=False,inplace=True)
#print(corr.SalePrice)


# COMBINING DATA
combined = train_set.append(test_set)
target = train_set.SalePrice
combined.drop("SalePrice",axis=1,inplace=True)
combined.reset_index(inplace=True)
combined.drop("index",axis=1,inplace=True)


# DISTRIBUTING NUMERICAL AND CATEGORICAL FEATURES

numfeat = combined.select_dtypes(exclude=["object"]).columns
train_num = combined[numfeat]
#print(train_num.shape)

catfeat = combined.select_dtypes(include=["object"]).columns
train_cat = combined[catfeat]
#print(train_cat.shape)


# HANDLING SKEWNESS

from scipy.stats import skew
train_set["SalePrice"] = np.log1p(train_set["SalePrice"])
skewd_feats = train_set[numfeat].apply(lambda x:skew(x.dropna()))
skewd_feats = skewd_feats[skewd_feats>0.5]
skewd_feats = skewd_feats.index
combined[skewd_feats] = np.log1p(combined[skewd_feats])


# FILLING NULL VALUES AND GETTING DUMMIES

combined = pd.get_dummies(combined)
combined = combined.fillna(combined.mean())
#print(combined.isnull().values.sum())

# DIVIDING TRAIN AND TEST DATA
train = combined[0:train_set.shape[0]]
test = combined[train_set.shape[0]:]
#print(test.shape)
#print(test_set.shape)


# MODELLING

from sklearn.linear_model import LassoCV,Lasso
model_lasso = LassoCV(alphas=[10,1,0.1,0.001,0.0005]).fit(train,target)
lasso_preds = model_lasso.predict(test)

# CREATING CSV FILE
df_op = pd.DataFrame()
df_op["Id"] = test["Id"]
df_op["SalePrice"] = lasso_preds
df_op[["Id","SalePrice"]].to_csv("/home/vic/Desktop/Kaggle/House Prices/output.csv",index=False)




