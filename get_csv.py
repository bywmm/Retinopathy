import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob

base_image_dir = "D:/dataset/tangniaobing/"
retina_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
retina_df['PatientId'] = retina_df['image'].map(lambda x: x.split('_')[0])
retina_df['path'] = retina_df['image'].map(lambda x: os.path.join(base_image_dir,
                                                         'train/{}.jpeg'.format(x)))
retina_df['exists'] = retina_df['path'].map(os.path.exists)
print(retina_df['exists'].sum(), 'images found of', retina_df.shape[0], 'total')
retina_df['eye'] = retina_df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)
from keras.utils.np_utils import to_categorical
retina_df['level_cat'] = retina_df['level'].map(lambda x: to_categorical(x, 1+retina_df['level'].max()))

retina_df.dropna(inplace = True)
retina_df = retina_df[retina_df['exists']]

# print(retina_df.sample(n=3))
print(retina_df.head())

# Examine the distribution of eye and severity
retina_df[['level', 'eye']].hist(figsize=(10, 5))
plt.show()

# Split Data into Training and Validation
from sklearn.model_selection import train_test_split
rr_df = retina_df[['PatientId','level']].drop_duplicates()
print(rr_df.head())

train_ids, valid_ids = train_test_split(rr_df['PatientId'],
                                   test_size = 0.25,
                                   random_state = 2018,
                                   stratify = rr_df['level'])
raw_train_df = retina_df[retina_df['PatientId'].isin(train_ids)]
valid_df = retina_df[retina_df['PatientId'].isin(valid_ids)]
print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
raw_train_df[['level', 'eye']].hist()

# Balance the distribution in the training set
train_df = raw_train_df.groupby(['level', 'eye']).apply(lambda x: x.sample(3000, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
# train_df[['level', 'eye']].hist(figsize = (10, 5))
# plt.show()

valid_df.to_csv('valid.csv')
train_df.to_csv('train.csv')
