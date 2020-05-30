import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# from util.tsne_impl import tsne
# from sklearn.cluster import KMeans

image_file = "eval/blender/image_array.npy"
if not os.path.isfile(image_file):
    pickleCache = 'cache/Blender2Importer_hpseq_loop_mv_0_cache.pkl'
    f = open(pickleCache, 'rb')
    (seqName, data, config) = pickle.load(f)
    f.close()
    data_img = np.zeros((len(data),128,128))
    for i, img in enumerate(data):
        data_img[i]+=img.dpt
    np.save("eval/blender/image_array",data_img)
else:
    image_data = np.load(image_file)

# normalize = image_data.max()



# tsne_file = "eval/blender/tsne_array.npy"
# if not os.path.isfile(tsne_file):
#     labels = np.ones(image_data.shape[0])
#     tsne_data = tsne(image_data.reshape(image_data.shape[0],-1)/normalize, 2, 50, 20.0)
#     np.save("eval/blender/tsne_array",tsne_data)
# else:
#     tsne_data = np.load("eval/blender/tsne_array.npy")
#
#
#
# # pd.display(tsne_df)
# plt.scatter(tsne_data[:,0], tsne_data[:,1], s=0.2)
# plt.scatter(tsne_data[:,0].mean(), tsne_data[:,1].mean())
# plt.show()
# x_mean = tsne_data[:,0].mean()
# y_mean = tsne_data[:,1].mean()
#
# distance = 0.1
# init_image = np.argmin((tsne_data[:,0]-x_mean)**2+(tsne_data[:,1] - y_mean)**2)
# print(init_image)
#
# model = KMeans(n_clusters=int(3040/2), random_state=0).fit(tsne_data)
#
# for i in range(int(3040/2)):
#     print(np.where(model.labels_ == i))
#     plt.scatter(tsne_data[np.where(model.labels_ == i),0],tsne_data[np.where(model.labels_ == i),1], s=0.2)
# plt.show()
