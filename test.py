import numpy as np
from scipy.spatial import distance
import pickle
import matplotlib.pyplot as plt
import matlab.engine
import time

pickleCache = 'cache/Blender2Importer_hpseq_loop_mv_0_cache.pkl'
eng = matlab.engine.start_matlab()
start_time = time.time()
tf = eng.isprime(37)
print("--- %s seconds ---" % (time.time() - start_time))
print(tf)
f = open(pickleCache, 'rb')
(seqName, data, config) = pickle.load(f)
f.close()
image_data = np.zeros((len(data),128,128))
joints_2d_crop = np.zeros((len(data),24,3))
joints_3d_crop = np.zeros((len(data), 24, 3))
for i, img in enumerate(data):
    image_data[i] += img.dpt
    joints_2d_crop[i]+=img.gtcrop
    joints_3d_crop[i]+=img.gt3Dcrop

sort_array = np.load("eval/blender/sort_array.npy")
ref_frames = np.load("eval/blender/ref_array.npy")
new_joints = {0:joints_2d_crop[0]}
ref = 0
for i in range(sort_array.shape[0]-1):
    dst = distance.cosine(image_data[sort_array[i]].flatten(), image_data[sort_array[i+1]].flatten())
    if dst == 0:
        new_joints[i+1] = new_joints[i]
    else:
        plt.imshow(image_data[sort_array[i]])
        plt.show()
        plt.imshow(image_data[sort_array[i+1]])
        plt.show()
        break





