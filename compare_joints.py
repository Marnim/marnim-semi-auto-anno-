import numpy as np
import struct
f = np.load("/home/marnim/Documents/Projects/semi-auto-anno-new/src/eval/BLEN_SA/new_joints.npz")
print f["new2D"].shape
print f["new2D"][0]
print f["new3D"].shape
print f["new3D"][0]

f = np.load("/home/marnim/Documents/Projects/semi-auto-anno-new/src/eval/msra/new_joints.npz")
print "msra", f["new2D"].shape
print f["new2D"][499]
print f["new3D"].shape
print f["new3D"][499]


file = '/home/marnim/Documents/Datasets/Blender_dataset/Blender/hpseq_loop_mv/c00_00000000_depth_0002.npy'
with open(file, mode='rb') as f:
    data = f.read()
    data = struct.unpack("i" * ((len(data) - 0) // 4), data[:])
    print data
    data = np.asarray(data)
    print data.shape
    data = data.reshape((1, 240, 320))
    print data
    f.close()