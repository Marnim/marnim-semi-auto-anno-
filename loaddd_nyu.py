import matplotlib.pyplot as plt
from data.importers import NYUImporter
import numpy as np
from PIL import Image
#from src.crop3d import HandDetectorICG
from scipy import io

#data = np.load("/home/giffy/Downloads/cvpr14_MSRAHandTrackingDB/Subject1/000023_depth.npy")
#joint = np.loadtxt("/home/giffy/Downloads/cvpr14_MSRAHandTrackingDB/Subject1/000023_joint.txt")
#joint = joint.reshape(21,3)
#plt.scatter(joint[:,0], joint[:,1])
#plt.imshow(data)
#plt.show()
#handDetector = HandDetectorICG()
basepath = "/home/marnim/Documents/Datasets/nyu_hand_dataset_v2/NYU_samples/"

# img = Image.open(filename)
# assert len(img.getbands()) == 3
# r, g, b = img.split()
# r = np.asarray(r, np.int32)
# g = np.asarray(g, np.int32)
# b = np.asarray(b, np.int32)
# dpt = np.bitwise_or(np.left_shift(g,8),b)
# labelMat = io.loadmat("/home/marnim/Desktop/joint_data.mat")
#
# joints3D = labelMat['joint_xyz'][1 - 1]
# print(joints3D.shape)
# plt.imshow(dpt)
# print(dpt.shape)
# joints3D[:, :, 0] = (joints3D[:, :, 0] * 588.03 / joints3D[:, :, 2] + 320)
# joints3D[:, :, 1] = 480-(joints3D[:, :, 1] * 587.07 / joints3D[:, :, 2] + 240)

di = NYUImporter(basepath)
Seq1_0 = di.loadSequence('train', shuffle=False)
joints = Seq1_0[1][0].gtcrop
print joints.shape
joint_connections = [[0,1],[1,13],[2,3],[3,13],[4,5],[5,13],[6,7],[7,13],[8,9],[9,10],[10,13],[11,13],[12,13]]
plt.imshow(Seq1_0[1][0].dpt)
plt.scatter(joints[:,0], joints[:,1], c="#ffffff")
for i in joint_connections:
    plt.plot(joints[i,0], joints[i,1])
plt.show()

joints3d = Seq1_0[1][0].gt3Dorig
for i in joint_connections:
    print np.sqrt((joints3d[i[0],0] - joints3d[i[1],0])**2 +
            (joints3d[i[0],1] - joints3d[i[1], 1])**2 +
            (joints3d[i[0],2] - joints3d[i[1], 2])**2 )

