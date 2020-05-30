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
#print joints
joint_connections = [[0,1],[1,13],[2,3],[3,13],[4,5],[5,13],[6,7],[7,13],[8,9],[9,10],[10,13],[11,13],[12,13]]
plt.imshow(Seq1_0[1][0].dpt)
plt.scatter(joints[:,0], joints[:,1], c="#ffffff")


# from: src/main_blender_semiauto.py:516
trainSeqs = [Seq1_0]

#ind = 0
#for i in trainSeqs[0].data:
#    print len(joints)
#    jt = joints[ind]
#    print "jt.shape value:", jt.shape[0]
#    jtI = di.joints3DToImg(jt)
#    print "jt.shape value:", jt.shape[0]
#    for joint in range(jt.shape[0]):
#        t=transformPoint2D(jtI[joint],i.T)
#        print "jtI[joint]", jtI[joint]
#        jtI[joint,0] = t[0]
#        jtI[joint,1] = t[1]

#for i in joint_connections:
#    print joints[i,0]
#    print joints[i,1]
#    plt.plot(joints[i,0], joints[i,1])
#plt.show()
y = [ 33.90531, 34.073704, 731.0689  ]

for i in joints:
    print i, 'gee'
    print np.sqrt((i[0] - y[0]) ** 2 + (i[1] - y[1]) ** 2)


jtI = [[ 33.88245, 34.448925, 738.46265 ],
       [ 37.846134, 52.80623, 743.1083  ],
       [ 51.888233, 21.79377, 742.80084 ],
       [ 49.823246, 42.970436, 759.5946  ],
       [ 64.68135, 16.77317, 744.2638  ],
       [ 59.93621, 40.61108, 753.8083  ],
       [ 77.378494, 23.862257, 741.30994 ],
       [ 73.54582, 45.334587, 753.279   ],
       [ 92.23913, 65.36187, 736.27814 ],
       [ 89.35941, 74.39116,743.83435 ],
       [ 80.82718, 86.72897, 765.5444  ],
       [ 43.290928, 105.28259, 740.7788  ],
       [ 58.01942, 108.86447, 764.6099  ],
       [ 58.429058, 73.660675, 771.86865 ]]
print (jtI)
print jtI[0]
ind = 0
for i in trainSeqs[0].data:
    joints = i.gtcrop
    print joints

    print np.sqrt(joints[0], jtI[0])
    #jt = joints[ind]
    #print joints[0], joints[0]


#joints3d = Seq1_0[1][0].gt3Dorig
#for i in joint_connections:
    #print np.sqrt((joints3d[i[0],0] - joints3d[i[1],0])**2 +
    #        (joints3d[i[0],1] - joints3d[i[1], 1])**2 +
    #        (joints3d[i[0],2] - joints3d[i[1], 2])**2 )

    #print joints3d[i[0]]
    #print joints3d[i[1]]
    #print joints3d[i[0], 0]
    #print np.sqrt((joints3d[i[0], 0] - y[0]) ** 2 +
    #        (joints3d[i[0], 1] - y[1]) ** 2 )
