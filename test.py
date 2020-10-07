import numpy as np
from scipy.spatial import distance
import pickle
import matplotlib.pyplot as plt
import matlab.engine
import time
import matplotlib
from util.handpose_evaluation import Blender2HandposeEvaluation
def loadDepthMapBlender(path, filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """

    depth = np.load(path+"/"+filename+"depth_0002.npy")
    joint = np.loadtxt(path+"/"+filename+"anno_blender.txt", delimiter = " ")
    joint = joint.reshape(24,3)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0] * 1000.
    elif len(depth.shape) == 2:
        depth *= 1000.
    else:
        raise IOError("Invalid file: {}".format(filename))
    depth[depth > 1000. - 1e-4] = 32001
    # return in mm
    return depth, joint

def joint3DToImg(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = np.zeros((3, ), np.float32)
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0]/sample[2]*fx+ux
    ret[1] = uy-sample[1]/sample[2]*fy
    ret[2] = sample[2]
    return ret

def joints3DToImg(sample, fx = 241.42, fy = 241.42, ux = 160., uy = 120.):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = joint3DToImg(sample[i], fx, fy, ux, uy)
    return ret

# pickleCache = 'cache/Blender2Importer_hpseq_loop_mv_0_cache.pkl'
pickleCache = 'cache/Blender2Importer_hpseq_loop_mv_0_cache.pkl'
eng = matlab.engine.start_matlab()
start_time = time.time()
tf = eng.isprime(37)
print("--- %s seconds ---" % (time.time() - start_time))
print(tf)
f = open(pickleCache, 'rb')
(seqName, data, config) = pickle.load(f)
f.close()
numjoints = 24
# numjoints = 21
image_data = np.zeros((len(data),128,128))
joints_2d_crop = np.zeros((len(data),numjoints,3))
joints_3d_crop = np.zeros((len(data), numjoints, 3))
for i, img in enumerate(data):
    image_data[i] += img.dpt
    joints_2d_crop[i]+=img.gtcrop
    joints_3d_crop[i]+=img.gt3Dcrop

def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.dot(np.asarray(M).reshape((3, 3)), np.asarray([pt[0], pt[1], 1]))
    return np.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])

jointConnections = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
                                 [12, 13], [14, 15], [15, 16], [16, 17], [17, 18], [19, 20], [20, 21], [21, 22],
                                 [22, 23]]
jointConnectionColors = [matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 1]]]))[0, 0],
                              matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 1]]]))[0, 0],
                              matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 1]]]))[0, 0],
                              matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 1]]]))[0, 0],
                              matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 1]]]))[0, 0]]

# slsqp = np.load("/home/giffy/Documents/marnim-semi-auto-anno-/eval/blender_old_sort/new_joints.npz")["new2D"]
slsqp = np.load("/home/giffy/Documents/marnim-semi-auto-anno-/eval/blender_old_sort/Li_init_closest_aligned_siftflow.npy")

# slsqp = joints3DToImg(slsqp[2502], 460.017965, 459.990440, 320., 240.)
# dpt, joint = loadDepthMapBlender("/home/giffy/Documents/Datasets/Blender_dataset/Blender/hpseq_loop_mv", "c00_00002502_")
# dpt1, joint1 = loadDepthMapBlender("/home/giffy/Documents/Datasets/Blender_dataset/Blender/hpseq_loop_mv", "c00_00002500_")
joint1 = slsqp[497]
# for joint in xrange(joint1.shape[0]):
#     joint1[joint, 0:2] = transformPoint2D(joint1[joint], data[497].T).squeeze()[0:2]
joint1 = joints_2d_crop[2580] + slsqp[497,1:]
# dpt, _ = loadDepthMapBlender("/home/giffy/Documents/Datasets/Blender_dataset/Blender/hpseq_loop_mv", "c00_00002502_")
# joint1 = joints3DToImg(joint1, 460.017965, 459.990440, 320., 240.)
# print(slsqp.shape)
plt.imshow(data[497].dpt)
plt.scatter(joint1[:,0],joint1[:,1])
for i in range(len(jointConnectionColors)):
    plt.plot([joint1[jointConnections[i][0]][0],joint1[jointConnections[i][1]][0]],
              [joint1[jointConnections[i][0]][1], joint1[jointConnections[i][1]][1]],
             color = jointConnectionColors[i])
plt.title("Frame initialized with SiftFlow")
plt.axis("off")
plt.savefig("slsqp_closest.png", dpi=1200)
plt.show()
print(joints_2d_crop[2502])
print(joint1)
# images = [10, 50, 75, 100]
# jointConnections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
#                                  [10, 11], [11, 12],
#                                  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
# jointConnectionColors = [matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.4]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.6]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 0.8]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.00, 1, 1]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.4]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.6]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 0.8]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.33, 1, 1]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.4]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.6]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 0.8]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.50, 1, 1]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.4]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.6]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 0.8]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.66, 1, 1]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.4]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.6]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 0.8]]]))[0, 0],
#                               matplotlib.colors.hsv_to_rgb(np.asarray([[[0.83, 1, 1]]]))[0, 0]]
#
# # final_our_joints = np.load("/home/giffy/Documents/semiauto_author/src/eval/msra_author_crop_fixed_2020_06_11/new_joints.npz")["new2D"]
# # final_our_joints = np.load("/home/giffy/Documents/marnim-semi-auto-anno-/eval/msra_22Sep/new_joints.npz")["new2D"]
# final_our_joints = joints_2d_crop
# for i in images:
#     # for joint in xrange(final_our_joints[i].shape[0]):
#     #     finafor joint in xrange(final_our_joints[i].shape[0]):
#     #     final_our_joints[i][joint, 0:2] = transformPoint2D(final_our_joints[i][joint], data[i].T).squeeze()[0:2]
#     plt.imshow(data[i].dpt)
#     plt.scatter(final_our_joints[i][:,0],final_our_joints[i][:,1])
#     for k in range(len(jointConnectionColors)):
#         plt.plot([final_our_joints[i][jointConnections[k][0]][0],final_our_joints[i][jointConnections[k][1]][0]],
#                   [final_our_joints[i][jointConnections[k][0]][1], final_our_joints[i][jointConnections[k][1]][1]],
#                  color = jointConnectionColors[k])
#     plt.axis("off")
#     plt.savefig("images/gt_image_"+str(i)+".png", dpi = 1200)
#     plt.show()






