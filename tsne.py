
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import Blender2Dataset
from data.importers import Blender2Importer
from semiautoanno import SemiAutoAnno
from util.handconstraints import Blender2HandConstraints
from util.handpose_evaluation import Blender2HandposeEvaluation
from util.sort import find_reference_frames

data_folder= "/home/giffy/semi-auto-anno/data/Blender"
def import_blender():
    di = Blender2Importer(data_folder, useCache=True)
    Seq1_0 = di.loadSequence('hpseq_loop_mv', camera=0, shuffle=False)
    trainSeqs = [Seq1_0]

    # create training data
    trainDataSet = Blender2Dataset(trainSeqs)
    dat = []
    gt = []
    for seq in trainSeqs:
        d, g = trainDataSet.imgStackDepthOnly(seq.name)
        dat.append(d)
        gt.append(g)
    train_data = np.concatenate(dat)
    train_gt3D = np.concatenate(gt)

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    imgSizeW = train_data.shape[3]
    imgSizeH = train_data.shape[2]
    nChannels = train_data.shape[1]

    hpe = Blender2HandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
    hpe.subfolder += '/' + eval_prefix + '/'

    hc = Blender2HandConstraints([Seq1_0.name])
    return di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe




lambdaW = 1e-5
lambdaM = 1e0
lambdaR = 1e0
lambdaP = 1e2
lambdaMu = 1e-2
lambdaTh = 1e0
muLagr = 1e2
ref_lambdaP = 1e1
ref_muLagr = 1e1
init_lambdaP = 1e0
init_muLagr = 1e1
eval_prefix = "blender_server"

reference_frame_calculator = find_reference_frames(eval_prefix, force=True, distance_threshold=0.12)
subset_idxs = reference_frame_calculator.calculate_reference_frames()
del reference_frame_calculator

di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe = import_blender()
cae_path = ""
depth_names = [ds.fileName for s in trainSeqs for ds in s.data]
li = np.asarray([(ds.gtcrop[:, 0:2] - (train_data.shape[3]/2.)) / (train_data.shape[3]/2.) for s in trainSeqs for ds in s.data], dtype='float32')
li = li[subset_idxs].clip(-1., 1.)
train_off3D = np.asarray([ds.com for s in trainSeqs for ds in s.data], dtype='float32')
train_trans2D = np.asarray([np.asarray(ds.T).transpose() for s in trainSeqs for ds in s.data], dtype='float32')
train_scale = np.asarray([s.config['cube'][2]/2. for s in trainSeqs], dtype='float32').repeat([len(s.data) for s in trainSeqs])
hc_pm = hc.hc_projectionMat()  # create 72 by #Constraints matrix that specifies constant joint length
boneLength = np.asarray(hc.boneLength, dtype='float32').reshape((len(trainSeqs), len(hc.hc_pairs)))
boneLength /= np.asarray([s.config['cube'][2]/2. for s in trainSeqs])[:, None]
lu_pm = hc.lu_projectionMat()  # create 72 by #Constraints matrix that specifies bounds on variable joint length
boneRange = np.asarray(hc.boneRanges, dtype='float32').reshape((len(trainSeqs), len(hc.lu_pairs), 2))
boneRange /= np.asarray([s.config['cube'][2]/2. for s in trainSeqs])[:, None, None]
zz_thresh = hc.zz_thresh
zz_pairs = hc.zz_pairs
zz_pairs_v1M, zz_pairs_v2M = hc.zz_projectionMat()

eval_params = {'init_method': 'closest',
                   'init_manualrefinement': True,  # True, False
                   'init_offset': 'siftflow',
                   'init_fallback': False,  # True, False
                   'init_incrementalref': False,  # True, False
                   'init_refwithinsequence': True,  # True, False
                   'init_optimize_incHC': True,  # True, False
                   'init_accuracy_tresh': 10.,  # px
                   'ref_descriptor': 'hog',
                   'ref_cluster': 'sm_greedy',
                   'ref_fraction': 10.,  # % of samples used as reference at max
                   'ref_threshold': 0.08,  # % of samples used as reference at max
                   'ref_optimization': 'SLSQP',
                   'ref_optimize_incHC': True,  # True, False
                   'joint_eps': 15.,  # mm, visible joints must stay within +/- eps to initialization
                   'joint_off': hc.joint_off,  # all joints must be smaller than depth from depth map
                   'eval_initonly': False,  # True, False
                   'eval_refonly': False,  # True, False
                   'optimize_bonelength': False,  # True, False
                   'optimize_Ri': False,  # True, False
                   'global_optimize_incHC': True,  # True, False
                   'global_corr_ref': 'closest',
                   'global_tempconstraints': 'local',  # local, global, none
                   'corr_patch_size': 24,  # px
                   'corr_method': cv2.TM_CCORR_NORMED  # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
                   }

def getSeqIdxForFlatIdx(i):
    nums = np.insert(np.cumsum(np.asarray([len(s.data) for s in trainSeqs])), 0, 0)
    d1 = nums - i
    d1[d1 > 0] = -max([len(s.data) for s in trainSeqs])
    seqidx = np.argmax(d1)
    idx = i - nums[seqidx]
    return seqidx, idx

nums = np.insert(np.cumsum(np.asarray([len(s.data) for s in trainSeqs])), 0, 0)
li_visiblemask = np.ones((len(subset_idxs), trainSeqs[0].data[0].gtorig.shape[0]))
for i, sidx in enumerate(subset_idxs):
    seqidx, idx = getSeqIdxForFlatIdx(sidx)
    vis = di.visibilityTest(di.loadDepthMap(trainSeqs[seqidx].data[idx].fileName),
                            trainSeqs[seqidx].data[idx].gtorig, 10.)
    # remove not visible ones
    occluded = np.setdiff1d(np.arange(li_visiblemask.shape[1]), vis)
    li_visiblemask[i, occluded] = 0

# li_visiblemask = None

li_posebits = []
pb_pairs = []
for i in range(len(subset_idxs)):
    seqidx, idx = getSeqIdxForFlatIdx(i)
    lip = []
    pip = []
    for p in range(len(hc.posebits)):
        if abs(train_gt3D[subset_idxs[i], hc.posebits[p][0], 2] - train_gt3D[subset_idxs[i], hc.posebits[p][1], 2]) > hc.pb_thresh/(trainSeqs[seqidx].config['cube'][2]/2.):
            if train_gt3D[subset_idxs[i], hc.posebits[p][0], 2] < train_gt3D[subset_idxs[i], hc.posebits[p][1], 2]:
                lip.append((hc.posebits[p][0], hc.posebits[p][1]))
                pip.append(hc.posebits[p])
            else:
                lip.append((hc.posebits[p][1], hc.posebits[p][0]))
                pip.append(hc.posebits[p])
    li_posebits.append(lip)
    pb_pairs.append(pip)

msr = SemiAutoAnno('./eval/' + eval_prefix, eval_params, train_data, train_off3D, train_trans2D, train_scale,
                       boneLength, boneRange, li, subset_idxs, di.getCameraProjection(), cae_path, hc_pm,
                       hc.hc_pairs, lu_pm, hc.lu_pairs, pb_pairs, hc.posebits, hc.pb_thresh,
                       hc.getTemporalBreaks(), hc.getHCBreaks(), hc.getSequenceBreaks(),
                       zz_pairs, zz_thresh, zz_pairs_v1M, zz_pairs_v2M, hc.finger_tips,
                       lambdaW, lambdaM, lambdaP, lambdaR, lambdaMu, lambdaTh, muLagr,
                       ref_lambdaP, ref_muLagr, init_lambdaP, init_muLagr, di, hc, depth_names,
                       li_visiblemask=li_visiblemask, li_posebits=li_posebits, normalizeRi=None,
                       useCache=True, normZeroOne=False, gt3D=train_gt3D, hpe=hpe, debugPrint=False)


# pickleCache = 'cache/Blender2Importer_hpseq_loop_mv_0_cache.pkl'
sorted_frames = np.load("eval/blender_server/sort_array.npy")
# ref_frames = np.load("eval/blender_server/ref_array.npy")
# f = open(pickleCache, 'rb')
# (seqName, data, config) = pickle.load(f)
# f.close()
old_img = train_data[0,0].dpt
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
new_joints = np.zeros((len(train_data.shape[0]), 24, 1, 2))

for i in sorted_frames:
    if i in subset_idxs:
        new_joints[i] += li[i].gtcrop.reshape(24, 1, 2)
        joint_3d = msr.project2Dto3D(li, i)
        old_img = train_data[i,0].dpt
        p0 = new_joints[i].astype('float32')
        continue
    new_frame = train_data[i,0].astype('uint8')
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_img.astype('uint8'), new_frame, p0, None, **lk_params)
    new_joints[i,:,:,:2] += p1
    for joint in range(24):
        if 127 < p1[joint, 0, 0].astype('int32'):
            p1[joint, 0, 0] = 127
        if 127 < p1[joint, 0, 1].astype('int32'):
            p1[joint, 0, 1] = 127
        if 0 > p1[joint, 0, 0].astype('int32'):
            p1[joint, 0, 0] = 0
        if 0 > p1[joint, 0, 1].astype('int32'):
            p1[joint, 0, 1] = 0
        new_joints[i,joint,0,2] += train_data[i,0].dpt[int(p1[joint,0,0]), int(p1[joint,0,1])]
    joints_3D
    msr.optimizeReferenceFramesLi_SLSQP()
    p0 = p1.copy()
    old_img = new_frame.copy()

print(new_joints)

frame = sorted_frames[2000]
plt.imshow(data[frame].dpt)
for i in range(24):
    plt.scatter(new_joints[frame,i,0,0], new_joints[frame,i,0,1])
plt.show()
# for j, frame in enumerate(sorted_frames):
#     if j > 250:
#         break
#     plt.imshow(data[frame].dpt)
#     for i in range(24):
#         plt.scatter(new_joints[frame, i, 0, 0], new_joints[frame, i, 0, 1])
#     plt.show()





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
