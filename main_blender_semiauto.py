"""This is the main file for training hand joint classifier on ICVL dataset

Created on 08.08.2014

@author: Markus Oberweger <oberweger@icg.tugraz.at>
"""

from multiprocessing import Pool
import numpy
import scipy
from sklearn.kernel_approximation import SkewedChi2Sampler, RBFSampler
import cv2
import matplotlib
matplotlib.use('Agg')  # plot to file
from semiautoanno import SemiAutoAnno
from util.handconstraints import Blender2HandConstraints
from util.handconstraints import MSRA2HandConstraints
from util.handconstraints import NYUHandConstraints
from util.handconstraints import ICVLHandConstraints
import matplotlib.pyplot as plt
import os
import cPickle
import sys
import numpy as np
from data.transformations import transformPoint2D
from data.importers import Blender2Importer
from data.importers import MSRA15Importer
from data.importers import NYUImporter
from data.importers import ICVLImporter
from data.dataset import Blender2Dataset
from data.dataset import ICVLDataset
from data.dataset import NYUDataset
from data.dataset import MSRA15Dataset
from util.handpose_evaluation import Blender2HandposeEvaluation
from util.handpose_evaluation import MSRA2HandposeEvaluation
from util.handpose_evaluation import ICVLHandposeEvaluation
from util.handpose_evaluation import NYUHandposeEvaluation

if __name__ == '__main__':
    dataset = raw_input("enter dataset name (blender, msra, nyu) : ")
    eval_prefix = raw_input("enter eval folder name : ")
    data_folder = raw_input("enter data_folder : ")
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(23455)
    di = hc = trainSeqs = train_data = train_gt3D = imgSizeW = imgSizeH = nChannels = hpe = None

    def import_blender():
        di = Blender2Importer(data_folder)
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
        train_data = numpy.concatenate(dat)
        train_gt3D = numpy.concatenate(gt)

        mb = (train_data.nbytes) / (1024 * 1024)
        print("data size: {}Mb".format(mb))

        imgSizeW = train_data.shape[3]
        imgSizeH = train_data.shape[2]
        nChannels = train_data.shape[1]

        hpe = Blender2HandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
        hpe.subfolder += '/' + eval_prefix + '/'

        hc = Blender2HandConstraints([Seq1_0.name])
        return di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe

    def import_msra():
        di = MSRA15Importer(data_folder)
        Seq1_0 = di.loadSequence('P0', shuffle=False)
        #print(Seq1_0)
        trainSeqs = [Seq1_0]

        # create training data
        trainDataSet = MSRA15Dataset(trainSeqs)
        dat = []
        gt = []
        for seq in trainSeqs:
            d, g = trainDataSet.imgStackDepthOnly(seq.name)
            dat.append(d)
            gt.append(g)
        train_data = numpy.concatenate(dat)
        train_gt3D = numpy.concatenate(gt)

        mb = (train_data.nbytes) / (1024 * 1024)
        print("data size: {}Mb".format(mb))

        imgSizeW = train_data.shape[3]
        imgSizeH = train_data.shape[2]
        nChannels = train_data.shape[1]

        hpe = MSRA2HandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
        hpe.subfolder += '/' + eval_prefix + '/'

        hc = MSRA2HandConstraints([Seq1_0.name])
        return di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe

    def import_nyu():
        di = NYUImporter(data_folder)
        Seq1_0 = di.loadSequence('train', shuffle=False)
        trainSeqs = [Seq1_0]

        # create training data
        trainDataSet = NYUDataset(trainSeqs)
        dat = []
        gt = []
        for seq in trainSeqs:
            d, g = trainDataSet.imgStackDepthOnly(seq.name)
            dat.append(d)
            gt.append(g)
        train_data = numpy.concatenate(dat)
        train_gt3D = numpy.concatenate(gt)

        mb = (train_data.nbytes) / (1024 * 1024)
        print("data size: {}Mb".format(mb))

        imgSizeW = train_data.shape[3]
        imgSizeH = train_data.shape[2]
        nChannels = train_data.shape[1]

        hpe = NYUHandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
        hpe.subfolder += '/' + eval_prefix + '/'

        hc = NYUHandConstraints([Seq1_0.name])
        return di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe

    def import_icvl():
        di = ICVLImporter(data_folder)
        Seq1_0 = di.loadSequence('hpseq_loop_mv', camera=0, shuffle=False)
        trainSeqs = [Seq1_0]

        # create training data
        trainDataSet = ICVLDataset(trainSeqs)
        dat = []
        gt = []
        for seq in trainSeqs:
            d, g = trainDataSet.imgStackDepthOnly(seq.name)
            dat.append(d)
            gt.append(g)
        train_data = numpy.concatenate(dat)
        train_gt3D = numpy.concatenate(gt)

        mb = (train_data.nbytes) / (1024 * 1024)
        print("data size: {}Mb".format(mb))

        imgSizeW = train_data.shape[3]
        imgSizeH = train_data.shape[2]
        nChannels = train_data.shape[1]

        hpe = ICVLHandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
        hpe.subfolder += '/' + eval_prefix + '/'

        hc = ICVLHandConstraints([Seq1_0.name])
        return di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe
    
    def import_alex():
        di = ALEXImporter(data_folder)
        Seq1_0 = di.loadSequence('hpseq_loop_mv', camera=0, shuffle=False)
        trainSeqs = [Seq1_0]

        # create training data
        trainDataSet = ALEXDataset(trainSeqs)
        dat = []
        gt = []
        for seq in trainSeqs:
            d, g = trainDataSet.imgStackDepthOnly(seq.name)
            dat.append(d)
            gt.append(g)
        train_data = numpy.concatenate(dat)
        train_gt3D = numpy.concatenate(gt)

        mb = (train_data.nbytes) / (1024 * 1024)
        print("data size: {}Mb".format(mb))

        imgSizeW = train_data.shape[3]
        imgSizeH = train_data.shape[2]
        nChannels = train_data.shape[1]

        hpe = ALEXHandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
        hpe.subfolder += '/' + eval_prefix + '/'

        hc = ALEXHandConstraints([Seq1_0.name])
        return di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe

    print("create data")
    if dataset == "blender":
        di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe = import_blender()
    elif dataset == "msra":
        di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe = import_msra()
    elif dataset == "nyu":
        di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe = import_nyu()
    elif dataset == "icvl":
        di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe = import_icvl()
    elif dataset == "alex":
        di, hc, trainSeqs, train_data, train_gt3D, imgSizeW, imgSizeH, nChannels, hpe = import_alex()


    # subset of all poses known, e.g. 10% labelled
    # TODO: Set this to [] in order to run reference frame selection
    subset_idxs = []
    #reference frames for NYU first 1000 images with some missing
    # nyu reference frames for first 517
    # subset_idxs = [2, 12, 21, 50, 70, 77, 84, 88, 92, 100, 112, 119, 120, 126, 136, 157, 196, 203, 212, 225, 233,
    #                240, 242, 251, 261, 265, 283, 293, 298, 299, 303, 327, 353, 357, 359, 368, 369, 372, 373, 381, 387, 389, 394, 404, 414, 417, 438, 440, 443, 446, 458]
    # TSNE reference frames for msra
    # subset_idxs = [ 282, 840, 1155, 1439, 1493, 1629, 1679, 1857, 2390, 2599, 2862, 2893, 3234, 3400, 3917, 4062, 5070, 5636, 5922, 6197, 6901, 6988, 7371, 7515, 7844, 7854, 8135]
    # subset_idxs = [  69,  215,  247,  250,  476,  670,  735,  742,  996, 1136, 1172, 1214, 1378, 1508, 1723, 1807, 1994, 2028, 2182, 2197, 2199, 2235, 2396, 2428, 2598, 2687, 2984, 3114, 3151, 3163, 3209, 3359, 3468, 3482, 3484, 3503, 3694, 3827, 3928, 4060, 4135, 4145, 4248, 4316, 4468, 4547, 4593, 4675, 4748, 4862, 4917, 5113, 5215, 5265, 5354, 5367, 5498, 5627, 5786, 5831, 5979, 6184, 6238, 6294, 6399, 6404, 6435, 6657, 6719, 6785, 7009, 7077, 7167, 7174, 7270, 7563, 7635, 7680, 7793, 7844, 7862, 7884, 8339, 8342, 8406]
    subset_idxs = [ 147, 1155, 1439, 1623, 1926, 2366, 4062, 4430, 4519, 5440, 5922, 5928, 6197, 6286, 6901, 7371, 8135]
    # Author's reference frames for msra
    # subset_idxs = [8, 10, 13, 15, 18, 28, 30, 47, 53, 62, 64, 65, 72, 75, 80, 96, 97, 107, 110, 114, 121, 131, 161, 163, 167, 170,
    #   172, 175, 183, 193, 196, 201, 208, 213, 230, 233, 237, 238, 241, 243, 245, 248, 253, 254, 256, 265, 271, 284, 292,
    #   302, 304, 311, 320, 331, 350, 353, 357, 360, 362, 376, 381, 384, 386, 390, 391, 399, 401, 408, 413, 416, 417, 425,
    #   430, 437, 440, 448, 453, 457, 471, 482, 487, 493, 497, 502, 506, 510, 516, 517, 524, 529, 540, 552, 561, 565, 580,
    #   590, 595, 599, 602, 605, 613, 641, 642, 644, 663, 676, 680, 687, 690, 696, 720, 726, 735, 745, 750, 765, 766, 774,
    #   790, 801, 827, 858, 876, 883, 892, 900, 903, 907, 912, 924, 933, 941, 943, 952, 959, 961, 970, 973, 978, 985, 992,
    #   1000, 1007, 1013, 1019, 1024, 1041, 1049, 1053, 1058, 1062, 1066, 1080, 1082, 1083, 1100, 1116, 1123, 1133, 1139,
    #   1147, 1150, 1152, 1161, 1162, 1164, 1177, 1183, 1185, 1189, 1203, 1214, 1226, 1233, 1247, 1260, 1279, 1329, 1332,
    #   1336, 1337, 1340, 1341, 1346, 1426, 1475, 1514, 1531, 1539, 1543, 1551, 1559, 1566, 1576, 1582, 1590, 1597, 1603,
    #   1610, 1613, 1617, 1634, 1640, 1656, 1735, 1739, 1743, 1750, 1754, 1763, 1771, 1776, 1785, 1786, 1798, 1805, 1806,
    #   1815, 1817, 1829, 1875, 1890, 1892, 1896, 1898, 1906, 1917, 1919, 1954, 1956, 1958, 1960, 1966, 1973, 1975, 1977,
    #   1987, 1996, 2003, 2004, 2013, 2025, 2030, 2033, 2036, 2040, 2046, 2052, 2059, 2077, 2081, 2104, 2107, 2123, 2131,
    #   2162, 2181, 2194, 2214, 2219, 2223, 2230, 2231, 2246, 2258, 2328, 2341, 2344, 2351, 2363, 2371, 2372, 2391, 2393,
    #   2396, 2405, 2419, 2424, 2426, 2429, 2438, 2439, 2441, 2446, 2447, 2454, 2458, 2474, 2500, 2517, 2519, 2523, 2525,
    #   2533, 2536, 2542, 2546, 2549, 2554, 2581, 2588, 2592, 2604, 2638, 2674, 2687, 2689, 2693, 2712, 2743, 2761, 2767,
    #   2772, 2778, 2782, 2784, 2824, 2828, 2866, 2886, 2909, 2919, 2928, 2937, 2940, 2941, 2946, 2958, 2968, 2982, 2998,
    #   3002, 3007, 3019, 3021, 3026, 3029, 3034, 3037, 3043, 3047, 3054, 3057, 3062, 3063, 3073, 3085, 3109, 3133, 3143,
    #   3170, 3193, 3273, 3277, 3289, 3303, 3306, 3315, 3320, 3333, 3350, 3358, 3360, 3372, 3396, 3402, 3409, 3420, 3458,
    #   3478, 3503, 3505, 3530, 3531, 3537, 3570, 3574, 3577, 3585, 3601, 3608, 3612, 3615, 3619, 3624, 3625, 3642, 3646,
    #   3653, 3708, 3715, 3720, 3734, 3736, 3755, 3773, 3799, 3808, 3814, 3867, 3915, 3942, 3963, 3984, 3994, 3999, 4001,
    #   4002, 4005, 4031, 4039, 4043, 4049, 4052, 4078, 4091, 4098, 4099, 4127, 4146, 4147, 4153, 4155, 4156, 4158, 4161,
    #   4166, 4168, 4176, 4184, 4192, 4198, 4208, 4209, 4222, 4225, 4228, 4231, 4232, 4247, 4252, 4260, 4286, 4302, 4309,
    #   4311, 4313, 4315, 4319, 4327, 4330, 4332, 4339, 4348, 4361, 4368, 4374, 4403, 4407, 4412, 4450, 4459, 4462, 4467,
    #   4473, 4474, 4493, 4495, 4497, 4501, 4505, 4523, 4526, 4527, 4528, 4531, 4537, 4549, 4551, 4557, 4576, 4587, 4597,
    #   4600, 4607, 4616, 4619, 4627, 4628, 4635, 4637, 4643, 4648, 4677, 4709, 4714, 4720, 4743, 4758, 4771, 4786, 4799,
    #   4805, 4811, 4815, 4819, 4822, 4828, 4836, 4841, 4856, 4864, 4868, 4872, 4888, 4890, 4892, 4895, 4915, 4939, 4943,
    #   4944, 4964, 4972, 4979, 4988, 4996, 5004, 5006, 5016, 5017, 5025, 5032, 5039, 5057, 5062, 5076, 5083, 5085, 5091,
    #   5099, 5126, 5154, 5159, 5243, 5266, 5268, 5284, 5285, 5287, 5309, 5333, 5351, 5355, 5356, 5405, 5408, 5412, 5413,
    #   5418, 5436, 5457, 5465, 5471, 5476, 5493, 5501, 5506, 5509, 5511, 5517, 5520, 5543, 5559, 5572, 5577, 5579, 5582,
    #   5585, 5588, 5591, 5596, 5607, 5618, 5623, 5633, 5641, 5646, 5649, 5655, 5659, 5667, 5668, 5670, 5672, 5675, 5683,
    #   5689, 5700, 5708, 5718, 5721, 5725, 5726, 5739, 5743, 5760, 5770, 5792, 5797, 5811, 5816, 5839, 5849, 5852, 5872,
    #   5874, 5885, 5895, 5900, 5903, 5909, 5915, 5927, 5938, 5958, 5973, 5979, 5984, 5993, 6004, 6006, 6022, 6030, 6039,
    #   6040, 6054, 6058, 6074, 6082, 6086, 6106, 6113, 6117, 6155, 6178, 6189, 6206, 6214, 6242, 6247, 6255, 6271, 6321,
    #   6337, 6339, 6353, 6383, 6387, 6398, 6414, 6468, 6501, 6545, 6578, 6596, 6598, 6604, 6630, 6638, 6652, 6658, 6672,
    #   6677, 6680, 6709, 6739, 6787, 6799, 6856, 6857, 6882, 6910, 6943, 6951, 6953, 6961, 6977, 6980, 7000, 7007, 7012,
    #   7015, 7019, 7025, 7028, 7038, 7040, 7048, 7051, 7061, 7067, 7075, 7077, 7082, 7087, 7105, 7108, 7112, 7115, 7118,
    #   7122, 7123, 7126, 7130, 7134, 7137, 7138, 7142, 7147, 7151, 7155, 7163, 7165, 7172, 7178, 7186, 7193, 7198, 7202,
    #   7206, 7209, 7218, 7223, 7234, 7247, 7251, 7252, 7256, 7285, 7291, 7296, 7309, 7317, 7329, 7334, 7346, 7356, 7369,
    #   7389, 7406, 7416, 7434, 7458, 7462, 7477, 7503, 7509, 7514, 7553, 7596, 7603, 7615, 7617, 7622, 7645, 7648, 7650,
    #   7653, 7655, 7666, 7677, 7689, 7695, 7704, 7724, 7743, 7765, 7773, 7778, 7788, 7798, 7810, 7812, 7838, 7868, 7899,
    #   7918, 7941, 7943, 7949, 7962, 7973, 7983, 7994, 7997, 7999, 8002, 8004, 8014, 8021, 8031, 8043, 8045, 8050, 8056,
    #   8071, 8082, 8093, 8096, 8102, 8129, 8132, 8135, 8148, 8154, 8170, 8172, 8176, 8185, 8194, 8195, 8215, 8226, 8228,
    #   8236, 8254, 8258, 8264, 8288, 8293, 8298, 8302, 8306, 8312, 8336, 8350, 8394, 8403, 8405, 8409, 8417, 8419, 8421,
    #   8425, 8470, 8481, 8486, 8495]
    # blender ref framesgiffy
    # subset_idxs = [206, 259, 729, 919, 991, 1032, 1047, 1185, 1241, 1431, 1647, 1692, 1787, 1830,
    #                1849, 1888, 1967, 2008, 2091, 2251, 2351, 2545, 2588, 2801, 2824, 2893, 3001]
    # Refer to Example 20 as a better example
    # Our blender ref
    # subset_idxs = [16, 21, 26, 29, 45, 49, 52, 54, 58, 104, 108, 114, 138, 144, 148, 170, 175, 178, 210, 214, 217, 231, 237, 249, 252, 259, 264, 283, 287, 296, 307, 345, 370, 381, 384, 386, 405, 412, 423, 429, 436, 458, 465, 469, 490, 498, 505, 526, 530, 533, 537, 546, 553, 576, 607, 612, 624, 631, 657, 667, 669, 673, 685, 697, 704, 735, 742, 751, 765, 781, 784, 789, 793, 801, 805, 816, 820, 827, 830, 874, 886, 888, 893, 896, 899, 911, 923, 934, 962, 969, 983, 1023, 1027, 1029, 1034, 1046, 1054, 1057, 1070, 1075, 1085, 1093, 1098, 1110, 1114, 1134, 1138, 1146, 1173, 1181, 1184, 1188, 1191, 1194, 1208, 1213, 1221, 1224, 1228, 1241, 1248, 1251, 1255, 1262, 1267, 1274, 1286, 1295, 1308, 1312, 1335, 1341, 1349, 1353, 1383, 1386, 1389, 1410, 1414, 1422, 1432, 1449, 1452, 1455, 1465, 1473, 1477, 1489, 1504, 1523, 1532, 1542, 1550, 1552, 1571, 1580, 1586, 1591, 1609, 1613, 1617, 1628, 1632, 1644, 1653, 1656, 1688, 1694, 1695, 1698, 1709, 1713, 1725, 1745, 1752, 1756, 1762, 1772, 1778, 1795, 1812, 1814, 1817, 1830, 1833, 1848, 1853, 1858, 1864, 1869, 1873, 1887, 1892, 1897, 1904, 1927, 1930, 1934, 1937, 1965, 1973, 1978, 1991, 2017, 2028, 2033, 2048, 2055, 2058, 2067, 2074, 2094, 2131, 2137, 2146, 2150, 2166, 2170, 2177, 2185, 2191, 2196, 2203, 2208, 2213, 2222, 2255, 2269, 2273, 2288, 2291, 2298, 2305, 2325, 2331, 2334, 2339, 2343, 2347, 2351, 2372, 2380, 2390, 2394, 2416, 2428, 2434, 2462, 2468, 2484, 2497, 2504, 2509, 2511, 2515, 2529, 2543, 2566, 2572, 2584, 2590, 2609, 2617, 2627, 2631, 2644, 2651, 2654, 2661, 2685, 2687, 2693, 2702, 2737, 2749, 2754, 2763, 2775, 2778, 2790, 2792, 2808, 2813, 2816, 2820, 2829, 2835, 2852, 2856, 2872, 2891, 2898, 2905, 2911, 2942, 2945, 2949, 2952, 2989, 3011, 3015, 3031, 3034, 3037]
    #Author reference frames
    # subset_idxs = [16, 21, 26, 29, 45, 49, 52, 54, 58, 104, 108, 114, 138, 144, 148, 170, 175, 178, 210, 214, 217, 231,
    #                237, 249, 252, 259, 264, 283, 287, 296, 307, 345, 370, 381, 384, 386, 405, 412, 423, 429, 436, 458,
    #                465, 469, 490, 498, 505, 526, 530, 533, 537, 546, 553, 576, 607, 612, 624, 631, 657, 667, 669, 673,
    #                685, 697, 704, 735, 742, 751, 765, 781, 784, 789, 793, 801, 805, 816, 820, 827, 830, 874, 886, 888,
    #                893, 896, 899, 911, 923, 934, 962, 969, 983, 1023, 1027, 1029, 1034, 1046, 1054, 1057, 1070, 1075,
    #                1085, 1093, 1098, 1110, 1114, 1134, 1138, 1146, 1173, 1181, 1184, 1188, 1191, 1194, 1208, 1213, 1221,
    #                1224, 1228, 1241, 1248, 1251, 1255, 1262, 1267, 1274, 1286, 1295, 1308, 1312, 1335, 1341, 1349, 1353,
    #                1383, 1386, 1389, 1410, 1414, 1422, 1432, 1449, 1452, 1455, 1465, 1473, 1477, 1489, 1504, 1523, 1532,
    #                1542, 1550, 1552, 1571, 1580, 1586, 1591, 1609, 1613, 1617, 1628, 1632, 1644, 1653, 1656, 1688, 1694,
    #                1695, 1698, 1709, 1713, 1725, 1745, 1752, 1756, 1762, 1772, 1778, 1795, 1812, 1814, 1817, 1830, 1833,
    #                1848, 1853, 1858, 1864, 1869, 1873, 1887, 1892, 1897, 1904, 1927, 1930, 1934, 1937, 1965, 1973, 1978,
    #                1991, 2017, 2028, 2033, 2048, 2055, 2058, 2067, 2074, 2094, 2131, 2137, 2146, 2150, 2166, 2170, 2177,
    #                2185, 2191, 2196, 2203, 2208, 2213, 2222, 2255, 2269, 2273, 2288, 2291, 2298, 2305, 2325, 2331, 2334,
    #                2339, 2343, 2347, 2351, 2372, 2380, 2390, 2394, 2416, 2428, 2434, 2462, 2468, 2484, 2497, 2504, 2509,
    #                2511, 2515, 2529, 2543, 2566, 2572, 2584, 2590, 2609, 2617, 2627, 2631, 2644, 2651, 2654, 2661, 2685,
    #                2687, 2693, 2702, 2737, 2749, 2754, 2763, 2775, 2778, 2790, 2792, 2808, 2813, 2816, 2820, 2829, 2835,
    #                2852, 2856, 2872, 2891, 2898, 2905, 2911, 2942, 2945, 2949, 2952, 2989, 3011, 3015, 3031, 3034, 3037]

    def getSeqIdxForFlatIdx(i):
        nums = numpy.insert(numpy.cumsum(numpy.asarray([len(s.data) for s in trainSeqs])), 0, 0)
        d1 = nums - i
        d1[d1 > 0] = -max([len(s.data) for s in trainSeqs])
        seqidx = numpy.argmax(d1)
        idx = i - nums[seqidx]
        return seqidx, idx

    # mark reference frames
    print subset_idxs
    for i in subset_idxs:
        seqidx, idx = getSeqIdxForFlatIdx(i)
        #print(seqidx, idx)
        #print trainSeqs[seqidx]
        #print trainSeqs[seqidx].data[idx]
        trainSeqs[seqidx].data[idx] = trainSeqs[seqidx].data[idx]._replace(subSeqName="ref")

    eval_params = {'init_method': 'closest',
                   'init_manualrefinement': True,  # True, False
                   'init_offset': 'siftflow',
                   'init_fallback': False,  # True, False
                   'init_incrementalref': True,  # True, False
                   'init_refwithinsequence': False,  # True, False
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

    #############################################################################
    cae_path = ""
    depth_names = [ds.fileName for s in trainSeqs for ds in s.data]
    li = numpy.asarray([(ds.gtcrop[:, 0:2] - (train_data.shape[3]/2.)) / (train_data.shape[3]/2.) for s in trainSeqs for ds in s.data], dtype='float32')
    li = li[subset_idxs].clip(-1., 1.)
    train_off3D = numpy.asarray([ds.com for s in trainSeqs for ds in s.data], dtype='float32')
    train_trans2D = numpy.asarray([numpy.asarray(ds.T).transpose() for s in trainSeqs for ds in s.data], dtype='float32')
    train_scale = numpy.asarray([s.config['cube'][2]/2. for s in trainSeqs], dtype='float32').repeat([len(s.data) for s in trainSeqs])
    hc_pm = hc.hc_projectionMat()  # create 72 by #Constraints matrix that specifies constant joint length
    boneLength = numpy.asarray(hc.boneLength, dtype='float32').reshape((len(trainSeqs), len(hc.hc_pairs)))
    boneLength /= numpy.asarray([s.config['cube'][2]/2. for s in trainSeqs])[:, None]
    lu_pm = hc.lu_projectionMat()  # create 72 by #Constraints matrix that specifies bounds on variable joint length
    boneRange = numpy.asarray(hc.boneRanges, dtype='float32').reshape((len(trainSeqs), len(hc.lu_pairs), 2))
    boneRange /= numpy.asarray([s.config['cube'][2]/2. for s in trainSeqs])[:, None, None]
    zz_thresh = hc.zz_thresh
    zz_pairs = hc.zz_pairs
    zz_pairs_v1M, zz_pairs_v2M = hc.zz_projectionMat()

    nums = numpy.insert(numpy.cumsum(numpy.asarray([len(s.data) for s in trainSeqs])), 0, 0)
    li_visiblemask = numpy.ones((len(subset_idxs), trainSeqs[0].data[0].gtorig.shape[0]))
    for i, sidx in enumerate(subset_idxs):
        seqidx, idx = getSeqIdxForFlatIdx(sidx)
        vis = di.visibilityTest(di.loadDepthMap(trainSeqs[seqidx].data[idx].fileName),
                                trainSeqs[seqidx].data[idx].gtorig, 10.)
        # remove not visible ones
        occluded = numpy.setdiff1d(numpy.arange(li_visiblemask.shape[1]), vis)
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

    # li_posebits = None
    # pb_pairs = []

    #############################################################
    # simple run
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
    msr = SemiAutoAnno('./eval/' + eval_prefix, eval_params, train_data, train_off3D, train_trans2D, train_scale,
                       boneLength, boneRange, li, subset_idxs, di.getCameraProjection(), cae_path, hc_pm,
                       hc.hc_pairs, lu_pm, hc.lu_pairs, pb_pairs, hc.posebits, hc.pb_thresh,
                       hc.getTemporalBreaks(), hc.getHCBreaks(), hc.getSequenceBreaks(),
                       zz_pairs, zz_thresh, zz_pairs_v1M, zz_pairs_v2M, hc.finger_tips,
                       lambdaW, lambdaM, lambdaP, lambdaR, lambdaMu, lambdaTh, muLagr,
                       ref_lambdaP, ref_muLagr, init_lambdaP, init_muLagr, di, hc, depth_names,
                       li_visiblemask=li_visiblemask, li_posebits=li_posebits, normalizeRi=None,
                       useCache=True, normZeroOne=False, gt3D=train_gt3D, hpe=hpe, debugPrint=False)

    # Test initialization
    jts = msr.li3D_aug
    gt3D = [j.gt3Dorig for s in trainSeqs for j in s.data]
    joints = []
    for i in range(jts.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(subset_idxs[i])
        joints.append(jts[i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    if dataset == "blender":
        hpe = Blender2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "msra":
        hpe = MSRA2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "nyu":
        hpe = NYUHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "icvl":
        hpe = ICVLHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "alex":
        hpe = ALEXHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    hpe.subfolder += '/'+eval_prefix+'/'
    print("Initialization:")
    print("Subset samples: {}".format(len(subset_idxs)))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe.getMeanError(), hpe.getMaxError(), hpe.getMedianError()))
    print("{}".format([hpe.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    if dataset == "blender":
        hpe_vis = Blender2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "msra":
        hpe_vis = MSRA2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "nyu":
        hpe_vis = NYUHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "icvl":
        hpe_vis = ICVLHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    elif dataset == "alex":
        hpe_vis = ALEXHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)

    hpe_vis.subfolder += '/'+eval_prefix+'/'
    hpe_vis.maskVisibility(numpy.repeat(li_visiblemask[:, :, None], 3, axis=2))
    print("Only visible joints:")
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_vis.getMeanError(), hpe_vis.getMaxError(), hpe_vis.getMedianError()))
    print("{}".format([hpe_vis.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_vis.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    # Tracking
    theta = msr.fitTracking(num_iter=None, useLagrange=False)
    numpy.savez('./eval/'+eval_prefix+'/params_tracking.npz', *theta)

    print("Testing ...")
    
    # evaluate optimization result
    gt3D = [j.gt3Dorig for s in trainSeqs for j in s.data]
    joints = []
    for i in range(train_data.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        joints.append(theta[1][i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    if dataset == "blender":
        hpe_ref = Blender2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], numpy.asarray(joints)[subset_idxs])
    elif dataset == "msra":
        hpe_ref = MSRA2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], numpy.asarray(joints)[subset_idxs])
    elif dataset == "nyu":
        hpe_ref = NYUHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], numpy.asarray(joints)[subset_idxs])
    elif dataset == "icvl":
        hpe_ref = ICVLHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], numpy.asarray(joints)[subset_idxs])
    elif dataset == "alex":
        hpe_ref = ALEXHandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], numpy.asarray(joints)[subset_idxs])
    hpe_ref.subfolder += '/'+eval_prefix+'/'
    print("Reference samples: {}".format(len(subset_idxs)))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_ref.getMeanError(), hpe_ref.getMaxError(), hpe_ref.getMedianError()))
    print("{}".format([hpe_ref.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_ref.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    joints = []
    for i in range(train_data.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        joints.append(theta[2][i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    if dataset == "blender":
        hpe_init = Blender2HandposeEvaluation(gt3D, joints)
    elif dataset == "msra":
        hpe_init = MSRA2HandposeEvaluation(gt3D, joints)
    elif dataset == "nyu":
        hpe_init = NYUHandposeEvaluation(gt3D, joints)
    elif dataset == "icvl":
        hpe_init = ICVLHandposeEvaluation(gt3D, joints)
    elif dataset == "alex":
        hpe_init = ALEXHandposeEvaluation(gt3D, joints)
    hpe_init.subfolder += '/'+eval_prefix+'/'
    print("Train initialization: {}".format(train_data.shape[0]))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_init.getMeanError(), hpe_init.getMaxError(), hpe_init.getMedianError()))
    print("{}".format([hpe_init.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_init.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    joints = []
    for i in range(train_data.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        joints.append(theta[1][i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)
    if dataset == "blender":
        hpe_full = Blender2HandposeEvaluation(gt3D, joints)
    elif dataset == "msra":
        hpe_full = MSRA2HandposeEvaluation(gt3D, joints)
    elif dataset == "nyu":
        hpe_full = NYUHandposeEvaluation(gt3D, joints)
    elif dataset == "icvl":
        hpe_full = ICVLHandposeEvaluation(gt3D, joints)
    elif dataset == "alex":
        hpe_full = ALEXHandposeEvaluation(gt3D, joints)

    hpe_full.subfolder += '/'+eval_prefix+'/'
    print("Train samples: {}".format(train_data.shape[0]))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_full.getMeanError(), hpe_full.getMaxError(), hpe_full.getMedianError()))
    print("{}".format([hpe_full.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_full.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    # save final joint annotations, in original scaling
    joints2D = numpy.zeros_like(joints)
    for i in xrange(len(joints)):
        joints2D[i] = di.joints3DToImg(joints[i])
    numpy.savez('./eval/'+eval_prefix+'/new_joints.npz', new2D=numpy.asarray(joints2D), new3D=numpy.asarray(joints))

    ind=0
    for i in trainSeqs[0].data:
        # print len(joints)
        original_joints = i.gtcrop
        jt = joints[ind]
        jtI = di.joints3DToImg(jt)
        for joint in range(jt.shape[0]):
            t=transformPoint2D(jtI[joint],i.T)
            # print "jtI[joint]", jtI[joint]
            jtI[joint,0] = t[0]
            jtI[joint,1] = t[1]

        # if ind <= 5:
        #     print '##################'
        #     print 'jtI', jtI
        #     print 'original_joints', original_joints
        #     #print jtI[:, 0], original_joints[:, 0]
        #     #sq = 0
        #     #for k in range(jt.shape[0]):
        #     #    print sq
        #     print jtI[:, 0] - original_joints[:, 0]
        #     print jtI[:, 1] - original_joints[:, 1]
        #     print jtI[:, 2] - original_joints[:, 2]
        #     sq = np.sqrt((jtI[:,0] - original_joints[:,0]) ** 2 +(jtI[:,1] - original_joints[:,1]) ** 2 )
        #     sq3 = np.sqrt((jtI[:, 0] - original_joints[:, 0]) ** 2 + (jtI[:, 1] - original_joints[:, 1]) ** 2 + (jtI[:, 2] - original_joints[:, 2]) ** 2)
        #     print 'sq', sq
        #     print sum(sq)
        #     print 'sq3', sq3
        #     print sum(sq3)
        hpe_full.plotResult(i.dpt, i.gtcrop, jtI,"{}_optimized_{}".format(eval_prefix, ind))
        if ind < 100:
            ind += 1
        else:
            break

    hpe_init.saveVideo('P0_init', trainSeqs[0], di, fullFrame=True, plotFrameNumbers=True)
    hpe_full.saveVideo('P0', trainSeqs[0], di, fullFrame=True, plotFrameNumbers=True)
    hpe_init.saveVideo3D('P0_init', trainSeqs[0])
    hpe_full.saveVideo3D('P0', trainSeqs[0])

