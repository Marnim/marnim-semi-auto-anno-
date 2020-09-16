"""Provides a class that retrieves hand constraints for different
datasets and annotations.

HandConstraints provides interface for retrieving hand constraints.

Copyright 2016 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of SemiAutoAnno.

SemiAutoAnno is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SemiAutoAnno is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SemiAutoAnno.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HandConstraints(object):
    """
    Class for modelling hand constraints
    """

    def __init__(self, num_joints):
        """
        Constructor
        """
        self.num_joints = num_joints

        # Posebit threshold
        self.pb_thresh = 10.  # mm

        # joint offset, must be this smaller than depth
        self.joint_off = []

    def jointLength(self, j0, j1):
        return numpy.sqrt(numpy.square(j0 - j1).sum())

    def jointLengths(self, joints):
        dists = []
        for p in self.hc_pairs:
            dists.append(self.jointLength(joints[p[0]], joints[p[1]]))

        return dists

    def jointRanges(self, joints):
        assert len(joints.shape) == 3  # we need array (samples, joints, 3)
        dists = numpy.zeros((joints.shape[0], len(self.lu_pairs)))
        for i in xrange(joints.shape[0]):
            for ip, p in enumerate(self.lu_pairs):
                length = self.jointLength(joints[i, p[0]], joints[i, p[1]])
                dists[i, ip] = length

        maxd = dists.max(axis=0)
        mind = dists.min(axis=0)
        ret = []
        for ip, p in enumerate(self.lu_pairs):
            ret.append((max(mind[ip]*0.8, 0.), maxd[ip]*1.2))

        return ret

    def hc_projectionMat(self):
        """
        Generate a matrix that encodes the constraints
        [[1, 0, 0, ...]  x-coordinate, first joint
         [0, 1, 0, ...]  y-coordinate
         [0, 0, 1, ...]  z-coordinate
         [0, 0, 0, ...]
         ...
         [0, 0, 0, ...]
         [-1, 0, 0, ...]  x-coordinate, second joint
         [0, -1, 0, ...]  y-coordinate
         [0, 0, -1, ...]] z-coordinate
        :return: matrix 3*numJoints x 3*numConstraints
        """
        M = numpy.zeros((3 * self.num_joints, 3 * len(self.hc_pairs)), dtype='float32')

        ip = 0
        for p in self.hc_pairs:
            # x coordinate
            M[3 * p[0] + 0, ip] = 1
            M[3 * p[1] + 0, ip] = -1
            # y coordinate
            ip += 1
            M[3 * p[0] + 1, ip] = 1
            M[3 * p[1] + 1, ip] = -1
            # z coordinate
            ip += 1
            M[3 * p[0] + 2, ip] = 1
            M[3 * p[1] + 2, ip] = -1
            ip += 1

        return M

    def lu_projectionMat(self):
        """
        Generate a matrix that encodes the constraints
        [[1, 0, 0, ...]  x-coordinate, first joint
         [0, 1, 0, ...]  y-coordinate
         [0, 0, 1, ...]  z-coordinate
         [0, 0, 0, ...]
         ...
         [0, 0, 0, ...]
         [-1, 0, 0, ...]  x-coordinate, second joint
         [0, -1, 0, ...]  y-coordinate
         [0, 0, -1, ...]] z-coordinate
        :return: matrix 3*numJoints x 3*numConstraints
        """
        M = numpy.zeros((3 * self.num_joints, 3 * len(self.lu_pairs)), dtype='float32')

        ip = 0
        for p in self.lu_pairs:
            # x coordinate
            M[3 * p[0] + 0, ip] = 1
            M[3 * p[1] + 0, ip] = -1
            # y coordinate
            ip += 1
            M[3 * p[0] + 1, ip] = 1
            M[3 * p[1] + 1, ip] = -1
            # z coordinate
            ip += 1
            M[3 * p[0] + 2, ip] = 1
            M[3 * p[1] + 2, ip] = -1
            ip += 1

        return M

    def zz_projectionMat(self):
        """
        Generate a matrix that encodes the constraints
        [[1, 0, 0, ...]  x-coordinate, first joint
         [0, 1, 0, ...]  y-coordinate
         [0, 0, 1, ...]  z-coordinate
         [0, 0, 0, ...]
         ...
         [0, 0, 0, ...]
         [-1, 0, 0, ...]  x-coordinate, second joint
         [0, -1, 0, ...]  y-coordinate
         [0, 0, -1, ...]] z-coordinate
        :return: matrix 3*numJoints x 3*numConstraints
        """
        pair_v1M = numpy.zeros((3 * self.num_joints, 3 * len(self.zz_pairs)), dtype='float32')
        pair_v2M = numpy.zeros((3 * self.num_joints, 3 * len(self.zz_pairs)), dtype='float32')

        ip = 0
        for p in self.zz_pairs:
            # x coordinate
            pair_v1M[3 * p[0][0] + 0, ip] = 1
            pair_v1M[3 * p[0][1] + 0, ip] = -1
            pair_v2M[3 * p[1][0] + 0, ip] = 1
            pair_v2M[3 * p[1][1] + 0, ip] = -1
            # y coordinate
            ip += 1
            pair_v1M[3 * p[0][0] + 1, ip] = 1
            pair_v1M[3 * p[0][1] + 1, ip] = -1
            pair_v2M[3 * p[1][0] + 1, ip] = 1
            pair_v2M[3 * p[1][1] + 1, ip] = -1
            # z coordinate
            ip += 1
            pair_v1M[3 * p[0][0] + 2, ip] = 1
            pair_v1M[3 * p[0][1] + 2, ip] = -1
            pair_v2M[3 * p[1][0] + 2, ip] = 1
            pair_v2M[3 * p[1][1] + 2, ip] = -1
            ip += 1

        return pair_v1M, pair_v2M

    def getTemporalBreaks(self):
        """
        Breaks in sequence from i to i+1
        :return: list of indices
        """
        self.checkBreaks()

        return self.temporalBreaks

    def getSequenceBreaks(self):
        """
        Breaks in sequence from i to i+1
        :return: list of indices
        """
        self.checkBreaks()

        return self.sequenceBreaks

    def getHCBreaks(self):
        """
        Breaks in hard constraints from i to i+1
        :return: list of indices
        """
        self.checkBreaks()

        return self.hc_breaks

    def checkBreaks(self):
        """
        Check sequence, hc and temporal breaks, all must be congruent
        :return: None
        """

        # all hc breaks must be included in temporal an sequence breaks
        for hcb in self.hc_breaks:
            if hcb not in self.sequenceBreaks:
                raise ValueError("HC break {} definde, but not in sequence breaks {}!".format(hcb, self.sequenceBreaks))

            if hcb not in self.temporalBreaks:
                raise ValueError("HC break {} definde, but not in temporal breaks {}!".format(hcb, self.temporalBreaks))

        # all sequence breaks must be in temporal breaks
        for sb in self.sequenceBreaks:
            if sb not in self.temporalBreaks:
                raise ValueError("Sequence break {} definde, but not in temporal breaks {}!".format(sb, self.temporalBreaks))


class Blender2HandConstraints(HandConstraints):
    def __init__(self, seq):
        super(Blender2HandConstraints, self).__init__(24)

        if not isinstance(seq, list):
            raise ValueError("Parameter person must be list!")

        if len(seq) > 1:
            raise NotImplementedError("Not supported!")

        self.posebits = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (12, 13),
                         (14, 15), (15, 16), (16, 17), (17, 18), (19, 20), (20, 21), (21, 22), (22, 23)]

        self.hc_pairs = [(0, 1), (1, 2), (2, 3),
                         (4, 5), (5, 6), (6, 7), (7, 8),
                         (9, 10), (10, 11), (11, 12), (12, 13),
                         (14, 15), (15, 16), (16, 17), (17, 18),
                         (19, 20), (20, 21), (21, 22), (22, 23),
                         (0, 4), (4, 9), (9, 14), (14, 19)]  # almost all joints are constraint

        self.lu_pairs = []  # pairs only constrained by a range by lower and upper bounds

        # zig-zag constraint
        self.zz_pairs = [((1, 0), (2, 1)), ((2, 1), (3, 2)),
                         ((5, 4), (6, 5)), ((6, 5), (7, 6)), ((7, 6), (8, 7)),
                         ((10, 9), (11, 10)), ((11, 10), (12, 11)), ((12, 11), (13, 12)),
                         ((15, 14), (16, 15)), ((16, 15), (17, 16)), ((17, 16), (18, 17)),
                         ((20, 19), (21, 20)), ((21, 20), (22, 21)), ((22, 21), (23, 22))]
        self.zz_thresh = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # breaks in the sequence, eg due to pause in acquiring, important for temporal constraints
        # got by thresholding |pi-pi+1|^2 > 20000
        temporalBreaks = {'hpseq': [957], 'hpseq_loop_mv': [3039]}
        self.temporalBreaks = temporalBreaks[seq[0]]

        # breaks in the sequence for hard constraints, eg person changed, important for constraint handling
        # got by thresholding d(Li)-d(Li+1) > 1
        hc_breaks = {'hpseq': [957], 'hpseq_loop_mv': [3039]}
        self.hc_breaks = hc_breaks[seq[0]]

        # mark different recording sequences
        sequenceBreaks = {'hpseq': [957], 'hpseq_loop_mv': [3039]}
        self.sequenceBreaks = sequenceBreaks[seq[0]]

        # finger tips are allowed to be on the surface
        self.finger_tips = [3, 8, 13, 18, 23]

        # joint offset, must be this smaller than depth
        self.joint_off = [(20, ), (10, ), (5, ), (3, ),
                          (15, ), (5, ), (5, ), (5, ), (3, ),
                          (20, ), (7, ), (5, ), (5, ), (1, ),
                          (15, ), (7, ), (5, ), (5, ), (3, ),
                          (15, ), (4, ), (4, ), (4, ), (2, )]

        # bone length
        self.boneLength = numpy.asarray([[34.185734, 41.417736, 22.82596, 79.337524, 29.080118, 26.109291, 22.746597,
                                          74.122597, 36.783268, 28.834019, 24.202969, 66.111115, 32.468117, 27.345839,
                                          24.003115, 56.063339, 24.817657, 16.407534, 17.916492, 16.825773, 12.676471,
                                          15.136428, 8.8747139]])

        self.boneRanges = numpy.asarray([[[]]])

        self.noisePairs = [((0, 1), (1, 0)),
                         ((0, 1), (1, 2)),
                         ((1, 2), (2, 3)),
                         ((3, 2), (2, 3)),

                         ((5, 4), (4, 5)),
                         ((4, 5), (5, 6)),
                         ((5, 6), (6, 7)),
                         ((6, 7), (7, 8)),
                         ((8, 7), (7, 8)),

                         ((10, 9), (9, 10)),
                         ((9, 10), (10, 11)),
                         ((10, 11), (11, 12)),
                         ((11, 12), (12, 13)),
                         ((13, 12), (12, 13)),

                         ((15, 14), (14, 15)),
                         ((14, 15), (15, 16)),
                         ((15, 16), (16, 17)),
                         ((16, 17), (17, 18)),
                         ((18, 17), (17, 18)),

                         ((20, 19), (19, 20)),
                         ((19, 20), (20, 21)),
                         ((20, 21), (21, 22)),
                         ((21, 22), (22, 23)),
                         ((23, 22), (22, 23))]
        self.joint_dict = {"mcp": [[0, 1, 2], [4, 5, 6], [9, 10, 11], [14, 15, 16], [19, 20, 21]],
                           "pip": [[5, 6, 7], [10, 11, 12], [15, 16, 17], [20, 21, 22]],
                           "dip": [[6, 7, 8], [11, 12, 13], [16, 17, 18], [21, 22, 23]]}

class MSRA2HandConstraints(HandConstraints):
    """Class added to incorporate MSRA dataset : GJC5630"""
    def __init__(self, seq):
        super(MSRA2HandConstraints, self).__init__(21)

        if not isinstance(seq, list):
            raise ValueError("Parameter person must be list!")

        if len(seq) > 1:
            raise NotImplementedError("Not supported!")

        self.posebits = [(0,1),(1,2),(2,3),(3,4),
               (0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),
               (0,13),(13,14),(14,15),(15,16)
               ,(0,17),(17,18),(18,19),(19,20)]

        self.hc_pairs = [(0,1),(1,2),(2,3),(3,4),
               (0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),
               (0,13),(13,14),(14,15),(15,16)
               ,(0,17),(17,18),(18,19),(19,20)] # almost all joints are constraint

        self.lu_pairs = []  # pairs only constrained by a range by lower and upper bounds

        # zig-zag constraint
        self.zz_pairs = [((1, 0), (2, 1)), ((2, 1), (3, 2)),((3,2),(4,3)),
                         ((5, 0), (6, 5)), ((6, 5), (7, 6)), ((7, 6), (8, 7)),
                         ((9, 0), (10, 9)), ((10, 9), (11, 10)), ((11, 10), (12, 11)),
                         ((13, 0), (14, 13)), ((14, 13), (15, 14)), ((15, 14), (16, 15)),
                         ((17, 0), (18, 17)), ((18, 17), (19, 18)), ((19, 18), (20, 19))]
        self.zz_thresh = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # breaks in the sequence, eg due to pause in acquiring, important for temporal constraints
        # got by thresholding |pi-pi+1|^2 > 20000
        temporalBreaks = {'P0': [8498], 'P1': [8498], 'P2': [8499], 'P3': [8499], 'P4': [8499], 'P5': [8499], 'P6': [8499], 'P7': [8499], 'P8': [8499]}
        self.temporalBreaks = temporalBreaks[seq[0]]

        # breaks in the sequence for hard constraints, eg person changed, important for constraint handling
        # got by thresholding d(Li)-d(Li+1) > 1
        hc_breaks = {'P0': [8498], 'P1': [8498], 'P2': [8499], 'P3': [8499], 'P4': [8499], 'P5': [8499], 'P6': [8499], 'P7': [8499], 'P8': [8499]}
        self.hc_breaks = hc_breaks[seq[0]]

        # mark different recording sequences
        sequenceBreaks = {'P0': [8498], 'P1': [8498], 'P2': [8499], 'P3': [8499], 'P4': [8499], 'P5': [8499], 'P6': [8499], 'P7': [8499], 'P8': [8499]}
        self.sequenceBreaks = sequenceBreaks[seq[0]]

        # finger tips are allowed to be on the surface
        self.finger_tips = [4, 8, 12, 16, 20]

        # joint offset, must be this smaller than depth
        #['CT', 'T1', 'T2', 'T3', 'CI', 'I1', 'I2', 'I3', 'I4', 'CM', 'M1', 'M2', 'M3', 'M4',
         #'CR', 'R1', 'R2', 'R3', 'R4', 'CP', 'P1', 'P2', 'P3', 'P4']
        self.joint_off = [(20, ),
                          (5, ), (5, ), (5, ), (3, ),
                          (7, ), (5, ), (5, ), (1, ),
                          (7, ), (5, ), (5, ), (3, ),
                          (4, ), (4, ), (4, ), (2, ),
                          (10, ), (5, ), (3, ), (3, )]

        # bone length
        self.boneLength = numpy.asarray([[[68.9637694396768, 30.79999884543665, 23.099807667814037, 20.899925683360703,
                                          74.27702533253469, 36.30003332133455, 24.19997604234352, 21.999475401018096,
                                          77.44074024766874, 31.899742678861838, 20.9000184069895, 19.799822843864032,
                                          80.15929840517569, 23.099913829709394, 14.300112781988808, 15.399850997009693,
                                          30.705838208555708, 27.27987218316834, 22.00008182848418, 21.999874839848523]],
                                         [[68.96372676799015, 30.80005573306645, 23.099996270995373, 20.90002452941144,
                                          74.2768438303143, 36.30053616897276, 24.2000982082718, 21.999780868226836,
                                          77.44084154043007, 31.899926361357053, 20.899795910718826, 19.79954110711911,
                                          80.1591107078921, 23.10026539225252, 14.300394848380925, 15.399762037771861,
                                          30.161057457755014, 27.279902117493044, 22.0002202859994, 21.999891024786912,]],
                                         [[68.96368494657168, 30.800331530457633, 23.099576773441065, 20.90037139747525,
                                          74.2770839573014, 36.29963112622495, 24.199951513587788, 21.999847692384176,
                                          77.44030886599046, 31.900149245732347, 20.899847096330625, 19.799986497470176,
                                          80.15924321000044, 23.09998751623038, 14.300092268583443, 15.400144593477046,
                                          29.833699438219174, 27.279842470346143, 22.000541605206003, 21.99945698221887]],
                                         [[62.69435162071696, 27.999916132610174, 21.000057753793538, 19.000071544583722,
                                          67.52448491090915, 32.9997889922042, 21.99992313622934, 19.999959748199494,
                                          70.4004689402677, 29.000845287680878, 18.99973242258953, 18.00008159675949,
                                          72.87205889585938, 20.999640093344443, 12.99962690079988, 14.000055984173793,
                                          27.54751336490686, 24.799814187097756, 20.000075411708448, 20.00006933337983]],
                                         [[62.69434194510697, 27.999883129756096, 20.999979704059236, 18.999996770233942,
                                          67.52425609932777, 33.00062743140354, 21.999811818797458, 20.000034506970227,
                                          70.40071620381427, 29.00007699300124, 18.999567629459364, 18.00008260350213,
                                          72.87198216790867, 21.000094701929306, 12.999796249557164, 13.99989119672006,
                                          27.17115764648238, 24.800389363072515, 19.999767601899777, 19.999827595859415]],
                                         [[62.694332862707775, 27.999917191084695, 21.000069869493284, 19.00011753568909,
                                          67.52441253043229, 33.00080689649873, 21.999818477660227, 19.99985379996565,
                                          70.4007714922216, 28.999645190415663, 19.000006987893453, 17.999862067527054,
                                          72.87206959446124, 21.00002592403161, 13.000046026456976, 13.999799214274464,
                                          27.92835401164917, 24.800173465522388, 20.000027472231107, 19.99958803600714,]],
                                         [[56.42466821284466, 25.200017052248192, 18.899991084124874, 17.099937039065384,
                                          60.77227911021604, 29.699763699142483, 19.80018198142632, 18.00001009941106,
                                          63.36038376430498, 26.09985332832352, 17.099945650790826, 16.200418430707273,
                                          65.58438103306305, 18.900339262828073, 11.699889381101, 12.599927697014772,
                                          24.5996342812246, 22.319452382171054, 18.000652012913275, 17.999466478204308]],
                                         [[56.424680465023464, 25.20035509766479, 18.89963101081342, 17.10005107173951,
                                          60.77170084743391, 29.700170465840756, 19.800077736716094, 18.00002936219827,
                                          63.36088585594427, 26.099984793290595, 17.100177461067467, 16.199786504766056,
                                          65.58473950318016, 18.899935872907076, 11.700176084145042, 12.59966550349652,
                                          25.67912117265697, 22.32086477132999, 17.999580880953857, 18.000352852097112]],
                                         [[56.4249924021262, 25.200095009455822, 18.899480006867936, 17.099855098438706,
                                          60.7717800130949, 29.700569899077706, 19.800059837535844, 18.000640091952288,
                                          63.360594249501794, 26.099863602325573, 17.099970673951454, 16.199611831152023,
                                          65.5850017105283, 18.899895393890404, 11.70022793495922, 12.599753107104895,
                                          24.661286398726237, 22.319903964847196, 18.000161738717768, 18.000279719493268]]
                                         ])
        self.boneLength = self.boneLength[int(seq[0][-1])]
        self.boneRanges = numpy.asarray([[[]]])

        self.noisePairs = [((0, 1), (1, 0)),
                         ((0, 1), (1, 2)),
                         ((1, 2), (2, 3)),
                         ((2, 3), (3, 4)),
                         ((4, 3), (3, 4)),

                         ((0, 5), (5, 0)),
                         ((0, 5), (5, 6)),
                         ((5, 6), (6, 7)),
                         ((6, 7), (7, 8)),
                         ((8, 7), (7, 8)),

                         ((0, 9), (9, 0)),
                         ((0, 9), (9, 10)),
                         ((9, 10), (10, 11)),
                         ((10, 11), (11, 12)),
                         ((12, 11), (11, 12)),

                         ((0, 13), (13, 0)),
                         ((0, 13), (13, 14)),
                         ((14, 15), (15, 16)),
                         ((16, 15), (15, 16)),

                         ((0, 17), (17, 0)),
                         ((0, 17), (17, 18)),
                         ((18, 19), (19, 20)),
                         ((20, 19), (19, 20))]
        self.joint_dict = {"mcp": [[0, 1, 2], [0, 5, 6], [0, 9, 10], [0, 13, 14], [0, 17, 18]],
                           "pip": [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]],
                           "dip": [[2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18, 19, 20]]}

class NYUHandConstraints(HandConstraints):
    """Class added to incorporate MSRA dataset : GJC5630"""
    def __init__(self, seq):
        super(NYUHandConstraints, self).__init__(14)

        if not isinstance(seq, list):
            raise ValueError("Parameter person must be list!")

        if len(seq) > 1:
            raise NotImplementedError("Not supported!")

        self.posebits = [(0,1),(1,13),(2,3),(3,13),(4,5),(5,13),(6,7),(7,13),(8,9),(9,10),(10,13),(11,13),(12,13)]

        self.hc_pairs = [(0,1),(1,13),(2,3),(3,13),(4,5),(5,13),(6,7),(7,13),
                         (8,9),(9,10),(10,13),(11,13),(12,13)] # almost all joints are constraint

        self.lu_pairs = []  # pairs only constrained by a range by lower and upper bounds

        # zig-zag constraint
        self.zz_pairs = [((1, 0), (13, 1)), ((3, 2), (13, 3)),((5,4),(13,5)),
                         ((7, 6), (13, 7)), ((9, 8), (10, 9)), ((10, 9), (13, 10))]
        self.zz_thresh = [0, 0, 0, 0, 0, 0]

        # breaks in the sequence, eg due to pause in acquiring, important for temporal constraints
        # got by thresholding |pi-pi+1|^2 > 20000
        temporalBreaks = {'train': [516]}
        self.temporalBreaks = temporalBreaks[seq[0]]

        # breaks in the sequence for hard constraints, eg person changed, important for constraint handling
        # got by thresholding d(Li)-d(Li+1) > 1
        hc_breaks = {'train': [516]}
        self.hc_breaks = hc_breaks[seq[0]]

        # mark different recording sequences
        sequenceBreaks = {'train': [516]}
        self.sequenceBreaks = sequenceBreaks[seq[0]]

        # finger tips are allowed to be on the surface
        self.finger_tips = [0,2,4,6,8]

        # joint offset, must be this smaller than depth
        self.joint_off = [(15, ), (4, ), (2, ),
                          (15,), (7,), (2,),
                          (20,), (7,), (2,),
                          (15,), (5,), (2,),
                          (10, ), (10, )]

        # bone length
        self.boneLength = numpy.asarray([41.85334921577805,69.66413740885572,50.23376323156331,74.15567499326374,
                                          55.267100516074834,78.42223745524034,50.07083060557674,77.48224738164176,
                                          22.857164302443376,42.35612137756076,60.9853759003987,81.29413092056052,
                                          81.32021897718393])
        self.boneRanges = numpy.asarray([[[]]])

        self.noisePairs = [((0, 1), (1, 0)),
                         ((0, 1), (1, 13)),
                         ((1, 13), (13, 1)),

                         ((2, 3), (3, 2)),
                         ((2, 3), (3, 13)),
                         ((3, 13), (13, 3)),

                         ((4, 5), (5, 4)),
                         ((4, 5), (5, 13)),
                         ((5, 13), (13, 5)),

                         ((6, 7), (7, 6)),
                         ((6, 7), (7, 13)),
                         ((7, 13), (13, 7)),

                         ((8, 9), (9, 8)),
                         ((8, 9), (9, 10)),
                         ((9, 10), (10, 13)),
                         ((10, 13), (13, 10))]


class ICVLHandConstraints(HandConstraints):
    """Class added to incorporate MSRA dataset : GJC5630"""
    def __init__(self, seq):
        super(ICVLHandConstraints, self).__init__(21)

        if not isinstance(seq, list):
            raise ValueError("Parameter person must be list!")

        if len(seq) > 1:
            raise NotImplementedError("Not supported!")

        self.posebits = [(0,1),(1,2),(2,3),(3,4),
               (0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),
               (0,13),(13,14),(14,15),(15,16)
               ,(0,17),(17,18),(18,19),(19,20)]

        self.hc_pairs = [(0,1),(1,2),(2,3),(3,4),
               (0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),
               (0,13),(13,14),(14,15),(15,16)
               ,(0,17),(17,18),(18,19),(19,20)] # almost all joints are constraint

        self.lu_pairs = []  # pairs only constrained by a range by lower and upper bounds

        # zig-zag constraint
        self.zz_pairs = [((1, 0), (2, 1)), ((2, 1), (3, 2)),((3,2),(4,3)),
                         ((5, 0), (6, 5)), ((6, 5), (7, 6)), ((7, 6), (8, 7)),
                         ((9, 0), (10, 9)), ((10, 9), (11, 10)), ((11, 10), (12, 11)),
                         ((13, 0), (14, 13)), ((14, 13), (15, 14)), ((15, 14), (16, 15)),
                         ((17, 0), (18, 17)), ((18, 17), (19, 18)), ((19, 18), (20, 19))]
        self.zz_thresh = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # breaks in the sequence, eg due to pause in acquiring, important for temporal constraints
        # got by thresholding |pi-pi+1|^2 > 20000
        temporalBreaks = {'P0': [500], 'P1': [500], 'P2': [500], 'P3': [500], 'P4': [500], 'P5': [500], 'P6': [500], 'P7': [500], 'P8': [500]}
        self.temporalBreaks = temporalBreaks[seq[0]]

        # breaks in the sequence for hard constraints, eg person changed, important for constraint handling
        # got by thresholding d(Li)-d(Li+1) > 1
        hc_breaks = {'P0': [500], 'P1': [500], 'P2': [500], 'P3': [500], 'P4': [500], 'P5': [500], 'P6': [500], 'P7': [500], 'P8': [500]}
        self.hc_breaks = hc_breaks[seq[0]]

        # mark different recording sequences
        sequenceBreaks = {'P0': [500], 'P1': [500], 'P2': [500], 'P3': [500], 'P4': [500], 'P5': [500], 'P6': [500], 'P7': [500], 'P8': [500]}
        self.sequenceBreaks = sequenceBreaks[seq[0]]

        # finger tips are allowed to be on the surface
        self.finger_tips = [4, 8, 12, 16, 20]

        # joint offset, must be this smaller than depth
        self.joint_off = [(20, ), (10, ), (5, ), (3, ),
                          (15, ), (5, ), (5, ), (5, ), (3, ),
                          (20, ), (7, ), (5, ), (5, ), (1, ),
                          (15, ), (7, ), (5, ), (5, ), (3, ),
                          (15, ), (4, ), (4, ), (4, ), (2, )]

        # bone length
        self.boneLength = numpy.asarray([[68.9637694396768, 30.79999884543665, 23.099807667814037, 20.899925683360703,
                                          74.27702533253469, 36.30003332133455, 24.19997604234352, 21.999475401018096,
                                          77.44074024766874, 31.899742678861838, 20.9000184069895, 19.799822843864032,
                                          80.15929840517569, 23.099913829709394, 14.300112781988808, 15.399850997009693,
                                          30.705838208555708, 27.27987218316834, 22.00008182848418, 21.999874839848523],
                                         [68.96372676799015, 30.80005573306645, 23.099996270995373, 20.90002452941144,
                                          74.2768438303143, 36.30053616897276, 24.2000982082718, 21.999780868226836,
                                          77.44084154043007, 31.899926361357053, 20.899795910718826, 19.79954110711911,
                                          80.1591107078921, 23.10026539225252, 14.300394848380925, 15.399762037771861,
                                          30.161057457755014, 27.279902117493044, 22.0002202859994, 21.999891024786912,],
                                         [68.96368494657168, 30.800331530457633, 23.099576773441065, 20.90037139747525,
                                          74.2770839573014, 36.29963112622495, 24.199951513587788, 21.999847692384176,
                                          77.44030886599046, 31.900149245732347, 20.899847096330625, 19.799986497470176,
                                          80.15924321000044, 23.09998751623038, 14.300092268583443, 15.400144593477046,
                                          29.833699438219174, 27.279842470346143, 22.000541605206003, 21.99945698221887],
                                         [62.69435162071696, 27.999916132610174, 21.000057753793538, 19.000071544583722,
                                          67.52448491090915, 32.9997889922042, 21.99992313622934, 19.999959748199494,
                                          70.4004689402677, 29.000845287680878, 18.99973242258953, 18.00008159675949,
                                          72.87205889585938, 20.999640093344443, 12.99962690079988, 14.000055984173793,
                                          27.54751336490686, 24.799814187097756, 20.000075411708448, 20.00006933337983],
                                         [62.69434194510697, 27.999883129756096, 20.999979704059236, 18.999996770233942,
                                          67.52425609932777, 33.00062743140354, 21.999811818797458, 20.000034506970227,
                                          70.40071620381427, 29.00007699300124, 18.999567629459364, 18.00008260350213,
                                          72.87198216790867, 21.000094701929306, 12.999796249557164, 13.99989119672006,
                                          27.17115764648238, 24.800389363072515, 19.999767601899777, 19.999827595859415],
                                         [62.694332862707775, 27.999917191084695, 21.000069869493284, 19.00011753568909,
                                          67.52441253043229, 33.00080689649873, 21.999818477660227, 19.99985379996565,
                                          70.4007714922216, 28.999645190415663, 19.000006987893453, 17.999862067527054,
                                          72.87206959446124, 21.00002592403161, 13.000046026456976, 13.999799214274464,
                                          27.92835401164917, 24.800173465522388, 20.000027472231107, 19.99958803600714,],
                                         [56.42466821284466, 25.200017052248192, 18.899991084124874, 17.099937039065384,
                                          60.77227911021604, 29.699763699142483, 19.80018198142632, 18.00001009941106,
                                          63.36038376430498, 26.09985332832352, 17.099945650790826, 16.200418430707273,
                                          65.58438103306305, 18.900339262828073, 11.699889381101, 12.599927697014772,
                                          24.5996342812246, 22.319452382171054, 18.000652012913275, 17.999466478204308],
                                         [56.424680465023464, 25.20035509766479, 18.89963101081342, 17.10005107173951,
                                          60.77170084743391, 29.700170465840756, 19.800077736716094, 18.00002936219827,
                                          63.36088585594427, 26.099984793290595, 17.100177461067467, 16.199786504766056,
                                          65.58473950318016, 18.899935872907076, 11.700176084145042, 12.59966550349652,
                                          25.67912117265697, 22.32086477132999, 17.999580880953857, 18.000352852097112],
                                         [56.4249924021262, 25.200095009455822, 18.899480006867936, 17.099855098438706,
                                          60.7717800130949, 29.700569899077706, 19.800059837535844, 18.000640091952288,
                                          63.360594249501794, 26.099863602325573, 17.099970673951454, 16.199611831152023,
                                          65.5850017105283, 18.899895393890404, 11.70022793495922, 12.599753107104895,
                                          24.661286398726237, 22.319903964847196, 18.000161738717768, 18.000279719493268]
                                         ])
        self.boneLength = self.boneLength[int(seq[0][-1])]
        self.boneRanges = numpy.asarray([[[]]])

        self.noisePairs = [((0, 1), (1, 0)),
                         ((0, 1), (1, 2)),
                         ((1, 2), (2, 3)),
                         ((2, 3), (3, 4)),
                         ((4, 3), (3, 4)),

                         ((0, 5), (5, 0)),
                         ((0, 5), (5, 6)),
                         ((5, 6), (6, 7)),
                         ((6, 7), (7, 8)),
                         ((8, 7), (7, 8)),

                         ((0, 9), (9, 0)),
                         ((0, 9), (9, 10)),
                         ((9, 10), (10, 11)),
                         ((10, 11), (11, 12)),
                         ((12, 11), (11, 12)),

                         ((0, 13), (13, 0)),
                         ((0, 13), (13, 14)),
                         ((14, 15), (15, 16)),
                         ((16, 15), (15, 16)),

                         ((0, 17), (17, 0)),
                         ((0, 17), (17, 18)),
                         ((18, 19), (19, 20)),
                         ((20, 19), (19, 20))]
