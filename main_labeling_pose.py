"""This is the main file for annotation of depth files

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
import collections
import getopt
import numpy
import os
import sys
import pwd
from PyQt5.QtWidgets import QApplication
from data.importers import Blender2Importer
from data.importers import MSRA15Importer
from util.handconstraints import Blender2HandConstraints
from util.handconstraints import MSRA2HandConstraints
from util.handpose_evaluation import Blender2HandposeEvaluation
from util.handpose_evaluation import MSRA2HandposeEvaluation
from util.interactivedatasetlabeling import InteractiveDatasetLabeling
from util.sort import find_reference_frames

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


if __name__ == '__main__':
    person = "hpseq_loop_mv"
    start_idx = 0
    dataset = "blender"
    data_folder = "/home/giffy/Documents/Datasets/Blender_dataset/Blender"
    # data_folder = "/home/giffy/Documents/Datasets/cvpr15_MSRAHandGestureDB"
    eval_folder  = "blender_pose"

    if dataset is None:
        print("Dataset must be specified:")
        dataset = raw_input().lower()
        if len(dataset.strip()) == 0:
            sys.exit(2)

    if data_folder is None:
        print("Data location must be specified:")
        data_folder = raw_input()
        if len(data_folder.strip()) == 0:
            sys.exit(2)
    if eval_folder is None:
        print("eval location must be specified:")
        eval_folder = raw_input()
        if len(eval_folder.strip()) == 0:
            sys.exit(2)

    if person is None:
        print 'Person must be specified:'
        person = raw_input().lower()
        if len(person.strip()) == 0:
            sys.exit(2)

    if start_idx is None:
        print 'Start frame index can be specified:'
        start_idx = raw_input().lower()
        if len(start_idx.strip()) == 0:
            start_idx = 0
        else:
            start_idx = int(start_idx)

    rng = numpy.random.RandomState(23455)

    # subset to label
    subset_idxs = []

    if dataset == 'blender':
        blender_persons = ["hpseq_loop_mv"]
        while person not in blender_persons:
            print("Invalid person name. Valid names are ", blender_persons)
            person = raw_input("Please enter one of the valid person names.").lower()
        di = Blender2Importer(data_folder+'/', useCache=True)
        Seq2 = di.loadSequence(person, camera=0, shuffle=False)
        hc = Blender2HandConstraints([Seq2.name])
        hpe = Blender2HandposeEvaluation([j.gt3Dorig for j in Seq2.data], [j.gt3Dorig for j in Seq2.data])
        for idx, seq in enumerate(Seq2.data):
            ed = {'vis': [], 'pb': {'pb': [], 'pbp': []}}
            Seq2.data[idx] = seq._replace(gtorig=numpy.zeros_like(seq.gtorig), extraData=ed)

        # common subset for all
        dpt = collections.deque()
        for d in Seq2.data:
            dpt.append(d.dpt)
        dpt = numpy.asarray(dpt)
        #0.035
        reference_frame_calculator = find_reference_frames(eval_folder, force=True, distance_threshold=0.2)
        subset_idxs = reference_frame_calculator.calculate_reference_frames(dpt)
        print(len(subset_idxs))
        del reference_frame_calculator
    elif dataset == 'msra':
        blender_persons = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
        while person not in blender_persons:
            print("Invalid person name. Valid names are ", blender_persons)
            person = raw_input("Please enter one of the valid person names.").lower()
        di = MSRA15Importer(data_folder+'/', useCache=True)
        Seq2 = di.loadSequence(person, shuffle=False)
        hc = MSRA2HandConstraints([Seq2.name])
        hpe = MSRA2HandposeEvaluation([j.gt3Dorig for j in Seq2.data], [j.gt3Dorig for j in Seq2.data])
        for idx, seq in enumerate(Seq2.data):
            ed = {'vis': [], 'pb': {'pb': [], 'pbp': []}}
            Seq2.data[idx] = seq._replace(gtorig=numpy.zeros_like(seq.gtorig), extraData=ed)

        # common subset for all
        dpt = collections.deque()
        for d in Seq2.data:
            dpt.append(d.dpt)
        dpt = numpy.asarray(dpt)
        #0.035
        reference_frame_calculator = find_reference_frames(eval_folder, force=True, distance_threshold=0.2)
        subset_idxs = reference_frame_calculator.calculate_reference_frames(dpt)
        print(len(subset_idxs))
        del reference_frame_calculator

    else:
        raise NotImplementedError("")

    replace_off = 0
    replace_file = None  # './params_tracking.npz'

    output_path = di.basepath

    filename_joints = output_path+person+"/joint.txt"
    filename_pb = output_path+person+"/pb.txt"
    filename_vis = output_path+person+"/vis.txt"
    filename_log = output_path+person+"/annotool_log.txt"

    # create empty file
    if not os.path.exists(filename_joints):
        annoFile = open(filename_joints, "w+")
        annoFile.close()
    else:
        bak = filename_joints+".bak"
        i = 0
        while os.path.exists(bak):
            bak = filename_joints+".bak.{}".format(i)
            i += 1
        os.popen("cp "+filename_joints.replace(" ", "\\")+" "+bak)

    if not os.path.exists(filename_pb):
        annoFile = open(filename_pb, "w+")
        annoFile.close()
    else:
        bak = filename_pb+".bak"
        i = 0
        while os.path.exists(bak):
            bak = filename_pb+".bak.{}".format(i)
            i += 1
        os.popen("cp "+filename_pb+" "+bak)

    if not os.path.exists(filename_vis):
        annoFile = open(filename_vis, "w+")
        annoFile.close()
    else:
        bak = filename_vis+".bak"
        i = 0
        while os.path.exists(bak):
            bak = filename_vis+".bak.{}".format(i)
            i += 1
        os.popen("cp "+filename_vis+" "+bak)

    app = QApplication(sys.argv)
    browser = InteractiveDatasetLabeling(Seq2, hpe, di, hc, filename_joints, filename_pb, filename_vis, filename_log,
                                         subset_idxs, start_idx, replace_file, replace_off)
    browser.show()
    app.exec_()
    print browser.curData, browser.curVis, browser.curPb, browser.curPbP, browser.curCorrected

