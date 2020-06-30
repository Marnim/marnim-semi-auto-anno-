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

import getopt
import numpy
import os
import pwd
from PyQt4.QtGui import QApplication
from data.importers import Blender2Importer
from util.handpose_evaluation import Blender2HandposeEvaluation
from util.interactivedetector import InteractiveDetector
import sys

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
    data_folder = "/home/giffy/Documents/Datasets/Blender"

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

    if dataset == 'blender':
        blender_persons = ["hpseq_loop_mv"]
        while person not in blender_persons:
            print("Invalid person name. Valid names are ",blender_persons)
            person = raw_input("Please enter one of the valid person names.").lower()
        di = Blender2Importer(data_folder+'/', useCache=True)
        Seq2 = di.loadSequence(person, camera=0, shuffle=False)
        hpe = Blender2HandposeEvaluation([j.gt3Dorig for j in Seq2.data], [j.gt3Dorig for j in Seq2.data])
        for idx, seq in enumerate(Seq2.data):
            Seq2.data[idx] = seq._replace(com=numpy.zeros((3,)))
    else:
        raise NotImplementedError("")

    # we need to detect all files
    subset_idxs = []

    output_path = di.basepath

    filename_dets = output_path+person+'/detections.txt'
    filename_log = output_path+person+'/detecttool_log.txt'

    # create empty file
    if not os.path.exists(filename_dets):
        annoFile = open(filename_dets, "w")
        annoFile.close()
    else:
        bak = filename_dets+'.bak'
        i = 0
        while os.path.exists(bak):
            bak = filename_dets+'.bak.{}'.format(i)
            i += 1
        os.popen('cp '+filename_dets+' '+bak)

    app = QApplication(sys.argv)
    browser = InteractiveDetector(Seq2, hpe, di, filename_dets, filename_log, subset_idxs, start_idx)
    browser.show()
    app.exec_()
