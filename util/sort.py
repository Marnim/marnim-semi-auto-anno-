import os
import pickle
import numpy as np
import cv2
from sklearn.metrics.pairwise import pairwise_distances

class find_reference_frames:
    def __init__(self, eval_folder, dist_func = "cosine", distance_threshold = 0.2, force = False):
        self.eval_folder = "./eval/"+eval_folder+"/"
        self.dist_func = dist_func
        self.distance_threshold = distance_threshold
        self.force = force
    def calculate_reference_frames(self, train_data):
        ref_array_file = "ref_array.npy"
        if not os.path.isfile(self.eval_folder+ref_array_file) or self.force:
            dst = pairwise_distances(train_data.reshape(train_data.shape[0],-1), metric=self.dist_func).reshape(train_data.shape[0], train_data.shape[0])
            sort_array = np.ones(train_data.shape[0], dtype=int) * -1
            sort_array[0] = 0
            dst[:,0] = np.inf
            min_distance = []
            for i in range(1,sort_array.shape[0]):
                dst[sort_array[i-1],sort_array[i-1]] = np.inf
                sub_array = dst[sort_array[i-1]]
                min_dst = sub_array.min()
                min_distance.append(min_dst)
                min_loc = np.where(sub_array == min_dst)[0]
                sort_array[i] = min_loc[0]
                dst[:,sort_array[i]] = np.inf
            min_distance = np.asarray(min_distance)
            cv2.destroyAllWindows()

            print(min_distance.min(), min_distance.max(), np.median(min_distance))
            breaks = []
            median = np.median(min_distance)
            prev = 0
            ref_frames = []
            for i in range(min_distance.shape[0]):
                if min_distance[i] > self.distance_threshold:
                    ref_frames.append(sort_array[prev])
                    prev = i
            np.save(self.eval_folder+ref_array_file, ref_frames)
            np.save(self.eval_folder+"sort_array", sort_array)
        else:
            ref_frames = np.load(self.eval_folder + ref_array_file)
        return ref_frames