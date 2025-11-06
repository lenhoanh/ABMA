import os
import glob
import numpy as np
import scipy.io as scio


# https://github.com/vt-le/astnet
class LabelVideoDataset:
    def __init__(self, dataset_name, test_set):
        """
        param dataset_name='ped2', 'avenue', 'shanghaitech', 'chad'
        """
        root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
        self.dataset_name = dataset_name

        # shanghaitech (files '.npy' in folder test_frame_mask)
        if self.dataset_name == 'shanghaitech':
            self.anomaly_labels = os.path.join(root, self.dataset_name, 'test_frame_mask/*')
        elif self.dataset_name == 'ped2' or self.dataset_name == 'avenue':
            self.anomaly_labels = os.path.join(root, self.dataset_name, self.dataset_name + '.mat')
        elif self.dataset_name == 'chad':
            self.anomaly_labels = os.path.join(root, self.dataset_name, 'CHAD_Meta', 'anomaly_labels/*')
        else:  # iitb, ped2_shorten, avenue_shorten, shanghaitech_shorten, iitb_shorten
            self.anomaly_labels = os.path.join(root, self.dataset_name, 'anomaly_labels/*')

        test_dataset_path = os.path.join(root, self.dataset_name, test_set)
        video_folders = (os.listdir(test_dataset_path))
        video_folders.sort()
        self.video_folders = [os.path.join(test_dataset_path, folder) for folder in video_folders]

    def __call__(self):
        print(f'__call__(): self.dataset_name = {self.dataset_name}')

        if self.dataset_name == 'ped2' or self.dataset_name == 'avenue':  # 'mat'
            # load file 'ped2.mat' or 'avenue.mat'
            abnormal_mat = scio.loadmat(self.anomaly_labels, squeeze_me=True)['gt']

            all_gt = []
            for i in range(abnormal_mat.shape[0]):
                length = len(os.listdir(self.video_folders[i]))
                sub_video_gt = np.zeros((length,), dtype=np.int8)

                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1  # TODO
                    end = one_abnormal[1, j]
                    sub_video_gt[start: end] = 1

                all_gt.append(sub_video_gt)

            return all_gt
        else: # self.dataset_name == 'shanghaitech', 'chad', iitb, ped2_shorten, avenue_shorten, shanghaitech_shorten, iitb_shorten:
            np_list = glob.glob(self.anomaly_labels)
            np_list.sort()

            gt = []
            for npy in np_list:
                gt.append(np.load(npy))

            return gt
