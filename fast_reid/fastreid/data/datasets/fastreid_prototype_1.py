# import glob
# import os
# import os.path as osp
# import re
# import warnings
# from fast_reid.fastreid.data.datasets import DATASET_REGISTRY
# from fast_reid.fastreid.data.datasets.bases import ImageDataset


# # @DATASET_REGISTRY.register()
# # class FastREID_Prototype_1(ImageDataset):
# #     _junk_pids = [0, -1]
# #     dataset_dir = ''
# #     dataset_url = '' 
# #     dataset_name = "FastREID_Prototype_1"
    
# #     def __init__(self, root='datasets', **kwargs):
# #         self.root = root
# #         self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
# #         # allow alternative directory structure
# #         self.data_dir = self.dataset_dir
# #         data_dir = osp.join(self.data_dir, 'FastREID_Prototype_1')
# #         if osp.isdir(data_dir):
# #             self.data_dir = data_dir
# #         else:
# #             warnings.warn('The current data structure is deprecated. Please '
# #                           'put data folders such as "bounding_box_train" under '
# #                           '"MOT17-ReID".')
        
# #         self.train_dir = osp.join(self.data_dir, 'train')
# #         print(self.train_dir)
# #         self.query_dir = osp.join(self.data_dir, 'test')
# #         print(self.query_dir)
# #         self.gallery_dir = osp.join(self.data_dir, 'test')
# #         self.extra_gallery_dir = osp.join(self.data_dir, 'train')
# #         self.extra_gallery = False

# #         required_files = [
# #             self.data_dir,
# #             self.train_dir,
# #             # self.query_dir,
# #             # self.gallery_dir,
# #         ]
        
# #         self.check_before_run(required_files)

# #         train = lambda: self.process_dir(self.train_dir)
# #         query = lambda: self.process_dir(self.query_dir, is_train=False)
# #         gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
# #                           (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])
        
# #         super(FastREID_Prototype_1, self).__init__(train, query, gallery, **kwargs)
        
# #     def process_dir(self, dir_path, is_train=True):

# #         img_paths = glob.glob(osp.join(dir_path, '*.bmp'))
# #         pattern = re.compile(r'([-\d]+)_MOT17-([-\d]+)-FRCNN')

# #         data = []
# #         for img_path in img_paths:
# #             pid, camid = map(int, pattern.search(img_path).groups())
# #             if pid == -1:
# #                 continue  # junk images are just ignored
# #             # assert 0 <= pid   # pid == 0 means background
# #             # assert 1 <= camid <= 5
# #             camid -= 1  # index starts from 0
# #             if is_train:
# #                 pid = self.dataset_name + "_" + str(pid)
# #                 camid = self.dataset_name + "_" + str(camid)
# #             data.append((img_path, pid, camid))

# #         return data
    
# @DATASET_REGISTRY.register()
# class FastREID_Prototype_1(ImageDataset):
#     dataset_dir = ''
#     dataset_name = "FastREID_Prototype_1"
    
#     def __init__(self, root='datasets', **kwargs):
#         self.root = root
#         self.dataset_dir = osp.join(self.root, self.dataset_dir)
        
#         # allow alternative directory structure
#         self.data_dir = self.dataset_dir
#         data_dir = osp.join(self.data_dir, 'FastREID_Prototype_1')
#         if osp.isdir(data_dir):
#             self.data_dir = data_dir
            
#         self.train_dir = osp.join(self.data_dir, 'train')
#         self.query_dir = osp.join(self.data_dir, 'test')
#         self.gallery_dir = osp.join(self.data_dir, 'test')
#         self.extra_gallery_dir = osp.join(self.data_dir, 'train')
#         self.extra_gallery = False

#         self.convert_labels = {
#             'Brahmos Missile': 1,
#         }
        
#         required_files = [
#             self.data_dir,
#             self.train_dir,
#             # self.query_dir,
#             # self.gallery_dir,
#         ]
        
#         self.check_before_run(required_files)
        
#         # train = lambda: self.get_data(self.train_dir, 1)
#         # query = lambda: self.get_data(self.query_dir, 2)
#         # gallery = lambda: self.get_data(self.gallery_dir, 3)

#         # super(FastREID_Prototype_1, self).__init__(train, query, gallery, **kwargs)
        
#         train_data = self.get_data(self.train_dir, 1)
#         val_data = self.get_data(self.query_dir, 2)
#         gallery_data = self.get_data(self.gallery_dir, 3)

#         super().__init__(train_data, val_data, gallery_data)
        
#     def get_data(self, path, cam_id):
#         data = []
#         absolute_path = osp.join(path)
#         sub_1_dirs = os.listdir(absolute_path)
#         for sub_1_dir in sub_1_dirs:
#             sub_1_path = osp.join(absolute_path, sub_1_dir)
#             if sub_1_dir == '.DS_Store':
#                 continue
#             filenames = os.listdir(sub_1_path)
#             for filename in filenames:
#                 if filename == '.DS_Store':
#                     continue
#                 filepath = osp.join(sub_1_path, filename)
#                 data.append((filepath, self.convert_labels[sub_1_dir], cam_id))
#         return data
    
import os
from fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from fast_reid.fastreid.data.datasets.bases import ImageDataset

@DATASET_REGISTRY.register()
class SuperClassDataset(ImageDataset):
    def __init__(self, root='/home/cipoll17/BoT-SORT/datasets', **kwargs):
        train_path = root + '/FastREID_Prototype_1/train'
        val_path = root + '/FastREID_Prototype_1/val'
        gallery_path = root + '/FastREID_Prototype_1/train'

        self.convert_labels = {
            'Brahmos_Missle': 1,
            'brahmos_missle': 1,
        }
        
        train_data = self.get_data(train_path, 1)
        val_data = self.get_data(val_path, 2)
        gallery_data = self.get_data(gallery_path, 3)

        super().__init__(train_data, val_data, gallery_data)
    def get_data(self, path, cam_id):
        data = []
        absolute_path = os.path.join(path)
        sub_1_dirs = os.listdir(absolute_path)
        for sub_1_dir in sub_1_dirs:
            sub_1_path = os.path.join(absolute_path, sub_1_dir)
            if sub_1_dir == '.DS_Store':
                continue
            filenames = os.listdir(sub_1_path)
            for filename in filenames:
                if filename == '.DS_Store':
                    continue
                filepath = os.path.join(sub_1_path, filename)
                data.append((filepath, self.convert_labels[sub_1_dir], cam_id))
        return data
    
if __name__ == "__main__":
    # dataset = FastREID_Prototype_1()
    dataset = SuperClassDataset()