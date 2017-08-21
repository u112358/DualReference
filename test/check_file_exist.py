from util.file_reader import *
import os

CACD = FileReader(data_dir='/scratch/BingZhang/dataset/CACD2000_Cropped',
                  data_info='/scratch/BingZhang/dataset/CACD2000/celenew.mat', reproducible=True,
                  contain_val=False, val_list='./data/val_list.txt')
path = CACD.path
for i in range(CACD.total_images):
    try:
        path_t = os.path.join(CACD.prefix, path[i].encode('utf-8'))
    except:
        path_t = os.path.join(CACD.prefix, path[i])
    if not os.path.isfile(path_t):
        print(path[i])
