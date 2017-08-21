from util.file_reader import *
import os

CACD = FileReader(data_dir='/scratch/BingZhang/dataset/CACD2000_Cropped',
                  data_info='/scratch/BingZhang/dataset/CACD2000/celenew.mat', reproducible=True,
                  contain_val=False, val_list='./data/val_list.txt')
path = CACD.path
for i in path:
    try:
        path_t = os.path.join(CACD.prefix, i.encode('utf-8'))
    except:
        path_t = os.path.join(CACD.prefix, i)
    if not os.path.isfile(path_t):
        print(i)
        