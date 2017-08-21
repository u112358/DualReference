from util.file_reader import *
import os

CACD = FileReader(data_dir='/scratch/BingZhang/dataset/CACD2000_Cropped',
                  data_info='/scratch/BingZhang/dataset/CACD2000/celenew2.mat', reproducible=True,
                  contain_val=False, val_list='./data/val_list.txt')
for i in range(CACD.total_images):
    try:
        path_t = os.path.join(CACD.prefix, CACD.path[i][0].encode('utf-8').replace('jpg','png'))
    except:
        path_t = os.path.join(CACD.prefix, CACD.path[i][0].replace('jpg','png'))
    if not os.path.isfile(path_t):
        print(CACD.path[i][0])
