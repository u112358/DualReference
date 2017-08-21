import unittest
from util.file_reader import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        CACD = FileReader(data_dir='/home/bingzhang/Documents/Dataset/CACD/CACD2000',
                          data_info='/home/bingzhang/Documents/Dataset/CACD/celenew.mat', reproducible=True,
                          contain_val=True, val_list='./data/val_list.txt')

        val_path = CACD.get_val(144)
        print(val_path)
        for i in range(10000):
            path,label = CACD.select_age_path(20,20)
            print(path,label)

if __name__ == '__main__':
    unittest.main()
