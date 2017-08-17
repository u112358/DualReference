import unittest
from util.file_reader import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        CACD = FileReader(data_dir='/home/bingzhang/Documents/Dataset/CACD',data_info='celenew.mat',reproducible=True,contain_val=False)
        print CACD.__str__()
        paths,labels =  CACD.select_identity_path(10,10)
        print labels

if __name__ == '__main__':
    unittest.main()
