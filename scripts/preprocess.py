import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import scipy.misc
import argparse



train = 'train'
img_train = 'img_train'
train_cropped = 'train_cropped'
img_train_cropped = 'img_train_cropped'
test_folder = 'test'
test_output_folder = 'test_output'

def prepare_test_data():
    #no cropping
    test_file_list = []
    [test_file_list.append(os.path.join(test_folder, f)) for f in os.listdir(test_folder)]
    file_dict = {'input' : test_file_list}
    df = pd.DataFrame(file_dict)
    csv = df.to_csv('test.csv',index = None, header= True, encoding= 'utf-8'  )

def prepare_train_data():
    if not os.path.exists(train_cropped):
        os.makedirs(train_cropped)
    if not os.path.exists(img_train_cropped):
        os.makedirs(img_train_cropped)
    img_train_filelist = []
    crop_info = []
    train_filelist = []
    avg_size = []
    for f in os.listdir(train):
        #read input image
        img = np.asarray(Image.open(os.path.join(train , f) ).convert('L'))

        #get crop coods
        rows , cols = np.nonzero(img)
        rightmost = cols.max()
        leftmost = cols.min()
        up = rows.min()
        down = rows.max()
        crop_info.append([up , down , leftmost , rightmost])
        #crop image
        img1 = img[ up : down + 1 , leftmost : rightmost + 1 ]

        #save cropped image as numpy
        filename , ext = f.split('.')
        train_filelist.append(os.path.join(train_cropped , filename + '.npy'))
        #scipy.misc.imsave(os.path.join(train_cropped , filename + '.png'), img1)
        np.save(os.path.join(train_cropped , filename) , img1)

        #read skeleton
        img = np.asarray(Image.open(os.path.join(img_train , f )).convert('L'))
        #crop
        img2 = img[ up : down + 1 , leftmost : rightmost + 1 ]
        #save cropped image as npy
        np.save(os.path.join(img_train_cropped , filename ), img2)
        #scipy.misc.imsave(os.path.join(img_train_cropped , filename + '.png'), img2)
        img_train_filelist.append(os.path.join(img_train_cropped , filename + '.npy')) # add to csv

    fileDict = {'input' : train_filelist , 'output': img_train_filelist , 'crop_info' : crop_info}
    df = pd.DataFrame(fileDict)
    dataframe = df.to_csv('train.csv' ,index = None , header = True , encoding = 'utf-8' )


def main():
    prepare_test_data()



if __name__ == "__main__":
    main()
