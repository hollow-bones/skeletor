import torch 
import torchvision 
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import scipy.misc
import scipy.ndimage
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 1
IMAGE_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.pad_width = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        names = self.data.iloc[idx]
        image = np.load(names['input'])
        mask = np.load(names['output'])
        angle = np.random.randint(0, high = 360)
        image = torch.tensor(self.transform_image(image , angle)).float() 
        mask = torch.tensor(self.transform_image(mask , angle)).float()  
        sample = {}
        sample['images'] = image
        sample['masks'] = mask
        return sample
    
    def transform_image(self,image, angle):
        image = np.pad(image, pad_width= self.pad_width ,mode='constant', constant_values=0)
        image = scipy.ndimage.rotate(image , angle)
        image = scipy.misc.imresize(image , (IMAGE_SIZE , IMAGE_SIZE))
        return image

class TestDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.pad_width = 0
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        names = self.data.iloc[idx]
        #print(names['input'])
        image = Image.open(names['input'])
        image = self.transform(image)
        # angle = np.random.randint(0, high = 360)
        # image = torch.tensor(self.transform_image(image , angle)).float() 
        # mask = torch.tensor(self.transform_image(mask , angle)).float()  
        sample = {}
        name = names['input'].split('\\')[1]
        sample['images'] = image
        sample['names'] = name
        return sample
    

trainDataset = TrainDataset('train.csv' )
trainloader = torch.utils.data.DataLoader(trainDataset,batch_size= BATCH_SIZE, shuffle=True,num_workers=4)   
testDataset = TestDataset('test.csv')
testloader = torch.utils.data.DataLoader(testDataset,batch_size= 1, shuffle=True,num_workers=4)   


# def main():
#     # batch = next(iter(trainloader))
#     # images = batch['images']
#     # masks = batch['masks']
#     # for i in range(len(images)):
#     #     img1 = Image.fromarray(images[i].numpy() )
#     #     img2 = Image.fromarray(masks[i].numpy() )
#     #     img1.show()
#     #     img2.show()

#     batch = next(iter(testloader))
#     images = batch['images']
#     for i in range(len(images)):
#         img1 = Image.fromarray(images[i].numpy() )
#         img1.show()
        

# if __name__ == "__main__":
#     main()

    
