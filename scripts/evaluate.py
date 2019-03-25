import os
import sys
from skeletor.misc import get_freer_gpu
from skeletor.dataloader import TestDataset
import logging
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from skeletor.unet import UNet

def initialize_UNET(restore_model_path, device):

    net = UNet().to(device)
    net.load_state_dict(torch.load(restore_model_path))
    net.eval()
    return net


def evaluate(restore_model_path,
             test_csv_file,
             output_folder=None):


    trainloader_params = {'batch_size': 3,
                          'shuffle': True,
                          'num_workers': 4}

    logging.info("Setting the train loader")
    test_dset = TestDataset(test_csv_file)
    testloader = torch.utils.data.DataLoader(test_dset,
                                              **trainloader_params)



    device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    if output_folder is None:
        output_folder = 'test_output'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.ToPILImage()

    net = initialize_UNET(restore_model_path, device)
    for _,batch in enumerate(testloader):
            image = batch['images'].to(device)
            out_image = net(image)
            out_image = out_image[0].cpu()
            out_image = transform(out_image)
            out_image.save(os.path.join(output_folder,batch['names'][0] ))


if __name__ == "__main__":
    restore_model_path = '/home/akaberto/learn/skeletor/weights/weights_000300.pth'
    test_csv_file = "/home/akaberto/learn/skeletor/test.csv"
    evaluate(restore_model_path, test_csv_file)
