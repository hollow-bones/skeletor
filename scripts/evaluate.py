import os
from dataloader import * 
from model import *
import sys
import argparse

sys.path.append("./desire")


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--dset_type', default='test', type=str)

def initialize_UNET(model_path):
    net = UNET().to(DEVICE)
    path = os.path.join(os.getcwd(), model_path)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

def evaluate(args):
        output_folder = 'test_output'
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        transform = transforms.ToPILImage()

        # Init the net and load weights
        net = UNET().to(DEVICE)
        path = os.path.join(os.getcwd(), model_path)
        net.load_state_dict(torch.load(args.model_path))
        net.eval()

        net = initialize_UNET()
        for _,batch in enumerate(testloader):
                image = batch['images'].to(DEVICE)
                out_image = net(image)
                out_image = out_image[0].to('cpu')
                out_image = transform(out_image) 
                out_image.save(os.path.join(output_folder,batch['names'][0] ))

def main():
        evaluate()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
