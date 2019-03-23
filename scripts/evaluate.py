import os
import sys
from dataloader import *
from skeletor.misc import get_freer_gpu

def initialize_UNET(device):

    net = UNet().to(device)
    PATH = os.path.join(os.getcwd(), sys.argv[1])
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net


def evaluate(restore_model_path,
             output_folder=None):

    device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    if output_folder is None:
        output_folder = 'test_output'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.ToPILImage()

    net = initialize_UNET(device)
    for _,batch in enumerate(testloader):
            image = batch['images'].to(DEVICE)
            out_image = net(image)
            out_image = out_image[0].cpu()
            out_image = transform(out_image)
            out_image.save(os.path.join(output_folder,batch['names'][0] ))


if __name__ == "__main__":
    evaluate()
