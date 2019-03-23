import logging
import os
import sys
import time


import torch.optim as optim
from torch.optim import lr_scheduler
from skeletor.losses import BCEDicedLoss
from skeletor.misc import get_freer_gpu
from skeletor.dataloader import TrainDataset
from skeletor.unet import UNet

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

import argparse


def train(train_csv_file,
          epochs=800,
          batch_size=3,
          save_weight_interval=100,
          model_restore_path=None,
          model_save_root="./weights",
          model_save_name="unet_run1"):


    trainloader_params = {'batch_size': 3,
                          'shuffle': True,
                          'num_workers': 4}

    logging.info("Setting the train loader")
    train_dset = TrainDataset(train_csv_file,
                              rotate=False)
    trainloader = torch.utils.data.DataLoader(train_dset,
                                              **trainloader_params)


    device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    net = UNet()

    if model_restore_path is not None:
        logging.info("Loading previous model from {}".format(model_restore_path))
        net.load_state_dict(torch.load(model_restore_path))
        logging.info("Successfully Loaded model")

    net.to(device)
    logging.info(net)
    net.training()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr= .01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = BCEDicedLoss()

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs
            images, masks = data['images'] , data['masks']
            images = images.to(device)
            masks = masks.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(images, masks)
            loss.backward()

            # print statistics
            running_loss += loss.item()
            if i % 100:
                logging.info('[{}, {}] loss: {}'.format((epoch + 1,
                                                         i + 1,
                                                         running_loss / (1200/batch_size))))
                running_loss = 0.0
        scheduler.step()

        if(epoch % save_weight_interval == 0):
            save_path = os.path.join(model_save_root,
                                     '{}_{}'.format(model_save_root,
                                                    str(epoch).zfill(int(np.log(epochs)))))
            torch.save(net.state_dict(), save_path)

            save_path = os.path.join(model_save_root,
                                     '{}_{}'.format(model_save_root,
                                                    str(epoch).zfill(int(np.log(epochs)))))
    torch.save(net.state_dict(), save_path)
    logging.info("Done with training")


if __name__ == "__main__":
    train_csv_file = os.path.abspath("raw_data/train.csv")
    train(train_csv_file)
