from model import *
from dataloader import *
import torch.optim as optim
import os

sys.path.append("./desire")


load_model = False
criterion = nn.BCELoss( reduction= 'mean')
sigmoid = nn.Sigmoid()
epochs = 400
# def calculate_loss(outputs , masks) :
#     loss = torch.zeros(1).to(DEVICE)
#     for i in range(BATCH_SIZE):
#         loss += criterion(softmax(outputs[i]), softmax(masks[i]))
#     return loss

def train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if load_model:
        net = UNET().to(device)
        PATH = os.path.join(os.getcwd(), 'model.pt')
        net.load_state_dict(torch.load(PATH))
        net.eval()
        print('loaded model')
    else :
        net = UNET().to(device)

    #criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print('training begins')
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            images, masks = data['images'] , data['masks']
            images = images.view([BATCH_SIZE,1,IMAGE_SIZE , IMAGE_SIZE]).to(device)
            masks = masks.view([BATCH_SIZE,1,IMAGE_SIZE , IMAGE_SIZE]).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(sigmoid(outputs), sigmoid(masks))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1200 == 1199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 1200))
                running_loss = 0.0

    print('Finished Training')
    PATH = os.path.join(os.getcwd() , 'model_new.pt')
    torch.save(net.state_dict(), PATH)

def main():
    train()

if __name__ == "__main__":
    main()
