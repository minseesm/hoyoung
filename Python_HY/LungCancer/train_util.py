import torch
import torch.nn.init
from torch.utils.data import DataLoader

import os           # File path
import sys          # sys.exit()

from dataset.data_preset import LungCancerDataset
import model.Linear as structure

import matplotlib.pyplot as plt

################
# Configration #
################

def configuration():
    print("====== Setting to train your model ======")
    epochs = int(input("Epoch : "))
    batch_size = int(input("Batch_size : "))
    learning_rate = float(input("Learning Rate : "))

    return epochs, batch_size, learning_rate

########################
# Ready to Train Model #
########################

def train_model(select):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if select == '1':
        model = structure.Linear().to(device)
        print(model)
    elif select == '2':
        node_num = int(input("node num : "))
        model = structure.Linear_custom(node_num).to(device)
        print(model)
    
    return device, model

########################
# Design Training Flow #
########################

def training(epochs, batch_size, learning_rate, model, device):

    # Configration model
    criterion = torch.nn.MSELoss().to(device)  # With softmax function in cost function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []

    model.train()           # when test, model.eval()

    # Get data set
    current_path = os.path.dirname(__file__)

    input_path = f"{current_path}/dataset/lungcancer.csv"

    dataset = LungCancerDataset(input_path, "Train")
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    total_batch = len(data_loader)

    print('Total batch size : {}'.format(total_batch))

    for epoch in range(epochs):
        avg_cost = 0

        for X, Y in data_loader:        # X : img  Y : label

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            output = model(X)
            cost = criterion(output, Y)

            cost.backward()
            optimizer.step()

            avg_cost += cost.item()/batch_size
        
        print(cost.item())
        if epoch % 10 == 0:
            losses.append(avg_cost)

        print("--- Epoch : {:4d}  cost : {:.8} ---".format(epoch + 1, cost.item()))

    plt.plot(losses)
    plt.show()

    return model

def go_test():
    test_select = input("Do you want to test model right now? (y/n) : ")
    if test_select == 'y':
        return True
    else :
        sys.exit()