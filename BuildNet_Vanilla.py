from dataloader import TFDNA_ds
import torch
import os
import helper_classes as hp
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import EncoderNet, FCN, LinNet
import torch.optim as optim
from tqdm.notebook import tqdm
from datetime import datetime


window_size = 300
slide_step = 200
n_epochs = 3
batch_size = 20
f_train = 0.9
best_loss = np.inf

os.chdir('/home/labs/barkailab/benjak/Python/Deeplearning/project/')
file = open("log.txt", "a")
file.write("Begin task: " + str(datetime.now()) + "\n")

# allows the access to the files
os.chdir('/home/labs/antebilab/naamab_lab/Benny/Deep_learning_project/')
TF_path = 'Amino_acid_data/final_tf_data.h5'
DNA_path = 'signal_159_TFs'
data = TFDNA_ds(TF_path=TF_path, DNA_path=DNA_path, seq_length=window_size, sliding_window_step=slide_step)
os.chdir('/home/labs/barkailab/benjak/Python/Deeplearning/project/')

###
input_size = 1502 + 1 + window_size
N_emb = [26, 128, 64, 16, 1]
N_hidden = [input_size, input_size * 4, input_size, window_size]
net = LinNet(N_hidden=N_hidden, N_emb=N_emb, window_size=window_size)
###
loss_func = hp.Custom_MSE_Loss(c=1)
optimizer = optim.Adam(net.parameters(), lr=1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

N = (10 ** 5) * 3
N_train = int(f_train * N)
training_dataset = torch.utils.data.Subset(data, range(N_train))
validation_dataset = torch.utils.data.Subset(data, range(N_train, N))
training_loss = []
validation_loss = []

N_GPUs = torch.cuda.device_count()
if N_GPUs > 1:
    batch_size = batch_size * N_GPUs
net = nn.DataParallel(net)

file.write("Running with # GPUs: " + str(N_GPUs) + "\n")
data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
net.to(device)


# loss_checkpoint = torch.load('loss_hist.pt')
# validation_loss = loss_checkpoint['v_loss']
# training_loss = loss_checkpoint['t_loss']


#### @@@@ LOAD PREV @@@@
# checkpoint = torch.load('trained_model.pt')
# net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# best_loss = checkpoint['best_loss']


def compute_accuracy_and_loss(dataloader, net, valid=False):
    total = 0
    correct = 0
    loss = 0

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    n_batches = 0
    with torch.no_grad():
        for batched_e, l in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                batched_e = batched_e.to(torch.device('cuda'))
                l = l.cuda()
            pred = net(batched_e)

            loss += loss_func(pred, l).item()

    loss = loss / n_batches
    return loss


pbar = tqdm(range(n_epochs))

for epoch in pbar:
    file.write("Starting epoch #: " + str(epoch) + "\n")
    if len(validation_loss) > 1:
        pbar.set_description(
            'val loss:' + '{0:.5f}'.format(validation_loss[-1]) + ', train loss:' + '{0:.5f}'.format(
                training_loss[-1]))

    net.train()  # put the net into "training mode"
    for batched_e, l in data_loader:
        if torch.cuda.is_available():
            batched_e = batched_e.to(torch.device('cuda'))
            l = l.cuda()

        optimizer.zero_grad()
        pred = net(batched_e)
        loss = loss_func(pred, l)
        loss.backward()
        optimizer.step()

    scheduler.step()
    net.eval()  # put the net into evaluation mode
    train_loss = compute_accuracy_and_loss(data_loader, net)
    valid_loss = compute_accuracy_and_loss(validation_data_loader, net, valid=True)
    file.write("valid loss = " + str(valid_loss) + "\n")
    torch.cuda.empty_cache()

    training_loss.append(train_loss)
    validation_loss.append(valid_loss)
    torch.save({'t_loss': training_loss, 'v_loss': validation_loss}, 'loss_hist.pt')

    # keep model with best loss
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        },
            'trained_model.pt')
        file.write("Is best loss \n")

    plt.figure()
    plt.plot(training_loss, label='training')
    plt.plot(validation_loss, label='validation')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

file.write("Finished \n")
