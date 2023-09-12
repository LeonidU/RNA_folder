import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
#import UNet
import gc

epochs = 5000
bs = 256
batch_size = 4
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()

class BasicUNet(nn.Module):
    def __init__(self):
        super(BasicUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)

        )
        self.middle = nn.Sequential(
            nn.Conv1d(64, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3
    
def corrupt(xs, amount):
    noise = torch.rand(xs.shape, dtype=torch.double)
    amount = amount.view(-1, 1) 
    return xs * (1-amount) + noise * amount



net = BasicUNet().to(torch.double).to(device)
train_ds, valid_ds = dataset.prepare_dataset("../RNA_2/", device=device,batch_size=batch_size) #merge_ds.map(preprocess_fn).shuffle(1024).batch(bs).prefetch(tf.data.AUTOTUNE)

# Define a loss function
loss_fn = nn.MSELoss()

# Define an optimizer
opt = optim.Adam(net.parameters(), lr=1e-4)

# Record the losses
losses, avg_losses = [], []
val_losses = []


amount = torch.linspace(0.0, 1.0, batch_size) # Left to right -> more corruption
# Iterate over epochs.
print("train_ds=",len(train_ds))
print("valid_ds=",len(valid_ds))

for epoch in tqdm(range(epochs)):
    torch.cuda.empty_cache()
    gc.collect()
    val_losses = []
    # Iterate over the batches of the dataset.
    for step, (xb,) in enumerate(train_ds):
        xb_copy = xb[:].to(torch.double).to(device)
        labels = xb[:, 1, :].to(torch.double)
        labels_source = xb[:, 2, :].to(torch.double)
        xb = xb[:, 0, :].double()
        # Create noisy version of the input
        noise_amount = torch.rand((xb.shape[0],), dtype=torch.double)
        noisy_xb = corrupt(xb, noise_amount)
        noised_tensor = torch.cat([noisy_xb, labels, labels_source], dim=1)
        noised_tensor = noised_tensor.view(batch_size,3, dataset.WINDOW*dataset.SIZE)
#        print(noised_tensor.shape)
        # Get the model prediction
        pred = net(noised_tensor.double().to(device))
#        print(pred.shape)
#        print(xb_copy.shape)
        # Calculate the loss to determine how close the output is to the input
        loss = loss_fn(pred, xb_copy)
        del noised_tensor
        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Store the loss
        losses.append(loss.item())
#    print("train_ds=",len(train_ds))
#    print("valid_ds=",len(valid_ds))
    torch.cuda.empty_cache()
    if epoch % 50 == 0:
        with torch.no_grad():
            for step, (xb,) in enumerate(valid_ds):
                xb_copy = xb[:].to(torch.double).to(device)
                labels = xb[:, 1, :].to(torch.double)
                labels_source = xb[:, 2, :].to(torch.double)
                xb = xb[:, 0, :].double()
                # Create noisy version of the input
                noise_amount = torch.rand((xb.shape[0],), dtype=torch.double)
                noisy_xb = corrupt(xb, noise_amount)
                noised_tensor = torch.cat([noisy_xb, labels, labels_source], dim=1)
                noised_tensor = noised_tensor.view(batch_size,3, dataset.WINDOW*dataset.SIZE)
                pred = net(noised_tensor.double().to(device))
                print("PRED", pred[:,0,:])
                print("XB", xb_copy[:,0,:])
                val_loss = loss_fn(pred[:,0,:], xb_copy[:,0,:])
                val_losses.append(val_loss) 
#            del xb
#            del noise_amount
#            del xb_copy
            del noised_tensor
    # Calculate the average loss for this epoch
#    avg_loss = sum(losses[-len(xb):])/len(xb)
#    avg_losses.append(avg_loss)
        print("Average validation set loss = %5.5f" % (sum(val_losses)/len(val_losses)))

#print(avg_losses[-1])
