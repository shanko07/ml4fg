#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, torch, torch.nn as nn, pickle
from sklearn.model_selection import train_test_split
import glob, os, timeit, numpy as np, gc
import torch.nn.functional as F
from onehotcy import one_hot


# In[2]:


torch.cuda.is_available()


# In[3]:


all_data = pd.read_csv('../megasequence-to-cancer.csv')


# In[4]:


all_data.head()


# In[ ]:





# In[ ]:


#genome = pickle.load(open('../hg19.pickle', 'rb'))


# In[ ]:





# In[7]:

# TODO: remove sequences that are > 60M bp because it is just too long to fit in memory even for batch = 1
all_data = all_data[all_data['sequence'].str.len() < 60000000]


# In the first step we will split the data in training and remaining dataset
data_train, data_rem = train_test_split(all_data, train_size=0.8, random_state=543)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
data_validation, data_test = train_test_split(data_rem, test_size=0.5, random_state=82)


# In[ ]:





# In[8]:


# for chip-seq data: also shuffling nucleotides can be done to keep the GC content the same as positive example
# because sequencing has biases with GC content and this would be a way to "fix it"
class BedPeaksDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_set):
        super(BedPeaksDataset, self).__init__()
        self.atac_data = data_set

    def __iter__(self): 
        for i,row in enumerate(self.atac_data.itertuples()):
            seq = row.sequence
            value = np.float32(1)
            if row._3 == "Normal":
                value = np.float32(0)
            yield(one_hot(seq), value) # positive example

train_dataset = BedPeaksDataset(data_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers = 0)

#validation_dataset = BedPeaksDataset(data_validation)
#validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000, num_workers=0)

#test_dataset = BedPeaksDataset(data_test)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, num_workers=0)

data_train


# In[ ]:





# In[9]:


def run_one_epoch(train_flag, dataloader, model, optimizer, device="cuda"):

    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []
    accuracies = []

    for (x,y) in dataloader: # collection of tuples with iterator
        
        torch.cuda.empty_cache()
        gc.collect()

        (x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = model(x) # forward pass
        output = output.squeeze(1) # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable
        losses.append(loss.detach().cpu().numpy())
        op = output.detach().cpu().numpy()
        yd = y.detach().cpu().numpy()
        
        #del loss
        del output

        if train_flag: 
            loss.backward() # back propagation
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            optimizer.zero_grad()

        
        #accuracy = torch.mean( ( (op > .5) == (yd > .5) ).astype(int) )
        accuracy = np.mean( ( (op > .5) == (yd > .5) ).astype(int) )
        #accuracies.append(accuracy.detach().cpu().numpy())
        accuracies.append(accuracy)
        print(losses[-1], accuracies[-1])
        del loss
        #del output
        del x
        del y
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return( np.mean(losses), np.mean(accuracies) )


# In[14]:


def train_model(model, train_data, validation_data, epochs=100, patience=10, batch_size=1000, verbose = True):
    """
    Train a 1D CNN model and record accuracy metrics.
    """
    # Move the model to the GPU here to make it runs there, and set "device" as above
    # TODO CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Make new BedPeakDataset and DataLoader objects for both training and validation data.
    # TODO CODE
    train_dataset = BedPeaksDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers = 0, timeout=0)
    validation_dataset = BedPeaksDataset(validation_data)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers = 0, timeout=0)

    # 2. Instantiates an optimizer for the model. 
    # TODO CODE
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=False)

    # 3. Run the training loop with early stopping. 
    # TODO CODE
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    patience_counter = patience
    best_val_loss = np.inf
    
    
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob('checkpoints/model_checkpoint*.pt')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    
    
    check_point_filename = 'checkpoints/model_checkpoint.pt' # to save the best model fit to date
    local_min_counter = 0
    for epoch in range(epochs):
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss: 
            if patience_counter < patience:
                local_min_counter = local_min_counter + 1
            torch.save(model.state_dict(), f'checkpoints/model_checkpoint-{local_min_counter}.pt')
            best_val_loss = val_loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(f'checkpoints/model_checkpoint-{local_min_counter}.pt')) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" % 
              (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter ))

    # 4. Return the fitted model (not strictly necessary since this happens "in place"), train and validation accuracies.
    # TODO CODE
    return model, train_accs, val_accs, train_losses, val_losses, validation_dataloader


torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):

    def __init__(self, 
                 n_output_channels = 1, 
                 n_hidden = 32, 
                 dropout = 0.2,
                 n_fc = 1,
                 lstm_hidden=10):
        
        super(LSTM, self).__init__()
        self.lstm = nn.Sequential(nn.LSTM(5, lstm_hidden, batch_first=True, dropout=dropout))
        
        fc_layers = [nn.Linear(lstm_hidden, n_hidden)]
        for i in range(n_fc-1):
            fc_layers += [ nn.Dropout(dropout),
                          nn.ELU(inplace=True),
                          nn.Linear(n_hidden, n_hidden)
            ]
        fc_layers += [nn.Dropout(dropout),
                      nn.ELU(inplace=True),
                      nn.Linear(n_hidden, n_output_channels)
        ]
        self.dense_net = nn.Sequential(*fc_layers)

    def forward(self, x):
        
        #print(x.size())
        # switch sequence and channels and send to LSTM
        net, (hn, cn) = self.lstm(x.swapaxes(1, 2))
        #print(net.size())
        net = net[:, -1, :]
        #print(net.size())
        net = net.view(net.size(0), -1)
        #print(net.size())
        net = self.dense_net(net)
        return(net)


torch.cuda.empty_cache()

lstm_model = LSTM(dropout=.28, n_hidden=32, n_fc=1, lstm_hidden=2)

lstm_model, train_accs, val_accs, train_losses, val_losses, lstm_validation_datloader = train_model(lstm_model, data_train
                                                                         , data_validation, epochs=300
                                                                         , patience=20, batch_size=1)