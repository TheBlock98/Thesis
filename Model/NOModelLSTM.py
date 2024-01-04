
"""
We want to build a LSTM model to predict future price  
(HLC3) of BTC using dataSetIMRCleaned.csv like dataSet.
First of all  we want to predict HLC3 to one step, in the second phase we want build multiOutput model to
predict HLC3 to:  1 steps in this phase. We implement using pyTorch.

We want use Boruta to do PCA and select important features.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score


config = {
    "data": {
        "window_size": 720,
        "train_split_size": 0.70,
        "test_split_size": 0.20,
        "valid_split_size": 0.10,
      
    },
    
    "model": {
        "input_size": 62,  # since we are only using 62 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 64,
        "dropout": 0.8,
        "num_class": 1,
        "model_path": "Model/modelLSTMV0.pth",
        "model_name": "LSTMV0.txt",
        "save_model": True,
    },
    "training": {
        #"device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epochs": 100,
        "learning_rate": 0.01, # 1e-3
        "loss_function": nn.MSELoss(),
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 3,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-6,
        "early_stopping": True,
        "early_stopping_patience": 5,
        
        
        
      },
}

dataSet = pd.read_csv("/content/dataSetIMRNormalized.csv")
hlc3 = dataSet['hlc3']
dataSet.drop(labels=['hlc3'], axis=1, inplace=True)
dataSet.insert(0, 'hlc3', hlc3)

# Dataloading (sliding windows)
config["model"]["input_size"] = dataSet.shape[1] # with 1 have number of features


from google.colab import drive
drive.mount('/content/drive')

def sliding_windows(df, windowSize):
  x = []
  y = []
  for i in range(len(df) - windowSize):
    # Convert DataFrame slice to values and append to x
    _x = df.iloc[i:(i + windowSize)].values
    # Append the hlc3 value at position i + windowSize as the target
    _y = df.iloc[i + windowSize]["hlc3"]
    x.append(_x)
    y.append(_y)

  return np.array(x), np.array(y)


x, y = sliding_windows(dataSet, config["data"]["window_size"])
train_size = int(len(y) * config["data"]["train_split_size"])
val_size = int(len(y) * config["data"]["valid_split_size"])
test_size = int(len(y) * config["data"]["test_split_size"])

# transform from np.array  x and y to tensors
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

print(dataX.size())
# split seqences into train and test
trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))
valX = Variable(torch.Tensor(np.array(x[train_size:train_size + val_size])))
valY = Variable(torch.Tensor(np.array(y[train_size:train_size + val_size])))
print(valX.size())
testX = Variable(torch.Tensor(np.array(x[train_size + val_size:])))
testY = Variable(torch.Tensor(np.array(y[train_size + val_size:])))


class LSTM(nn.Module):
  def __init__(self, input_size, num_lstm_layers, lstm_size, dropout, num_class):
    super(LSTM, self).__init__()
    self.input_size = config["model"]["input_size"]
    self.num_lstm_layers = num_lstm_layers
    self.lstm_size = lstm_size
    self.dropout = dropout
    self.num_class = num_class

    # Fully connected layer as input layer
    self.fc_input = nn.Linear(62, lstm_size) 

    
    # Define the LSTM layers
    self.lstm = nn.LSTM(input_size=input_size,
                        hidden_size=lstm_size,
                        num_layers=num_lstm_layers,
                        dropout=dropout,
                        batch_first=True)

    # Define the Dropout layer
    self.dropout_layer = nn.Dropout(dropout)

    # Define the output fully connected layer
    # Define the output fully connected layer
    self.fc_output = nn.Linear(lstm_size, num_class)


  def forward(self, x):

    # Pass data through the input fully connected layer
    x = F.relu(x)
    
    # Initialize hidden and cell states with zeros
    h_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_size))
    c_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.lstm_size))

      # Pass through LSTM layers
    out = x
    for i in range(self.num_lstm_layers):
      out, (h_0, c_0) = self.lstm(out, (h_0.detach(), c_0.detach()))
      out = F.relu(out)
      out = self.dropout_layer(out)

    # We take the output of the last sequence only for the fully connected layer
    out = out[:, -1, :]
    out = self.fc(out)
    return out

model = LSTM(input_size=config["model"]["input_size"],
             num_lstm_layers=config["model"]["num_lstm_layers"],
             lstm_size=config["model"]["lstm_size"],
             dropout=config["model"]["dropout"],
             num_class=1)

optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
# optimizer = optim.RMSprop(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=weight_decay)
# optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["learning_rate"])
criterion = torch.nn.MSELoss()    # mean-squared error for regression
scheduler = config["training"]["scheduler"](optimizer, patience=config["training"]["scheduler_patience"],
                                            factor=config["training"]["scheduler_factor"], 
                                            min_lr=config["training"]["scheduler_min_lr"])


best_val_loss = float('inf')
patience_counter = 0

class EarlyStoppingException(Exception):
  pass

def training(model, optimizer, criterion, scheduler, trainX, trainY, valX, valY, epoch, best_val_loss,patience_counter):
  print(f"Epoch {epoch+1} \n ---------- ")
  model.train()
  optimizer.zero_grad()
  outputs = model(trainX)
  loss = criterion(outputs, trainY)
  loss.backward()
  optimizer.step()

  # Evaluate the model and compute validation loss
  model.eval()
  with torch.no_grad():
      val_outputs = model(valX)
      val_loss = criterion(val_outputs, valY)

  # Reduced learning rate when a metric has stopped improving
  scheduler.step(val_loss)
  # Check for early stopping
  if config["training"]["early_stopping"]:
      if val_loss < best_val_loss:
          best_val_loss = val_loss.item()  # Capture the best validation loss
          patience_counter = 0  # Reset counter if validation loss improves
      else:
          patience_counter += 1
          # Stop training if val_loss has not improved after patience limit
          if patience_counter >= config["training"]["early_stopping_patience"]:
              print(f'Early stopping: val_loss has not improved ' +
                    f'for {config["training"]["early_stopping_patience"]} consecutive epochs.')
              raise EarlyStoppingException()
              
    
  # Output training/validation statistics
  print(f'Epoch[{epoch+1}/{config["training"]["num_epochs"]}] | ' +
        f'Loss train: {loss.item():.6f}, ' +
        f'Loss val: {val_loss.item():.6f} | ' +
        f'LR: {optimizer.param_groups[0]["lr"]:.6f}'+
       f"best validation loss: {best_val_loss:.6f}" )
  return loss, val_loss, best_val_loss, patience_counter


def testing(model,testX,testY,criterion):
    # Test the model
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        test_outputs = model(testX)
        test_loss = criterion(test_outputs, testY)
        print(f'Test loss: {test_loss.item():.6f}')
    # compute performance KPY MSE, RMSE, MAE, accuracy, r2
    y_pred = test_outputs.detach().numpy()
    y_true = testY.detach().numpy()
    kpy_mse = mean_squared_error(y_true, y_pred)
    kpy_rmse = np.sqrt(kpy_mse)
    kpy_mae = mean_absolute_error(y_true, y_pred)
    kpy_r2 = r2_score(y_true, y_pred)
    kpy_accuracy = accuracy_score(y_true, y_pred > 0.5)
    print(f'KPY MSE: {kpy_mse:.6f}, RMSE: {kpy_rmse:.6f}, MAE: {kpy_mae:.6f}, ' +
          f'Accuracy: {kpy_accuracy:.6f}, R2: {kpy_r2:.6f}')
    return test_outputs,kpy_mse, kpy_rmse, kpy_mae, kpy_r2,kpy_accuracy


for epoch in range(config["training"]["num_epochs"]):
    try:
        print(f"Epoch {epoch + 1}\n-------------------------------")
        loss, val_loss, best_val_loss, patience_counter = training(model, optimizer, criterion, scheduler, trainX, trainY, valX, valY, epoch, best_val_loss, patience_counter)
        test_outputs, kpy_mse, kpy_rmse, kpy_mae, kpy_r2, kpy_accuracy = testing(model, testX, testY, criterion)
    except EarlyStoppingException:
        print("Early stopping triggered, proceeding to the next epoch...")
        break
    print(f"Best validation loss: {best_val_loss:.6f}")
print("Done!")


# Save the model
torch.save(model.state_dict(), config["training"]["model_path"])
print(f'ModelLSTMV0 saved to {config["training"]["model_path"]}')


data_predict = test_outputs.data.numpy()
dataY_plot = testY.data.numpy()
# denormalize data
data_predict = scaler.inverse_transform(data_predict)
dataY_plot = scaler.inverse_transform(dataY_plot)
# invert scaling for plotting
dataY_plot = dataY_plot[:, 0]
data_predict = data_predict[:, 0]
# invert scaling for plotting
dataY_plot = dataY_plot[-len(data_predict):]
data_predict = data_predict[-len(data_predict):]
# Plot predictions vs. actual
plt.plot(dataY_plot, label='Data')
plt.plot(data_predict, label='Predictions')
plt.legend()
plt.axvline(x=train_size, c='r', linestyle='--')
plt.suptitle('Time-Series Prediction')
plt.show()



  
