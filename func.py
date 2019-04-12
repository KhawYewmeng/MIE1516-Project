import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, GRU, Concatenate, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import RepeatVector
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.utils import plot_model
import tensorflow.python.keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
import copy

def load_data_fc(path,window):
  # loads and pre-processes the data for architecture A1
  df = pd.read_csv(path)
  # transform the data
  test_size=0.2
  df_norm = df
  for i in range(0,30):
      df_norm.iloc[:,i] = StandardScaler().fit_transform(df_norm.iloc[:,i].values.reshape(-1, 1)) 
  feature_no = df_norm.shape[1]
  df_norm = df_norm.values
  # reshaping into the desired window
  df_norm = df_norm[0:df_norm.shape[0]//window*window]
  df_norm = df_norm.reshape( (df_norm.shape[0]//window*window)//window,window,feature_no)
  # shuffling the reshaped data
  np.random.seed(100)
  randIndx = np.arange(df_norm.shape[0])
  np.random.shuffle(randIndx)
  df_norm = df_norm[randIndx]
  # splitting the data
  train_x = df_norm[0: int((1-test_size)*df_norm.shape[0])]
  test_x = df_norm[int((1-test_size)*df_norm.shape[0]):]
  # remove anomalous data from train_x
  index = (train_x[:,:,30]==1).all(axis=1)
  train_x = train_x[~index]
  # remove the class column from train_x
  train_x = train_x[:,:,0:30]
  # test_y contains the class label
  test_y = test_x[:,:,30]
  # remove the class column from test_x
  test_x = test_x[:,:,0:30]
  train_x = train_x.reshape(train_x.shape[0],30)
  test_x = test_x.reshape(test_x.shape[0],30)
  test_y = test_y.reshape(test_y.shape[0],1)
  return train_x,test_x,test_y,feature_no

def load_data(path,window):
  # loads and pre-processes the data for architecture A2 and A5
  df = pd.read_csv(path)
  # transform the data
  test_size=0.2
  df_norm = df
  for i in range(0,30):
      df_norm.iloc[:,i] = StandardScaler().fit_transform(df_norm.iloc[:,i].values.reshape(-1, 1)) 
  feature_no = df_norm.shape[1]
  df_norm = df_norm.values
  # reshaping into the desired window
  df_norm = df_norm[0:df_norm.shape[0]//window*window]
  df_norm = df_norm.reshape( (df_norm.shape[0]//window*window)//window,window,feature_no)
  # shuffling the reshaped data
  np.random.seed(100)
  randIndx = np.arange(df_norm.shape[0])
  np.random.shuffle(randIndx)
  df_norm = df_norm[randIndx]
  # splitting the data
  train_x = df_norm[0: int((1-test_size)*df_norm.shape[0])]
  test_x = df_norm[int((1-test_size)*df_norm.shape[0]):]
  # remove anomalous data from train_x
  index = (train_x[:,:,30]==1).all(axis=1)
  train_x = train_x[~index]
  # remove the class column from train_x
  train_x = train_x[:,:,0:30]
  # test_y contains the class label
  test_y = test_x[:,:,30]
  # remove the class column from test_x
  test_x = test_x[:,:,0:30]
  return train_x,test_x,test_y,feature_no

def load_data_a4(path,window):
  # loads and pre-processes the data for architecture A4
  data = pd.read_csv(path)
  feature_no = data.shape[1]
  # transform the data
  test_size=0.2
  for i in range(0,30):
      data.iloc[:,i] = StandardScaler().fit_transform(data.iloc[:,i].values.reshape(-1, 1)) 
  data = data.values
  # reshaping into the desired window
  data = data[0:data.shape[0]//window*window]
  data = data.reshape( (data.shape[0]//window*window)//window,window,feature_no)
  # shuffling the reshaped data
  np.random.seed(100)
  randIndx = np.arange(data.shape[0])
  np.random.shuffle(randIndx)
  data = data[randIndx] 
  # splitting the data
  train_x = data[0: int((1-test_size)*data.shape[0])]
  test_x = data[int((1-test_size)*data.shape[0]):]
  # remove anomalous data from train_x
  index = (train_x[:,:,30]==1).all(axis=1)
  train_x = train_x[~index]
  # remove the class column from train_x
  train_x = train_x[:,:,0:30]
  # test_y contains the class label
  test_y = test_x[:,:,30]
  # remove the class column from test_x
  test_x = test_x[:,:,0:30]  
  train_x_red = copy.deepcopy(train_x)
  train_x_red[:,0:window-1,:] = train_x_red[:,1:window,:] 
  train_x_red[:,window-1,:] = 0  
  test_x_red = copy.deepcopy(test_x)
  test_x_red[:,0:window-1,:] = test_x_red[:,1:window,:] 
  test_x_red[:,window-1,:] = 0  
  test_y_red = copy.deepcopy(test_y)
  test_y_red[:,0:window-1] = test_y_red[:,1:window] 
  test_y_red[:,window-1] = 0  
  return train_x,test_x,test_y,feature_no, train_x_red, test_x_red, test_y_red

def plot_history(history):
  # plots accuracy and loss curves
  val_acc = history["val_acc"]
  train_acc = history["acc"]
  epoch  = range(0,len(val_acc))
  plt.figure()
  plt.title('Accuracy Plot')
  plt.plot(epoch, val_acc, 'green', label='Test Accuracy')
  plt.plot(epoch, train_acc, 'blue', label='Train Accuracy')
  plt.legend(loc = 'best')
  plt.show()
  val_loss = history["val_loss"]
  train_loss = history["loss"]
  epoch  = range(0,len(val_loss))
  plt.figure()
  plt.title('Loss Plot')
  plt.plot(epoch, val_loss, 'green', label='Test Loss')
  plt.plot(epoch, train_loss, 'blue', label='Train Loss')
  plt.legend(loc = 'best')
  plt.show()

def error_fc (test_y,test_x,model):
    # creates an error table and calculates the best F1 score for architecture A1
    test_y = test_y.reshape(56962)
    test_x_pred = model.predict(test_x)
    mse = np.mean(np.power(test_x - test_x_pred, 2), axis=1)
    error_df = pd.DataFrame({'MSE': mse,
                            'True_class': test_y})
    precision, recall, threshold = precision_recall_curve(error_df.True_class, error_df.MSE)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1_valid_range = F1.shape[0]-15
    max_F1 = np.amax(F1[:F1_valid_range ])
    max_F1_index = np.argmax(F1[:F1_valid_range ])
    return max_F1_index,max_F1,precision, recall, threshold,error_df

def error (test_y,test_x,model):
    # creates an error table and calculates the best F1 score for architecture A2 and A5
    test_x_pred = model.predict(test_x)
    test_x = test_x.reshape(test_x.shape[0]*test_x.shape[1],30)
    test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1])
    test_x_pred = test_x_pred.reshape(test_x_pred.shape[0]*test_x_pred.shape[1],30)
    mse = np.mean(np.power(test_x - test_x_pred, 2), axis=1)
    error_df = pd.DataFrame({'MSE': mse,
                            'True_class': test_y})
    precision, recall, threshold = precision_recall_curve(error_df.True_class, error_df.MSE)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1_valid_range = F1.shape[0]-15
    max_F1 = np.amax(F1[:F1_valid_range ])
    max_F1_index = np.argmax(F1[:F1_valid_range ])
    return max_F1_index,max_F1,precision, recall, threshold,error_df

def error_a4 (test_y,test_x,model,window,test_x_red):
  # creates an error table and calculates the best F1 score for architecture A4
  test_x_pred = model.predict(test_x)
  # removing the last transaction in each window since they are set to 0
  test_x_pred = test_x_pred[:,0:window-1,:]
  test_y = test_y[:,0:window-1]
  test_x_red = test_x_red[:,0:window-1,:]
  test_x_red = test_x_red.reshape(test_x_red.shape[0]*test_x_red.shape[1],30)
  test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1])
  test_x_pred = test_x_pred.reshape(test_x_pred.shape[0]*test_x_pred.shape[1],30)
  mse = np.mean(np.power(test_x_red - test_x_pred, 2), axis=1)
  error_df = pd.DataFrame({'MSE': mse,
                          'True_class': test_y})  
  precision, recall, threshold = precision_recall_curve(error_df.True_class, error_df.MSE)
  F1 = 2 * (precision * recall) / (precision + recall)
  F1_valid_range = F1.shape[0]-15
  max_F1 = np.amax(F1[:F1_valid_range ])
  max_F1_index = np.argmax(F1[:F1_valid_range ])
  return max_F1_index,max_F1,precision, recall, threshold,error_df

def MSE_scatter(max_F1_index,error_df,threshold):
    # plots the MSE scatter plot
    threshold_fixed = threshold[max_F1_index]
    groups = error_df.groupby('True_class')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.index, group.MSE, marker='o', ms=3, linestyle='',
                label= "Fraud" if name == 1 else "Normal")
    ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend(loc='best')
    plt.title("MSE Scatter Plot")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data index")
    plt.show()
    return threshold_fixed

def conf_matrix(threshold_fixed, error_df):
    # plots the confusion matrix corresponding to the best F1 score
    LABELS = ["Normal","Fraud"]
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.MSE.values]
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    
def print_results(max_F1,max_F1_index, recall, precision,error_df):
    # prints evaluation matrices
    print("Best F1 score:",max_F1)
    print("Recall at best F1 score:",recall[max_F1_index])
    print("Precision at best F1 score:",precision[max_F1_index])
    for K in (5,10,20,50,100):
      n_largest = error_df.nlargest(K, 'MSE', keep='first')
      try:
          print ("Precision at K =",K,":", n_largest['True_class'].value_counts(normalize = True).loc[1]*100,"%")
      except:
          print ("Precision at K =",K,": 0 %")
    ratio = error_df[error_df.True_class==1].MSE.mean(axis = 0)/error_df[error_df.True_class==0].MSE.mean(axis = 0)
    print("MSE Ratio:", ratio)    
