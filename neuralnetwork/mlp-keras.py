#matplotlib.use('Qt5Agg')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm as cm

import pandas as pd
from pandas import read_csv
from pandas import set_option

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras import optimizers
from keras import metrics

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from keras.models import model_from_json



font_normal = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontsize'   : 16 }

font_italic = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontstyle'  : 'italic',
                'fontsize'   : 16 }


markers = ('k-', 'k:')
#markers = ('k:', 'k--', 'k-.', 'k-')



# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)


Dd = 1.e-3
H = 18*Dd
x0 = 0.5*H


filenames = ['./dataset/dataset-'+str(i)+'.csv' for i in range(1,5)]
#print(filenames) 

names = ['run', 'Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'MEAN(Wa)', 'STD(Wa)', 'MEAN(La)', 'STD(La)']
#print(names)

#inputs = names[1:19]
#print(inputs)

#outputs = names[19:]
#print(outputs)

dataframes = [read_csv(filename, names=names, skiprows=[0]) for filename in filenames]
#print(dataframes)

#read_csv(filename) # for files with a header
#read_csv(filename, names=names, skiprows=[1]) # for files with a header to be ignored
#read_csv(filename, names=names, header=None) # for files without a header

# Drop last n rows (when dataframe files are still in production)
#dataframes = [dataframe.drop(dataframe.tail(1).index) for dataframe in dataframes] 
#print(dataframes)

## Augmented dataframe: x symmetry
#symmetric_dataframe = original_dataframe.copy(deep=True) # deep=False will copy only references to the data
#symmetric_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']] *= -1
##print(original_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']])
##print(symmetric_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']])

#full_dataframe = pd.concat([original_dataframe, symmetric_dataframe], ignore_index=True)
#print(full_dataframe)

# Shifting X origin
dataframes[0][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[0][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[0][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
dataframes[1][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[1][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[1][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
dataframes[2][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[2][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[2][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
dataframes[3][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[3][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[3][['X1', 'X2', 'X3', 'X4']] + x0, 0. )

original_dataframe = pd.concat(dataframes, ignore_index=True)
#print(original_dataframe)
#print(list(original_dataframe)) # to print the [columns]

dataframe = original_dataframe.loc[:, names[1:]].apply(pd.to_numeric)
#dataframe = full_dataframe.loc[:, names[1:]].apply(pd.to_numeric)
#print(dataframe)



dataset = dataframe.values
print(dataset.shape)

# Split into input (X) and output (Y) variables
#['Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'MEAN(Wa)', 'STD(Wa)', 'MEAN(La)', 'STD(La)']
#  0     1     2      3      4      5     6      7    8      9    10    11    12    13    14    15    16    17      18         19         20            21 
X = dataset[:,:6]
Y = dataset[:,19]
print(X.shape)
print(Y.shape)

# ToDo: Understand if MSE of Y at low values (e.g. 10-4) is also lesser, and if it is bad to convergence,
#Y = 1000*Y

test_size = 0.1 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# Question: is polynomial features necessary on neural network? or it does the job of making interactions?

# Scaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scalername = './deep/scaler.sav' 
dump(scaler, scalername)


# Model parameters
batch_size = int(X_train.shape[0]/1)

epochs = 15000

kernel_initializer = 'normal' # 'uniform'

loss = 'msle' # 'mean_squared_error' 'mape' 'mae' 'msle' 'logcosh'

learning_rate = 5.e-5

sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer = adam # default: 'adam' 'sgd'

activation = 'sigmoid' # relu sigmoid

metrics = ['mse', 'mae', 'mape', 'cosine']  # 'mse' 'mae', 'mape', 'cosine' for regression; 'acc' for classification;

# Define the model
model = Sequential()

# model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer, activation=activation))
model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
#model.add(BatchNormalization())
model.add(Activation(activation))
model.add(Dropout(0.5))

#model.add(Dense(5*X_train.shape[1], kernel_initializer=kernel_initializer))
#model.add(BatchNormalization())
#model.add(Activation(activation))
#model.add(Dropout(0.5))

#model.add(Dense(Y_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Dense(1, kernel_initializer=kernel_initializer))
#model.add(BatchNormalization())
  
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2)  
# ToDo: Save best model   


print('\n', history.history.keys(), '\n')

## Summarize history for mae
#figure = plt.figure(figsize=(10., 8.), dpi=300)
#plt.semilogy(history.history['mean_absolute_error'], markers[0])
#plt.semilogy(history.history['val_mean_absolute_error'], markers[1])
#plt.title('Model Mean Absolute Error')
#plt.ylabel('MAE', font_normal, rotation='vertical')
#plt.xlabel('Epoch', font_normal)
#plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#plt.legend(['Train', 'Validation'], loc='upper right')
##plt.show()
#plt.tight_layout()
#plt.savefig('./model/MAE.png')
#figure.clear()
#plt.close(figure)

# Summarize history for loss
figure = plt.figure(figsize=(10., 8.), dpi=300)
plt.semilogy(history.history['loss'], markers[0])
plt.semilogy(history.history['val_loss'], markers[1])
#plt.title('Model Loss')
plt.ylabel('Loss', font_normal, rotation='vertical')
plt.xlabel('Epoch', font_normal)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.legend(['Train', 'Validation'], loc='upper right')
#plt.show()
plt.tight_layout()
plt.savefig('./deep/Loss.png')
figure.clear()
plt.close(figure)




train_scores = model.evaluate(X_train, Y_train) 
print("\n%s - %s: %.6f %%" % ('Train', model.metrics_names[3], train_scores[3]))

X_test = scaler.transform(X_test)
test_scores = model.evaluate(X_test, Y_test) 
print("\n%s - %s: %.6f %%" % ('Test', model.metrics_names[3], test_scores[3]))
   
   
      
modelname = './deep/model.h5'
model.save(modelname, include_optimizer=False)    
    

    
    
#Evaluate model with standardized dataset
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(model, X, Y, cv=kfold)
#print("Model: %.2e (%.2e) AVG (VAR) of MSE" % (results.mean(), results.std()))







## Test on a single prediction sample X = [U V U*V], Y = [UU, VV, UV]
#predict_x = np.array([[0.01609, -0.00141, -0.00002264]])
#true_y = np.array([[0.00034395, 0.00000269, -0.00002686]])

#estimator = KerasRegressor(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=0)
#prediction = estimator.predict(predict_x)





## Create model
#model = Sequential()
#model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
#model.add(Dense(12, kernel_initializer='normal', activation='relu'))
#model.add(Dense(6, kernel_initializer='normal', activation='relu'))
#model.add(Dense(3, kernel_initializer='normal'))

## Compile model
#model.compile(loss='mean_squared_error', optimizer='adam')

## Validation
##kfold = KFold(n_splits=10, random_state=seed)
##results = cross_val_score(model, X, Y, cv=kfold)
##print("Model: %.2f (%.2f) AVG (STD) of RMSE" % (np.sqrt(results.mean()), np.sqrt(results.std())))

## Fit model
#model.fit(X, Y, epochs=50, batch_size=1000, verbose=0)





## Test on a single prediction sample X = [U V U*V], Y = [UU, VV, UV]
#predict_x = np.array([[0.01609, -0.00141, -0.00002264]])
#true_y = np.array([[0.00034395, 0.00000269, -0.00002686]])

## Predict
#prediction = model.predict(predict_x)

## Print
#print('predict_x')
#print(predict_x)
#print('true_y')
#print(true_y)
#print('prediction')
#print(prediction)
#print('%')
#print(100*(prediction-true_y)/true_y)


## Serialize model to JSON 
#model_json = model.to_json() 
#with open('model.json', 'w') as json_file:
  #json_file.write(model_json)

## Serialize weights to HDF5 
#model.save_weights('model.h5') 
#print('Saved model to disk')


## Load json and create model 
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read() 
#json_file.close() 
#loaded_model = model_from_json(loaded_model_json) 

## Evaluate loaded model on test data 
#loaded_model.compile(loss='mean_squared_error', optimizer='adam')

## Load weights into new model 
#loaded_model.load_weights('model.h5') 
#print('Loaded model from disk')



## Predict
#prediction = loaded_model.predict(predict_x)

## Print
#print('predict_x')
#print(predict_x)
#print('true_y')
#print(true_y)
#print('prediction')
#print(prediction)
#print('%')
#print(100*(prediction-true_y)/true_y)
