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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import advanced_activations
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


filenames = ['./dataset/original/dataset-'+str(i)+'.csv' for i in range(1,5)]
#filenames = ['./dataset/dataset-original.csv']
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
original_dataframe = pd.concat(dataframes, ignore_index=True)
original_dataframe['STD(Wa)'] = np.log(original_dataframe.loc[:, 'STD(Wa)'])
original_dataframe['STD(La)'] = np.log(original_dataframe.loc[:, 'STD(La)'])

# Augmented dataframe: x symmetry
symmetric_dataframe = original_dataframe.copy(deep=True) # deep=False will copy only references to the data
symmetric_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']] *= -1
##print(original_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']])
##print(symmetric_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']])

full_dataframe = pd.concat([original_dataframe, symmetric_dataframe], ignore_index=True)
#print(full_dataframe)

## Shifting X origin
#dataframes[0][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[0][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[0][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
#dataframes[1][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[1][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[1][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
#dataframes[2][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[2][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[2][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
#dataframes[3][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[3][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[3][['X1', 'X2', 'X3', 'X4']] + x0, 0. )

#original_dataframe[['X1', 'X2', 'X3', 'X4']] = np.where( original_dataframe[['X1', 'X2', 'X3', 'X4']] != 0, original_dataframe[['X1', 'X2', 'X3', 'X4']] + x0, 0. )
full_dataframe[['X1', 'X2', 'X3', 'X4']] = np.where( full_dataframe[['X1', 'X2', 'X3', 'X4']] != 0, full_dataframe[['X1', 'X2', 'X3', 'X4']] + x0, 0. )


#original_dataframe = pd.concat(dataframes, ignore_index=True)
#print(original_dataframe)
#print(list(original_dataframe)) # to print the [columns]

#dataframe = original_dataframe.loc[:, names[1:]].apply(pd.to_numeric)
dataframe = full_dataframe.loc[:, names[1:]].apply(pd.to_numeric)

#print(dataframe)

#dataframe = dataframe.loc[dataframe['STD(Wa)'] < 1.e-5]
#dataframe = dataframe.loc[dataframe['STD(Wa)'] > 1.e-7]
#dataframe = dataframe.loc[dataframe['Np'] > 3]
#dataframe = dataframe.loc[dataframe['Np'] == 4]



dataset = dataframe.values
print(dataset.shape)

# Split into input (X) and output (Y) variables
#['Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'MEAN(Wa)', 'STD(Wa)', 'MEAN(La)', 'STD(La)']
#  0     1     2      3      4      5     6      7    8      9    10    11    12    13    14    15    16    17      18         19         20            21 



X = dataset[:,:18]
Y = dataset[:,19]
print(X.shape)
print(Y.shape)


## Polynomial features
#polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
#X = polynomial_features.fit_transform(X)
#print(X.shape)


test_size = 0.05 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# Question: is polynomial features necessary on neural network? or it does the job of making interactions?

# Scaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
#scalername = './deep/scaler.sav' 
#dump(scaler, scalername)


# Model parameters
batch_size = int(X_train.shape[0]/1)

epochs = 300

kernel_initializer = 'normal' # 'uniform'

loss = 'mse' # 'mse' 'mape' 'mae' 'msle' 'logcosh'

activation = 'selu'
actname = 'selu'
# 'relu' 'sigmoid'
#keras.activations.elu(x, alpha=1.0)
#keras.activations.selu(x)
#keras.layers.LeakyReLU(alpha=0.3)
#keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

metrics = ['mse', 'mae', 'mape']  # 'mse' 'mae', 'mape', 'cosine' for regression; 'acc' for classification;

layers = '1'
neurons = '10x'
batchnorm = 'yes'
drop = 'yes'

# Define the model
model = Sequential()

# model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer, activation=activation))
model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation(activation))
model.add(Dropout(0.5))


model.add(Dense(10*X_train.shape[1], kernel_initializer=kernel_initializer))
model.add(BatchNormalization())
model.add(Activation(activation))
model.add(Dropout(0.5))


#model.add(Dense(Y_train.shape[1], kernel_initializer=kernel_initializer))
model.add(Dense(1, kernel_initializer=kernel_initializer))

#model.add(Dense(1, kernel_initializer=kernel_initializer))

  

learning_rate = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]
lr = 1.e-2

lrdecay = 1.e-4

loss_list = []


adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lrdecay) #, amsgrad=False)
sgd = optimizers.SGD(lr=lr, decay=lrdecay, momentum=0.9, nesterov=True)
rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=lrdecay)

optimizer = adam
optname = 'adam'
# 'adam' 'sgd'

    
    
model.compile(loss=loss, optimizer='adam', metrics=metrics)

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)  
 
 
 
loss_list.append(history.history['loss'][-1])
  
print('\n\n\nlr : ', lr, '   ', loss, ': ', loss_list[0])
 
 
casename = 'lr_'+str(lr)+'-lrdecay_'+str(lrdecay)+'-act_'+actname+'-opt_'+optname+'-hiddenlayers_'+layers+'-neurouns_'+neurons+'-batchnorm_'+batchnorm+'-dropout_'+drop+'-batchsize_'+str(batch_size)+'-epochs_'+str(epochs)+'-inputs_'+str(X.shape[1])
 
# Summarize history for loss
figure = plt.figure(figsize=(10., 8.), dpi=300)
plt.semilogy(history.history['loss'], 'k-')
plt.semilogy(history.history['val_loss'], 'k:')
plt.ylabel('Loss', font_normal, rotation='vertical')
plt.xlabel('Epoch', font_normal)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
figname = './deep/Loss_'+loss+'-'+casename+'.png'
plt.savefig(figname)
figure.clear()
plt.close(figure)


# Summarize history for mape
figure = plt.figure(figsize=(10., 8.), dpi=300)
plt.semilogy(history.history['mean_absolute_percentage_error'], 'k-')
plt.semilogy(history.history['val_mean_absolute_percentage_error'], 'k:')
plt.ylabel('MAPE', font_normal, rotation='vertical')
plt.xlabel('Epoch', font_normal)
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
figname = './deep/MAPE-'+casename+'.png'
plt.savefig(figname)
figure.clear()
plt.close(figure)

train_scores = model.evaluate(X_train, Y_train) 
print("\n%s - %s: %.6f - %s: %.6f%%" % ('Train', model.metrics_names[2], train_scores[2], model.metrics_names[3], train_scores[3]))

X_test = scaler.transform(X_test)
test_scores = model.evaluate(X_test, Y_test) 
print("\n%s - %s: %.6f - %s: %.6f%%" % ('Test', model.metrics_names[2], test_scores[2], model.metrics_names[3], test_scores[3]))
    
 
    
print('\n\n')


  #modelname = './deep/model.h5'
  #model.save(modelname, include_optimizer=False)    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
