import numpy as np
from numpy import arange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import pandas as pd
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load

font_normal = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontsize'   : 16 }

font_italic = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontstyle'  : 'italic',
                'fontsize'   : 16 }

font_italic_labels = { 'color'      : 'k',
                       'fontweight' : 'normal',
                       'fontstyle'  : 'italic',
                       'fontsize'   : 12 }


markers = ('k-', 'k:')
#markers = ('k:', 'k--', 'k-.', 'k-')


  
Dd = 1.e-3
H = 18*Dd
x0 = 0.5*H

EPS = 1.e-7


# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)



filenames = ['./dataset/dataset-'+str(i)+'.csv' for i in range(1,5)]
#print(filenames) 

names = ['run', 'Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'AVG(Wa)', 'STD(Wa)', 'AVG(La)', 'STD(La)']
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
dataframe = dataframe.loc[dataframe['Np'] == 4]


dataset = dataframe.values
#print(dataset.shape)

# Split into input (X) and output (Y) variables
#  0     1      2     3      4      5     6     7     8      9    10    11    12    13    14    15    16    17     18          19          20         21
#['Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'AVG(Wa)', 'STD(Wa)', 'AVG(La)', 'STD(La)']
X = dataset[:,0:18] 
inputs = 'SET2'
Y = dataset[:,19]
output = 'STD(Wa)'
print(X.shape)
print(Y.shape)

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)





## Model analysis

models = [DummyRegressor(), LinearRegression(), Lasso(), ElasticNet(), DecisionTreeRegressor(), ExtraTreesRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]
#models = [DecisionTreeRegressor(), ExtraTreesRegressor(), RandomForestRegressor()]
X_validation = X_test
Y_validation = Y_test

#print(inputs)
#print(output)

degrees = (1,)
for degree in degrees:
  
  print('\nDegree: ', degree)

  polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

  train_input = polynomial_features.fit_transform(X_train)
  scaler = StandardScaler().fit(train_input)
  rescaledX = scaler.transform(train_input)
  
  #scaler_filename = './machine/multiple_' + str(degree) + '_scaler.sav'
  #dump(scaler, scaler_filename, protocol=3)

  for model in models:
  
    name = repr(model).split('(')[0]   
    model.fit(rescaledX, Y_train)

    val_input = polynomial_features.fit_transform(X_validation)
    rescaledValidationX = scaler.transform(val_input)

    predictions = model.predict(rescaledValidationX)
    #print(predictions)
    
    mae = mean_absolute_error(Y_validation, predictions)
    mape = np.abs(100*(predictions-Y_validation)/(Y_validation+EPS))
    mse = mean_squared_error(Y_validation, predictions)
    r2 = r2_score(Y_validation, predictions)
                               
    print('\n' + name)
    print('\nMAE [-]:')
    print(mae.mean())
    print('\nMAPE [%]:')
    print(mape.mean())
    print('\nMSE [-]:')
    print(mse)
    print('\nR2 [-]:')
    print(r2)    












# Evaluate Algorithms with cross-validation

# Test options and evaluation metric
# 10-fold cross-validation is a good standard test when dataset is not too small (e.g. 500)
num_folds = 10
print('\nCross validation')
print('num_folds ' + str(num_folds))

# 'neg_mean_squared_error' MSE will give a gross idea of how wrong all predictions are (0 is perfect)
# others: 'r2', 'explained_variance'
#scores = [('R2', 'r2'), ('MSE', 'neg_mean_squared_error'), ('MAE', 'neg_mean_absolute_error')]
scores = [('MSE', 'neg_mean_squared_error')]



degrees = (1,)

print('\nStandardize and Using Polynomial Features') 

for degree in degrees:
    
  print('\nDegree %i' % degree)
           
  polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

  pipelines = []
  pipelines.append(('DUMMY', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('Dummy', DummyRegressor())])))
  pipelines.append(('LR', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('LR', LinearRegression())])))
  pipelines.append(('LASSO', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('LASSO', Lasso())])))
  pipelines.append(('EN', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('EN', ElasticNet())])))
  #pipelines.append(('KNN', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
  pipelines.append(('CART', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
  #pipelines.append(('SVR', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('SVR', SVR())])))
  
  # Ensemble
  pipelines.append(('RF', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
  pipelines.append(('ET', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))
  pipelines.append(('AB', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
  pipelines.append(('GBM', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
  
  kfold = KFold(n_splits=num_folds, random_state=seed)

  for score, scoring in scores:
      
    print('\nScore %s' % score)
    
    results = []
    names = []
      
    for name, model in pipelines:     
          
      cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
      results.append(cv_results)                                      
      names.append(name)
            
      #if score == 'MSE':
      #msg = "%s: %e (%e)" % (name+' MSE', np.fabs(cv_results.mean()), np.fabs(cv_results.std()))
      #else:
      msg = "%s: %e (%e)" % (name, cv_results.mean(), cv_results.std())
      print(msg)           
            
            
    figure = plt.figure(figsize=(7., 8.), dpi=300)
    ax = figure.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, font_italic_labels, rotation='vertical')
    plt.ylabel(score, font_italic, rotation='vertical')
    plt.savefig('./machine/'+inputs+'-'+output+'-degree-'+str(degree)+'-'+score+'.png')
    figure.clear()
    plt.close(figure)
            






#print('\nEnsembles and Using Polynomial Features') 

#for degree in degrees:
    
  #print('\nDegree %i' % degree)
           
  #polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

  #pipelines = []
  ## Boosting
  ## Building multiple models, typically of the same type, each of which learns to fix the prediction errors of a prior model in the sequence of models.
  ## The models make predictions which may be weighted by their demonstrated accuracy and the results are combined to create a final output prediction.
  #pipelines.append(('AB', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
  #pipelines.append(('GBM', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
  ## Bagging 
  ## Building multiple models, typically of the same type, from different subsamples with replacement of the training dataset.
  ## The final output prediction is averaged across the predictions of all of the sub-models.
  #pipelines.append(('RF', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
  #pipelines.append(('ET', Pipeline([("polynomial_features", polynomial_features), ('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))

 
  #kfold = KFold(n_splits=num_folds, random_state=seed)

  #for score, scoring in scores:
      
    #print('\nScore %s' % score)
    
    #results = []
    #names = []
      
    #for name, model in pipelines:     
          
      #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
      #results.append(cv_results)                                      
      #names.append(name)
            
      #if score == 'MSE':
        #msg = "%s: %e (%e)" % (name+' RMSE', np.sqrt(np.fabs(cv_results.mean())), np.sqrt(np.fabs(cv_results.std())))
      #else:
        #msg = "%s: %e (%e)" % (name, cv_results.mean(), cv_results.std())
      #print(msg)           
            
            
    #figure = plt.figure(figsize=(7., 7.), dpi=300)
    #ax = figure.add_subplot(111)
    #plt.boxplot(results)
    #ax.set_xticklabels(names, font_italic, rotation='vertical')
    #plt.ylabel(score, font_italic, rotation='vertical')
    #plt.savefig('ensemble-degree-'+str(degree)+'-'+score+'.png')
    #figure.clear()
    #plt.close(figure)





## Chosen model

#polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
#train_input = polynomial_features.fit_transform(X_train)

#scaler = StandardScaler().fit(train_input)
#rescaledX = scaler.transform(train_input)
  
##scaler_filename = 'amirfazli_degree3_scaler.sav'
##dump(scaler, scaler_filename, protocol=3)

#model = LinearRegression()
#print('\n'+repr(model).split('(')[0])
#reg = model.fit(rescaledX, Y_train)
#print('\nreg.intercept_')
#print(reg.intercept_)
#m = dict(zip(polynomial_features.get_feature_names(), reg.coef_))
#print('\nmodel dict')
#print(m)



#model_filename = 'linear_degree3_model.sav'
#dump(model, model_filename, protocol=3)


## Load model from disk
#loaded_model = load(model_filename) 
## Load scaler from disk
#scaler = load(scaler_filename)

#print('\nscaler mean: ', scaler.mean_)
#print('\nscaler std: ', scaler.scale_))

## Transform the validation dataset
#val_input = polynomial_features.fit_transform(X_validation)
#rescaledValidationX = scaler.transform(val_input)
#predictions = model.predict(rescaledValidationX)

#error = np.abs(100*(predictions-Y_validation)/Y_validation)
#rmse = np.sqrt(mean_squared_error(Y_validation, predictions))
#r2 = r2_score(Y_validation, predictions)
                  
#print('\nRelative Error [%]:')
#print(error.mean())
#print('\nRMSE [-]:')
#print(rmse)
#print('\nR2 [-]:')
#print(r2)    




### Pre-defined data range

#filename = 'data-simplified.csv'

## Header
#names = ['replicate', 'reynolds', 'weber', 't*', 'A*']
#dataset = read_csv(filename, names=names, delim_whitespace=False)

## Write data summary
#print( '\nDataset shape\n', dataset.shape, '\n')
#print( 'Dataset describe\n', dataset.describe(), '\n')
#set_option('precision', 2)

#array = dataset.values

## Train: replicates 0, 1 and 2 of factorial design data
#X_train = array[:60,1:4] 
#Y_train = array[:60,4]

#print('\nValidation')
#X_validation = array[60:,1:4] 
#Y_validation = array[60:,4]
#print(Y_validation)









## Chosen model

#polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
#train_input = polynomial_features.fit_transform(X_train)

#scaler = StandardScaler().fit(train_input)
#rescaledX = scaler.transform(train_input)
  
##scaler_filename = 'amirfazli_degree2_scaler.sav'
##dump(scaler, scaler_filename, protocol=3)

#model = LinearRegression()
#print('\n'+repr(model).split('(')[0])
#reg = model.fit(rescaledX, Y_train)
#print('\nreg.intercept_')
#print(reg.intercept_)
#m = dict(zip(polynomial_features.get_feature_names(), reg.coef_))
#print('\nmodel dict')
#print(m)

##model_filename = 'linear_degree2_model.sav'
##dump(model, model_filename, protocol=3)