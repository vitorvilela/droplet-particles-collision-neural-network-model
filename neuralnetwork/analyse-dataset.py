#ToDo: 
# - Pré-script: criar pastas de output, remover >> .txt file
# - Testar: modificar origem (X' = X-0.5*H) e verificar coeficientes de correlação
#           ou verificar scater plot e ser cuidadoso nas conclusões (dependência da origem).


#import matplotlib
#matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
from matplotlib import cm as cm

import pandas as pd
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

import numpy as np



font_normal = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontsize'   : 18 }

font_italic = { 'color'      : 'k',
                'fontweight' : 'normal',
                'fontstyle'  : 'italic',
                'fontsize'   : 18 }

corr_method = 'spearman'
#pearson : standard correlation coefficient
#kendall : Kendall Tau correlation coefficient
#spearman : Spearman rank correlation
#callable : callable with input two 1d ndarrays
  
  
  
Dd = 1.e-3
H = 18*Dd
x0 = 0.5*H
  

#filenames = ['./dataset/dataset-'+str(i)+'.csv' for i in range(1,5)]
filenames = ['./dataset/dataset.csv']
#print(filenames) 



names = ['run', 'Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'AVG(Wa)', 'STD(Wa)', 'AVG(La)', 'STD(La)']
#print(names)

inputs = names[1:19]
#print(inputs)

outputs = names[19:]
#print(outputs)

dataframes = [read_csv(filename, names=names, skiprows=[0]) for filename in filenames]
#print(dataframes)

#read_csv(filename) # for files with a header
#read_csv(filename, names=names, skiprows=[1]) # for files with a header to be ignored
#read_csv(filename, names=names, header=None) # for files without a header

# Drop last n rows (when dataframe files are still in production)
original_dataframe = pd.concat(dataframes, ignore_index=True)
#dataframes = [dataframe.drop(dataframe.tail(1).index) for dataframe in dataframes] 
#print(dataframes)


# Augmented dataframe: x symmetry
symmetric_dataframe = original_dataframe.copy(deep=True) # deep=False will copy only references to the data
symmetric_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']] *= -1
#print(original_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']])
#print(symmetric_dataframe.loc[:, ['X1', 'X2', 'X3', 'X4']])


full_dataframe = pd.concat([original_dataframe, symmetric_dataframe], ignore_index=True)
#print(full_dataframe)


## Shifting X origin
#dataframes[0][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[0][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[0][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
#dataframes[1][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[1][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[1][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
#dataframes[2][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[2][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[2][['X1', 'X2', 'X3', 'X4']] + x0, 0. )
#dataframes[3][['X1', 'X2', 'X3', 'X4']] = np.where( dataframes[3][['X1', 'X2', 'X3', 'X4']] != 0, dataframes[3][['X1', 'X2', 'X3', 'X4']] + x0, 0. )

#full_dataframe[['X1', 'X2', 'X3', 'X4']] = np.where( full_dataframe[['X1', 'X2', 'X3', 'X4']] != 0, full_dataframe[['X1', 'X2', 'X3', 'X4']] + x0, 0. )
original_dataframe[['X1', 'X2', 'X3', 'X4']] = np.where( original_dataframe[['X1', 'X2', 'X3', 'X4']] != 0, original_dataframe[['X1', 'X2', 'X3', 'X4']] + x0, 0. )


#original_dataframe = pd.concat(dataframes, ignore_index=True)
#print(original_dataframe)
#print(list(original_dataframe)) # to print the [columns]

#dataframe = original_dataframe.loc[:, names[0:]].apply(pd.to_numeric)
dataframe = original_dataframe.loc[:, names[1:]].apply(pd.to_numeric)

dataframe[19:] = np.log10(original_dataframe.loc[:, names[19:]])

#dataframe = full_dataframe.loc[:, names[1:]].apply(pd.to_numeric)

#dataframe = dataframe.loc[dataframe['STD(Wa)'] < 1.e-5]
#print('\n\n', dataframe, '\n\n')
#dataframe = dataframe.loc[dataframe['STD(Wa)'] > 1.e-7]

#dataframe = dataframe.loc[dataframe['Np'] == 4]
#dataframe = dataframe.loc[dataframe['Np'] > 3]

#dataframe = full_dataframe.loc[:, names[1:]].apply(pd.to_numeric)
#print(dataframe)



# Summarize Data: descriptive statistics
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
   
  ## Confirming the dimensions of the dataframe.
  #print('\ndataframe shape\n', dataframe.shape, '\n')

  ## Look at the data types of each attribute. Are all attributes numeric?
  #print('dataframe types\n', dataframe.dtypes, '\n')

  ## Take a peek at the first 5 rows of the data. Look at the scales for the attributes.
  #print('dataframe head\n', dataframe.head(5), '\n')
  #set_option('precision', 1)

  # Summarize the distribution of each attribute (e.g. min, max, avg, std). How different are the attributes?
  print('dataframe describe\n', dataframe.describe(), '\n')
  set_option('precision', 2)

  ## Have attributes a strong correlation (e.g. > 0.70 or < -0.70) among them? and with the outputs?
  #print('dataframe correlation - ', corr_method, '\n', dataframe.corr(method=corr_method), '\n')



# Data visualizations
# Think about: 
# - Feature selection and removing the most correlated attributes
# - Normalizing the dataframe to reduce the effect of differing scales
# - Standardizing the dataframe to reduce the effects of differing distributions

for n in names[19:]:
  
  # Histograms
  # Get a sense of the data distributions (e.g. exponential, bimodal, normal).
  figure = plt.figure(figsize=(10., 8.), dpi=300)
  dataframe.loc[:,n].plot(kind='hist', bins=30, alpha=0.4, color='k',fontsize=16, legend=None)
  plt.xlabel(n, font_normal)
  plt.ylabel('Count', font_normal, rotation='vertical')
  plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  #plt.show()
  plt.tight_layout()
  plt.savefig('./dataset/results/histogram/'+n+'.png')
  figure.clear()
  plt.close(figure)
    
  ## Density
  ## Adds more evidence of the distribution. Skewed Gaussian distributions might be helpful later with transforms.
  #figure = plt.figure(figsize=(10., 8.), dpi=300)
  #dataframe.loc[:,n].plot(kind='density', color='k', fontsize=16, legend=None)
  #plt.xlabel(n, font_normal)
  #plt.ylabel('PDF', font_normal, rotation='vertical')
  #plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  #plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
  ##plt.show()
  #plt.tight_layout()
  #plt.savefig('./dataset/results/density/'+n+'.png')
  #figure.clear()
  #plt.close(figure)

  ## Box and whisker plots
  ## Boxplots summarize the distribution of each attribute, drawing a line for the median and a box around the 25th and 75th percentiles.
  ## The whiskers give an idea of the spread of the data and dots outside of the whiskers show candidate outlier values.
  ## Outliers: values that are 1.5 times greater than the size of spread of the middle 50% of the data (i.e. 75th - 25th percentile).
  #figure = plt.figure(figsize=(10., 8.), dpi=300)
  #dataframe.loc[:,n].plot(kind='box', color='k', fontsize=16, legend=None) 
  #plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
  ##plt.show()
  #plt.tight_layout()
  #plt.savefig('./dataset/results/box/'+n+'.png')
  #figure.clear()
  #plt.close(figure)


#for i in inputs: 
  #for o in outputs:
     
    ## Scatter
    ## Visualization of the interaction between variables.
    ## Higher correlated attributes show good structure in their relationship, maybe not linear, but nice predictable curved relationships.
    #figure = plt.figure(figsize=(10., 8.), dpi=300)
    #dataframe.loc[:,[i,o]].plot(kind='scatter', x=i, y=o, s=None, c='k', logy=True, fontsize=16) # logx=True, logy=True
    #plt.xlabel(i, font_normal)
    #plt.ylabel(o, font_normal, rotation='vertical')
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ##plt.show()
    #plt.tight_layout()
    #plt.savefig('./dataset/results/scatter/'+i+'-'+o+'.png')
    #figure.clear()
    #plt.close(figure)
        

#def correlation_matrix(df):
  #bar_scale = 0.5
  #figure = plt.figure(figsize=(18., 14.), dpi=150)
  #ax1 = figure.add_subplot(111)
  #cmap = cm.get_cmap('jet', lut=30) # 'Greys', 'jet', lut=30
  #cax = ax1.imshow(df.corr(method=corr_method), vmin=-bar_scale, vmax=bar_scale, interpolation="none", cmap=cmap) # vmin=-0.5, vmax=0.5, interpolation="nearest"
  ##ax1.grid(True)
  ##plt.title('Multiple Collision Correlation', fontsize=18)
  #labels = ['Re', 'We', 'Np', 'Csg', 'Cls', 'Als', 'D1', 'D2', 'D3', 'D4', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'Mwa', 'Swa', 'Mla', 'Sla'] 
  #ticks = np.arange(0, len(labels), 1)
  #ax1.set_xticks(ticks)
  #ax1.set_yticks(ticks)
  #ax1.set_xticklabels(labels, font_italic)
  #ax1.set_yticklabels(labels, font_italic)
  #cbar = figure.colorbar(cax, shrink=0.2, aspect=20, fraction=.15, pad=.03) # ticks=[-1, -0,75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
  #cbar.set_label('Spearman', size=20)
  #cbar.ax.tick_params(labelsize=16) 
  ##plt.show()
  #plt.tight_layout()
  #plt.savefig('./dataset/results/correlation/'+corr_method+'-'+str(bar_scale)+'.png')
  #figure.clear()
  #plt.close(figure)

#correlation_matrix(dataframe)





# Common Pandas Operations
# dataframe.insert(loc=5, column='UV', value=dataframe['U']*dataframe['V'])
# original_dataframe = original_dataframe.loc[original_dataframe['UU'] > 1e-10]
# original_dataframe['UU'] = numpy.log10(original_dataframe['UU'])
# dataframe = original_dataframe.loc[:, ['t', 'i', 'j', 'U', 'V', 'UV', 'TUU', 'TVV', 'TUV']]