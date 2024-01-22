#!/usr/bin/env python
# coding: utf-8

# # Stationarity check 

# We control the stationarity of dataSet in order to be sure that ML theory is applicabile in simple way otherwise we have to use online machine learning tecnique 

# In[3]:



import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

pio.renderers.default = 'notebook'
dataSet=pd.read_csv("FeatureEngineering&DATA/dataSetIMRCleaned.csv")


fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(20, 30))
fig.suptitle('Distribution probability of features', y=1.02)

""""


for i, col in enumerate(dataSet.iloc[:, 1:]):
  sns.histplot(dataSet[col], ax=axs[i // 6, i % 6])
  axs[i // 6, i % 6].set_title(col)

plt.tight_layout()
plt.show()




# In[7]:


fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(20, 30))
fig.suptitle('Distribution probability of features', y=1.02)



for i, col in enumerate(dataSet.iloc[:, 1:]):
    sns.lineplot(x=dataSet.index, y=dataSet[col], ax=axs[i // 6, i % 6])
    axs[i // 6, i % 6].set_title(col)

plt.tight_layout()
plt.show()


# In[8]:


# probability distribution of target
plt.figure(figsize=(8, 6))
sns.histplot(dataSet['hlc3'], kde=True, color='skyblue')
plt.title('Distribuzione di hlc3')
plt.xlabel('hlc3')
plt.ylabel('Frequenza')
plt.show()


# now we comput the AdFuller test

# In[25]:


from statsmodels.tsa.stattools import adfuller
import colored
import numpy as np


dataSet.replace([np.inf, -np.inf], np.nan, inplace=True)


dataSet.fillna(dataSet.mean(), inplace=True)


threshold = 0.05

for col in dataSet.iloc[:, 1:]:
    result = adfuller(dataSet[col])
    print(f'Colonna: {col}')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    # threshold checker 
    if result[1] <= threshold:
        print(colored.stylize('Stazionario (p-value <= {:.2f})'.format(threshold), colored.fg('green')))
    else:
        print(colored.stylize('Non stazionario (p-value > {:.2f})'.format(threshold), colored.fg('red')))
    
    print('-' * 40)
# so the dataSet not is stationarity we have to use a no stationarity resiliant model like RNN or try to transform our dataSet in stationarity
# but there will be some problems
"""

# Normalize data

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# Assuming 'dataSet' is a pandas DataFrame that has already been defined.
dataSet.info()
dataSet.replace([np.inf, -np.inf], 0, inplace=True)

# Create a MinMaxScaler and normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(dataSet)

# Save the scaler
joblib.dump(scaler, 'FeatureEngineering&DATA/scaler.pkl')

# Convert normalized_data back to a pandas DataFrame
normalized_df = pd.DataFrame(normalized_data, index=dataSet.index, columns=dataSet.columns)

# Save the normalized DataFrame to a CSV file
normalized_df.to_csv("FeatureEngineering&DATA/dataSetIMRNormalized.csv", index=False)


# Count the number of zeroes that are not NaN in the EOM column
zero_count = (normalized_df["EOM"] == 0).sum()

# Confirm there are no NaN values by counting them
nan_count = normalized_df["EOM"].isnull().sum()

print(f"Number of zeroes (not NaN) in 'EOM' column: {zero_count}")
print(f"Number of NaN values in 'EOM' column: {nan_count}")
