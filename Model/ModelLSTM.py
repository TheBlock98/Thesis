"""
We want to build a LSTM model to predict future price  
(HLC3) of BTC using dataSetIMRCleaned.csv like dataSet.
First of all  we want to predict HLC3 to one step, in the second phase we want build multiOutput model to
predict HLC3 to:  1 steps, 24 steps, 48 steps, 168 steps. We implement using pyTorch.

We want use Boruta to do PCA and select important features.
""" 




