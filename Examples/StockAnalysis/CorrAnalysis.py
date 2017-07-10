# Note:  To make matplotlib work undder virtualenv, please following this link to set the correct backend setting:
# Create a file ~/.matplotlib/matplotlibrc there and add the following code: backend: TkAgg
# Ref link: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

#Let's go ahead and start with some imports
# For division
from __future__ import division
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.linearmodels as snslin
sns.set_style('whitegrid')
#%matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]





# The tech stocks we'll use for this analysis
tech_list = ['SPY','BRK-B','NKX','PFF','VNQ','PSP', 'GOOG','HCN']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)


#******************************************************************
# Section 3: Stock analysis for all stock list
#******************************************************************
# Grab all the closing prices for the tech stock list into one DataFrame
closing_df = DataReader(tech_list,'google',start,end)['Close']

# Let's take a quick look
print(closing_df.tail())

# Make a new tech returns DataFrame for all stocks
tech_rets = closing_df.pct_change()

# Comparing Google to itself should show a perfectly linear relationship
#sns.jointplot(stock1,stock1,tech_rets,kind='scatter',color='seagreen')
#plt.show()



#******************************************************************
# Section 4: Finally correlation of all stocks
#******************************************************************
# calculate the correlation matrix for the normalized stock pct changes
corrmat =  tech_rets.corr()


##### Plot in heatmap()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

##### Only for ipython notebook?
#cmap=sns.diverging_palette(5, 250, as_cmap=True)

# Color spectrum from red to blue in 256 colors,
# which is consistent with the heatmap() color legend
#sns.palplot(sns.diverging_palette(255, 0, sep=1, n=256))

cmap=sns.diverging_palette(255, 0, sep=1, n=256, as_cmap=True)
corrmat.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
print(corrmat)




