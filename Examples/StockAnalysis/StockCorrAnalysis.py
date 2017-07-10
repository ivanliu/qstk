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



# The tech stocks we'll use for this analysis
tech_list = ['AAPL','GOOG','MSFT','AMZN']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)

print end
print start


# Define a Dictionary for stock and its stats (close, volume, etc.)
Stocks = {}
stock1 = 'AAPL'

#For loop for grabing google finance data and setting as a dataframe
for stock in tech_list:
    # Set DataFrame as the Stock Ticker
    Stocks[stock] = DataReader(stock,'google',start,end)

#******************************************************************
# Section 1: Basic stats, plots and analysis
#******************************************************************
# 1.1 Summary Stats
print("Print basic stats\n")
print(Stocks[stock1].describe())

# 1.2 General Info
print("information on this dataframe\n")
Stocks[stock1].info()

# 1.3 Let's see a historical view of the closing price
# Generating graph view and save it to a PDF file.
#plt.clf()
#(Stocks[stock1]['Close']).plot( legend=True, figsize=(10,4) )
#plt.savefig(stock1 + '-Close-Price.pdf', format='pdf')
#plt.show()

# 1.4 Now let's plot the total volume of stock being traded each day over the past 5 years
#plt.clf()
#(Stocks[stock1]['Volume']).plot( legend=True, figsize=(10,4) )
#plt.savefig(stock1 + '-Volume.pdf', format='pdf')
#plt.show()


# 1.5 Let's go ahead and plot out several moving averages
ma_day = [10,20,50]
column_names = []

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    column_names.append(column_name)
#    Stocks[stock1][column_name]=pd.rolling_mean(Stocks[stock1]['Close'],ma)
    Stocks[stock1][column_name] = Stocks[stock1]['Close'].rolling(window=ma,center=False).mean()

# updated dataframe Info
print("information on this updated dataframe\n")
Stocks[stock1].info()


#plt.clf()
#Stocks[stock1][ ['Close'] + column_names].plot(legend=True,figsize=(10,4))
#plt.savefig(stock1 + '-MA Lines.pdf', format='pdf')
#plt.show()

#******************************************************************
# Section 2: Daily Return Analysis
#******************************************************************
# 2.1 We'll use pct_change to find the percent change for each day
Stocks[stock1]['Daily Return'] = Stocks[stock1]['Close'].pct_change()

# Then we'll plot the daily return percentage
#Stocks[stock1]['Daily Return'].plot(figsize=(12,4),legend=True,linestyle='--',marker='o')
#plt.show()


# updated dataframe Info
print("information on this updated dataframe with more columns\n")
Stocks[stock1].info()

#let's get an overall look at the average daily return using a histogram.
# We'll use seaborn to create both a histogram and kde plot on the same figure.

# Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
#sns.distplot(Stocks[stock1]['Daily Return'].dropna(),bins=100,color='purple')
#plt.show()


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


# use seaborn for multiple comparison analysis
# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
#returns_fig = sns.PairGrid(tech_rets.dropna())

# Using map_upper we can specify what the upper triangle will look like.
#returns_fig.map_upper(plt.scatter,color='purple')

# We can also define the lower triangle in the figure, including the plot type (kde) or the color map (BluePurple)
#returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
#returns_fig.map_diag(plt.hist,bins=30)
#plt.show()

# We can simply call pairplot on our DataFrame for an automatic visual analysis of all the comparisons
#sns.pairplot( tech_rets.dropna() )
#plt.show()


#******************************************************************
# Section 4: Finally correlation of all stocks
#******************************************************************
snslin.corrplot(tech_rets.dropna(),annot=True)
plt.show()


#******************************************************************
# Section 5: Risk Analysis
#******************************************************************
# Let's start by defining a new DataFrame as a clenaed version of the oriignal tech_rets DataFrame
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)

# Set the x and y limits of the plot (optional, remove this if you don't see anything in your plot)
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])

#Set the plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

# Label the scatter plots, for more info on how this is done, chekc out the link below
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
plt.show()


#******************************************************************
# Section 6: Value at risk
#******************************************************************

# 6.1 Value at risk using the "bootstrap" method
# Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
sns.distplot(Stocks[stock1]['Daily Return'].dropna(),bins=100,color='purple')
plt.show()

# The 0.05 empirical quantile of daily returns
# The 0.05 empirical quantile of daily returns is at -0.015.
# That means that with 95% confidence, our worst daily loss will not exceed 1.5%.
# If we have a 1 million dollar investment, our one-day 5% VaR is 0.015 * 1,000,000 = $15,000.
print( rets[stock1].quantile(0.05) )

#6.2 Value at Risk using the Monte Carlo method based on Geometric Browninan Motion (GBM)

# Set up our time horizon
days = 365

# Now our delta
dt = 1/days

# Now let's grab our mu (drift) from the expected return data we got for VNQ
mu = rets.mean()[stock1]

# Now let's grab the volatility of the stock from the std() of the average return
sigma = rets.std()[stock1]


def stock_monte_carlo(start_price, days, mu, sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''

    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)

    # Run price array for number of days
    for x in xrange(1, days):

        # Calculate Shock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))

        # Calculate Drift
        drift[x] = mu * dt

        # Calculate Price
        price[x] = price[x - 1] + (price[x - 1] * (drift[x] + shock[x]))

    return price

# Get start price from AAPL.head()
start_price = 145.42  # apple price on 6/12/2017

for run in xrange(100):  # do simulation 100 times
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title('Monte Carlo Analysis for APPLE')
plt.show()

# Let's go ahead and get a histogram of the end results for a much larger run.
# (note: This could take a little while to run , depending on the number of runs chosen)

# Set a large numebr of runs
runs = 10000

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in xrange(runs):
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];



#Now that we have our array of simulations, we can go ahead and plot a histogram ,
# as well as use qunatile to define our risk for this stock.
# Now we'lll define q as the 1% empirical qunatile, this basically means that
# 99% of the values should fall between here

q = np.percentile(simulations, 1)

# Now let's plot the distribution of the end prices
plt.hist(simulations, bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" % start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for " + stock1 + " Stock after %s days" % days, weight='bold')
plt.show()
