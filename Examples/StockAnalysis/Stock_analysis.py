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


class stock_analysis:
    def __init__(self, stock_list):
        self.stock_list=stock_list
        # Set up End and Start times for data grab
        end = datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        Stocks = {}
        for stock in stock_list:
            # Set DataFrame as the Stock Ticker
            Stocks[stock] = DataReader(stock, 'google', start, end)
        self.Stocks=Stocks
        closing_df = DataReader(stock_list, 'google', start, end)['Close']
        self.closing_df=closing_df
        self.stock_rets = closing_df.pct_change()
    #Summary Stats
    def Summary_stats(self,index):
        return self.Stocks[self.stock_list[index]].describe()

    #correlation analuysis
    def cor_analysis(self):
        snslin.corrplot(self.stock_rets.dropna(), annot=True)
        plt.show()

    # risk analysis
    def risk_analsisy(self):
        rets = self.stock_rets.dropna()

        area = np.pi * 20

        plt.scatter(rets.mean(), rets.std(), alpha=0.5, s=area)

        # Set the plot axis titles
        plt.xlabel('Expected returns')
        plt.ylabel('Risk')

        # Label the scatter plots, for more info on how this is done, chekc out the link below
        # http://matplotlib.org/users/annotations_guide.html
        for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
            plt.annotate(
                label,
                xy=(x, y), xytext=(50, 50),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3'))
        plt.show()

    # var using bootstrap
    def var_bootstrap(self,index):
        stock1 =self.stock_list[index]
        self.Stocks[stock1]['Daily Return'] =self.Stocks[stock1]['Close'].pct_change()
        # Note the use of dropna() here, otherwise the NaN values can't be read by seaborn
        rets = self.stock_rets.dropna()
        sns.distplot(self.Stocks[stock1]['Daily Return'].dropna(), bins=100, color='purple')
        plt.show()
        # The 0.05 empirical quantile of daily returns
        # The 0.05 empirical quantile of daily returns is at -0.015.
        # That means that with 95% confidence, our worst daily loss will not exceed 1.5%.
        # If we have a 1 million dollar investment, our one-day 5% VaR is 0.015 * 1,000,000 = $15,000.
        print(rets[stock1].quantile(0.05))

    ## Value at Risk using the Monte Carlo method based on Geometric Browninan Motion (GBM)
    def var_MC(self,start_price,days,index):
        dt=1/days
        rets = self.stock_rets.dropna()
        stock1=self.stock_list[index]

        # Now let's grab our mu (drift) from the expected return data we got for VNQ
        mu = rets.mean()[stock1]

        # Now let's grab the volatility of the stock from the std() of the average return
        sigma = rets.std()[stock1]

        for run in xrange(100):  # do simulation 100 times
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
            plt.plot(price)
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.title('Monte Carlo Analysis for'+' '+stock1)
        plt.show()

        # Set a large numebr of runs
        runs = 10000

        # Create an empty matrix to hold the end price data
        simulations = np.zeros(runs)

        # Set the print options of numpy to only display 0-5 points from an array to suppress output
        np.set_printoptions(threshold=5)

        for run in xrange(runs):
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
            # Set the simulation data point as the last stock price for that run
            simulations[run] = price[days - 1];

        # Now that we have our array of simulations, we can go ahead and plot a histogram ,
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



def main():
    tech_list = ['AAPL','GOOG','MSFT','AMZN']
    example=stock_analysis(tech_list)
    #print(example.Summary_stats(1))
    #example.cor_analysis()
    #example.risk_analsisy()
    #example.var_bootstrap(0)
    # Set up our time horizon
    days = 365
    start_price = 145.42  # apple price on 6/12/2017
    example.var_MC(start_price,days,1)
if __name__ == "__main__":
    main()