""" A probability calculator for options
 
    The module provides basic probability calculation and visualization 
    for option operations, it's built on the open source package QSTK. 
    
    How to use this module - 
    1) As a command line tool
    $ python cal_prob.py -h
    
    2) As a package
    from cal_prod import ProbCalculator

"""
import QSTK.qstkutil.qsdateutil as du
#import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta

print "Pandas Version", pd.__version__


class ProbCalculator:
    """ Calculate the probability of an event occurring 
        based on the historical data
        
        Definition of the event -
        The price of stock S changed by $X within a time period Y
        
        @inputs: S (symbol), D (future date), P (strike price)
        @output: probability of such event occurring
    
    """
    def __init__(self, symbol):
        # input stock symbol
        self.symbol = symbol
        symbol = [symbol]
        # We need closing prices so the timestamp should be hours=16.
        dt_timeofday = dt.timedelta(hours=16)
        dt_hist_start = dt.datetime(2014, 9, 19)
        dt_hist_end = dt.datetime.today()
        # Get a list of historical ***trading*** days between the start and the end.
        ldt_timestamps = du.getNYSEdays(dt_hist_start, dt_hist_end, dt_timeofday)
        ldt_asked_len = len(ldt_timestamps)  # how many total historical trading days
        # Creating an object of the dataaccess class with Yahoo as the source.
        c_dataobj = da.DataAccess('Google')

        # Keys to be read from the data, it is good to read everything in one go.
        ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

        # TODO: we should have an independent sub-system which takes care of data fetching
        # Reading the data, now d_data is a dictionary with the keys above.
        # Timestamps and symbols are the ones that were specified before.
        ldf_data = c_dataobj.get_data(ldt_timestamps, symbol, ls_keys)
        d_data = dict(zip(ls_keys, ldf_data))

        # Filling the data for NAN
        for s_key in ls_keys:
            d_data[s_key] = d_data[s_key].fillna(method='ffill')
            d_data[s_key] = d_data[s_key].fillna(method='bfill')
            d_data[s_key] = d_data[s_key].fillna(1.0)

            # Getting the numpy ndarray of close prices.
        self.na_price = d_data['close'].values
        self.ldt_asked_len = len(self.na_price)


    def calculate(self, strike_price, future_date, lookback_months=-1):
        """ calculate the probability based on a single strike price and future date 
        """
        fd = dt.datetime.strptime(future_date, "%Y-%m-%d")
        td = dt.datetime.today()
        future_days = int((fd - td).days)


        # TODO: current price should be fetched in real time
        current_price = self.na_price[self.ldt_asked_len-1, 0]
        expected_price_change = (strike_price - current_price) / current_price
        total_price = self.na_price
        total_len = self.ldt_asked_len

        if lookback_months > 0:
            pd = td + relativedelta(months = - lookback_months)
            lookback_days = (td-pd).days
            total_price = self.na_price[self.ldt_asked_len - lookback_days - 1 : self.ldt_asked_len - 1, :]
            total_len = len(total_price)

        occurred_event = 0
        total_event = 0
        for i in range(total_len-future_days):  # loop over all historical trading days
            total_event += 1

            k = i + future_days
            if k >= total_len:
                k = total_len - 1

            delta = total_price[k, 0] - total_price[i, 0]
            actual_price_change = 0  # default price change is 0
            if self.na_price[i, 0] > 0.1:
                actual_price_change = delta / self.na_price[i, 0]

            if expected_price_change > 0:  # check price increase
                if actual_price_change >= expected_price_change:
                    occurred_event += 1
            else:    # check price drop
                if actual_price_change <= expected_price_change:
                    occurred_event += 1

        prob = float(occurred_event) / total_event

        ls_results = [prob, total_event, occurred_event]
        return ls_results



    def visualize(self, strike_price_list, future_date, lookback_months=-1):
        """ visualize the probabilities with multiple strike prices
            and same expiration day
            input date formate yyyy-mm-day
            lookback months is how many months historical data we use
        """

        future_probs = []
        for price in strike_price_list:
            [prob, total_event, occurred_event] = self.calculate(price, future_date, lookback_months = lookback_months)
            future_probs.append(prob)  # generate prob. list

        print future_probs

        # Generate the price prob. distribution graph
        # Plotting the prices with x-axis=timestamps
        future_date=dt.datetime.strptime(future_date, "%Y-%m-%d")
        title = self.symbol+" future price probability for "+future_date.strftime("%Y-%m-%d")

        plt.clf()
        plt.plot(strike_price_list, future_probs, linestyle='--', marker='o', color='b')
        plt.title(title)
        plt.grid(True)
        plt.ylabel('Touch Prob.')
        plt.xlabel('Price')
        plt.xticks(strike_price_list)
        plt.savefig(title + '_price_prob.pdf', format='pdf')
        plt.show()





def main():
    """ 
    Main function of this module 
    """
    from optparse import OptionParser
    usage = "usage: this tool is a probability calculator for options."
    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--symbol",
                      dest="symbol", default="GOOG",
                      help="symbol of the stock, e.g GOOG")
    parser.add_option("-p", "--strike_prices",
                      dest="strike_prices", default="950,960,970,980,990,1000",
                      help="comma separated strike price list, e.g 950,960,990")
    parser.add_option("-d", "--future_date",
                      dest="future_date", default="2017-8-10",
                      help="the future date associated with strike price")
    parser.add_option("-l", "--lookback_months",
                      dest="lookback_months", default="15",
                      help="how many historical data (months) for use")
    # TODO: output the results in different formats, for now just generate a plot
    # parser.add_option("-o", "--output",
    #                  dest="output", default="",
    #                  help="save the report into output")
    (options, args) = parser.parse_args()

    symbol = options.symbol
    strike_prices = options.strike_prices
    future_date = options.future_date
    lookback_months = options.lookback_months

    if not symbol or not strike_prices or not future_date:
        print "type -h to see help"
        exit(-1)

    # convert parameters to right types
    strike_prices = [float(x) for x in strike_prices.split(',')]
    future_date = str(future_date)
    lookback_months=int(lookback_months)
    # demo how to use APIs
    calculator = ProbCalculator(symbol)
    calculator.calculate(strike_prices[0], future_date, lookback_months)
    calculator.visualize(strike_prices, future_date, lookback_months)

if __name__ == "__main__":
    main()

