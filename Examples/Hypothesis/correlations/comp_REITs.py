'''
Created on 12/11/2016

@author: SpringForward
@contact: SpringForward
@summary: Generate PDF graph to compare visually a list of stocks' performance
    Input:
        - a list of stock symbols for comparisons
        - time window start and end dates for comparisons
    Output: A PDF graph
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

print "Pandas Version", pd.__version__


def gen_comparisons(ls_symbols, ls_dates):
    ''' Main Function'''

    # handling input parameters
    # Start and End date of a stock data
    ls_start = ls_dates[0]
    ls_end = ls_dates[1]

    dt_start = dt.datetime(ls_start[0], ls_start[1], ls_start[2])
    dt_end = dt.datetime(ls_end[0], ls_end[1], ls_end[2])

    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]

    # Plotting the prices with x-axis=timestamps
    stock_name = '-'.join(ls_symbols)
    start_date = '-'.join(map(str,ls_start))
    end_date   = '-'.join(map(str,ls_end))

    graph_name =  start_date +':' + \
                  end_date + ':' + \
                  stock_name + '.pdf'
    plt.clf()
    plt.plot(ldt_timestamps, na_normalized_price)
    plt.legend(ls_symbols)
    plt.ylabel('Normalized Close')
    plt.xlabel('Date')
    plt.savefig(graph_name, format='pdf')

    # # Copy the normalized prices to a new ndarry to find returns.
    # na_rets = na_normalized_price.copy()
    #
    # # Calculate the daily returns of the prices. (Inplace calculation)
    # # returnize0 works on ndarray and not dataframes.
    # tsu.returnize0(na_rets)
    #
    # # Plotting the plot of daily returns
    # plt.clf()
    # plt.plot(ldt_timestamps[0:50], na_rets[0:50, 0])  # $SPX 60 days
    # plt.plot(ldt_timestamps[0:50], na_rets[0:50, 2])  # "O" 60 days
    # plt.axhline(y=0, color='r')
    # plt.legend(['$GSPC', 'O'])
    # plt.ylabel('Daily Returns')
    # plt.xlabel('Date')
    # plt.savefig('rets_SPXvsO.pdf', format='pdf')


if __name__ == '__main__':


    # Start and End date of the charts
    # what's their performance so far in 2016?
    dt_start = [2013, 7, 1] # [year, month, day]
    dt_end = [2017, 1, 13]
    ls_dates = [dt_start, dt_end]

    # List of symbols to compare three similar REITs: O, KIM, ROIC
    ls_symbols = ["O", "$GSPC"]
    gen_comparisons(ls_symbols, ls_dates)
    print "done performance graph for {0}".format(ls_symbols)

    # compare O, ETF VNQ and health REIT HCN
    #ls_symbols = ["O", "VNQ", "HCN"]
    #gen_comparisons(ls_symbols, ls_dates)
    #print "done performance graph for {0}".format(ls_symbols)

    # compare O, VNQ and SPX
    #ls_symbols = ["O", "VNQ", "$GSPC"]
    #gen_comparisons(ls_symbols, ls_dates)
    #print "done performance graph for {0}".format(ls_symbols)

