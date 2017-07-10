'''
Created on 12/186/2016

@author: SpringForward
@contact: SpringForward
@summary:
    Correlation study of Federal Fund Rate(FFR) and REITs performance

    Events inputs
        a stock symbol
    Output
        1) overall correlation score
        2) print out each FFR period in the format of
            FFR period, Hike/Reduce, correlation score, stock P/L performance min/max/avg.
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

print "Pandas Version", pd.__version__


def main():
    ''' Main Function'''

    # List of symbols
    ls_symbols = ["KIM"]

    # FFR HIKE/REDUCE Period

    # Start and End date of a FFR period
    dt_start = dt.datetime(2004, 6, 30)
    dt_end = dt.datetime(2006, 6, 30)
    ffr_s = 1.25
    ffr_e = 5.25


    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    ldt_len = len(ldt_timestamps)  # how many total historical trading days

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
    na_price = d_data['actual_close'].values

    print "start price: {0}, close price: {1}".format(na_price[0,0], na_price[ldt_len-1, 0])


    # performance outputs
    res_min = np.min(na_price[:,0])
    res_max = np.max(na_price[:,0])
    res_avg = np.average(na_price[:,0])
    res_var = np.var(na_price[:,0])

    total_return = ((na_price[ldt_len-1,0] - na_price[0,0])*100)/na_price[0,0]  # total return percentage

    print "the prob. of FFR is from {0} to {1}; FFR(s):{2} to FFR(e):{3}\n".format(dt_start, dt_end, ffr_s, ffr_e)
    print "The Stock {0} performance stats: total_return: {1}, min:{2}, max:{3}, avg: {4}, var:{5}".\
        format(ls_symbols[0], total_return, res_min, res_max, res_avg, res_var)


    # determine the P/L in this interest change period
    if(total_return > 5):     # solid and significant gain
        print "It's a Gain period!!"
    elif (total_return < -5):  # solid and significant loss
        print "It's a Loss period!!"
    else:
        print "Flat period!!"    # insignificant P/L


    # Study the correlation between price and interest changes
    # 1) positive correlation 2) negative correlation 3) no correlation
    # generate FFR list
    ffr_list = []
    ffr_slope = (ffr_e - ffr_s)/(ldt_timestamps[ldt_len-1]-ldt_timestamps[0]).days
    for i in range(ldt_len):
        ffr = ffr_s + ffr_slope*(ldt_timestamps[i]-ldt_timestamps[0]).days
        ffr_list.append(ffr)

    cor_res = pearsonr(ffr_list, na_price[:,0])
    cof = cor_res[0]
    pvalue=cor_res[1]

    print "correlation stats: {0}, p-value: {1}".format(cof, pvalue)

if __name__ == '__main__':
    main()
