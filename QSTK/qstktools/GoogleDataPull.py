'''
Pulling Google CSV Data
'''
import datetime
import os
import pandas_datareader.data as web

'''
@data_path : string for where to place the output files
@ls_symbols: list of symbols to read from yahoo
@start_date: start date
@end_date: end date
'''
def get_data(data_path, ls_symbols, start_date, end_date):

    # Create path if it doesn't exist
    if not (os.access(data_path, os.F_OK)):
        os.makedirs(data_path)

    ls_missed_syms = []
    # utils.clean_paths(data_path)   

    _now = datetime.datetime.now()
    # Counts how many symbols we could not get
    miss_ctr = 0
    for symbol in ls_symbols:
        # for index with '^' we have to use different URL
        symbol_name = symbol

        symbol_data = list()
        # print "Getting {0}".format(symbol)

        try:
            # read the stock information from Google finance
            start = start_date
            end = end_date
            if symbol[0] == '^' or "VIX" in symbol :  # we have to call Stooq reader
                df = web.DataReader(symbol_name, 'stooq')
            else:
                df = web.DataReader(symbol_name, 'google', start, end)
                # reverse the time for backward compatibility
                df = df[::-1]

            # for backward compatible we have to add column called adj. price
            df['Adj Close'] = df['Close']

            #now writing data to file
            filename = data_path + symbol_name + ".csv"

            #Writing the header
            df.to_csv(filename, sep=',')
            print "Done: fetch data for stock: {0}".format(symbol_name)

        except:
            miss_ctr += 1
            ls_missed_syms.append(symbol_name)
            print "Unable to fetch data for stock: {0}".format(symbol_name)

    print "All done. Got {0} stocks. Could not get {1}".format(len(ls_symbols) - miss_ctr, miss_ctr)
    return ls_missed_syms

'''Read a list of symbols'''
def read_symbols(s_symbols_file):
    ls_symbols = []
    ffile = open(s_symbols_file, 'r')
    for line in ffile.readlines():
        str_line = str(line)
        if str_line.strip(): 
            ls_symbols.append(str_line.strip())
    ffile.close()
    return ls_symbols

'''Main Function'''
def main():
    path = '../QSData/Google/'
    ls_symbols = read_symbols('symbols.txt')
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    get_data(path, ls_symbols, start, end)

if __name__ == '__main__':
    main()
