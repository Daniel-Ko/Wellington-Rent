import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt

plots = []
filtered_csvs = []

rentdf = pd.read_csv(".\\data\\ta-geometric-mean.csv")
regiondf = pd.read_csv(".\\data\\region-geometric-mean-rents.csv")


def csv_by_TA(rentdf):
    cleaned_rdf = rentdf[rentdf['Month'].notnull()
                         & rentdf['Wellington'].notnull()
                         & rentdf['Auckland'].notnull()]

    cleaned_rdf['Month'] = pd.to_datetime(rentdf['Month'])
    # last14years_mask = (cleaned_rdf['Month'] > '2006-01-01') & (cleaned_rdf['Month'] <= '2018-08-14')

    cleaned_rdf.set_index(['Month'], inplace=True)

    ax = cleaned_rdf.plot(
        y=['Wellington', 'Auckland', 'Lower Hutt', 'Upper Hutt'])

    ax.set_xlim(pd.Timestamp('2006-01-01'), pd.Timestamp('2018-08-14'))

    plots.append(ax)

def csv_by_Region(rentdf):
    cleaned_rdf = rentdf[rentdf['Month'].notnull()
                         & rentdf['Wellington'].notnull()
                         & rentdf['Auckland'].notnull()]

    cleaned_rdf['Month'] = pd.to_datetime(rentdf['Month'])
    # last14years_mask = (cleaned_rdf['Month'] > '2006-01-01') & (cleaned_rdf['Month'] <= '2018-08-14')

    cleaned_rdf.set_index(['Month'], inplace=True)

    ax = cleaned_rdf.plot(
        y=['Wellington', 'Auckland'])

    ax.set_xlim(pd.Timestamp('2006-01-01'), pd.Timestamp('2018-08-14'))

    plots.append(ax)
    

csv_by_TA(rentdf)
csv_by_Region(regiondf)

