import pandas as pd
import numpy as np

from data_reading import csv_reading as reader

DF_DICT = {}


def main():

    DF_DICT['TA-rent'] = reader.ta_process(
        reader.df_clean(
            df=reader.df_setup(
                csv=".\\data\\ta-geometric-mean.csv",
                date_index='Month'
            ),
            filter_cols=reader.TA_FILTERS
        ))

    DF_DICT['REG-rent'] = reader.df_clean(
        df=reader.df_setup(
            csv=".\\data\\region-geometric-mean-rents.csv",
            date_index='Month'),
        filter_cols=reader.REG_FILTERS
    )

    DF_DICT['VIC-enroll'], DF_DICT['VIC-efts'], DF_DICT['VIC-comp'] = reader.vicsumm_process(
        reader.df_setup(
            csv=".\\data\\tertiary_summaries_08-16.csv",
            date_index='Year'
        )
    )

    DF_DICT['HOUSING'] = reader.df_clean(
        df=reader.df_setup(
            csv=".\\data\\region-geometric-mean-rents.csv",
            date_index='Month'),
        filter_cols=[]
    )

    # Group by year
    DF_DICT['REG-rent'] = reader.yearsum(DF_DICT['REG-rent'], 'Year')
    DF_DICT['TA-rent'] = reader.yearsum(DF_DICT['TA-rent'], 'Year')

    # Merge TA+Reg
    comb_df = merge(DF_DICT['REG-rent'], DF_DICT['TA-rent'], on='Year')

    print(comb_df)

    # Concat
    comb_df = pd.concat(
        [comb_df, DF_DICT['VIC-enroll'], DF_DICT['VIC-efts'], DF_DICT['VIC-comp'], DF_DICT['HOUSING']], axis='columns')

    # print(comb_df)


def merge(df1, df2, on):
    unique_cols = df1.columns.difference(df2.columns)
    return pd.merge(df1[unique_cols], df2, on=on, how='outer', suffixes=('_x', ''))


if __name__ == "__main__":
    main()
