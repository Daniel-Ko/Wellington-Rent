import argparse

import pandas as pd
import numpy as np

from data_reading import csv_reading as reader

DF_DICT = {}
comb_df = pd.DataFrame()


def create_combined_df(include_REG, include_TA):
    to_group_by_year = [
        "TA-rent",
        "REG-rent",
        "HOUSING",
        "EMPLOYMENT",
        "REAL_TWI",
        "MORTGAGE",
        "GDP",
        "CPI",
    ]

    read_into_df(
        key="TA-rent",
        path=".\\data\\ta-geometric-mean.csv",
        date_index="Month",
        filter_col=reader.TA_FILTERS,
        sdary_process=reader.ta_process,
    )

    read_into_df(
        key="REG-rent",
        path=".\\data\\region-geometric-mean-rents.csv",
        date_index="Month",
        filter_col=reader.REG_FILTERS,
    )

    read_into_df(
        key="VIC_SUMM",
        path=".\\data\\tertiary_summaries_08-16.csv",
        date_index="Year",
        filter_col=[],
        sdary_process=reader.vicsumm_process,
    )

    DF_DICT["VIC-enroll"], DF_DICT["VIC-efts"], DF_DICT["VIC-comp"] = DF_DICT[
        "VIC_SUMM"
    ]

    read_into_df(
        key="HOUSING", path=".\\data\\Housing.csv", date_index="Month", filter_col=[]
    )

    read_into_df(
        key="EMPLOYMENT",
        path=".\\data\\Employment.csv",
        date_index="Month",
        filter_col=[],
    )

    read_into_df(
        key="REAL_TWI", path=".\\data\\Real_twi.csv", date_index="Month", filter_col=[]
    )

    read_into_df(
        key="MORTGAGE", path=".\\data\\Mortgage.csv", date_index="Month", filter_col=[]
    )

    read_into_df(key="GDP", path=".\\data\\GDP.csv", date_index="Month", filter_col=[])

    read_into_df(key="CPI", path=".\\data\\CPI.csv", date_index="Month", filter_col=[])

    # Group by year
    for key in to_group_by_year:
        DF_DICT[key] = reader.agg_by_year(DF_DICT[key], "Year")

    # Merge TA+Reg
    comb_df = merge(DF_DICT["REG-rent"], DF_DICT["TA-rent"], on="Year")

    # Concat
    comb_df = pd.concat(
        [
            comb_df,
            DF_DICT["VIC-enroll"],
            DF_DICT["VIC-efts"],
            DF_DICT["VIC-comp"],
            DF_DICT["HOUSING"],
            DF_DICT["EMPLOYMENT"],
            DF_DICT["REAL_TWI"],
            DF_DICT["MORTGAGE"],
            DF_DICT["GDP"],
            DF_DICT["CPI"],
        ],
        axis="columns",
    )

    # Only keep columns with more than 5 missing values
    comb_df = comb_df.dropna(thresh=5)
    comb_df = comb_df.interpolate(limit_direction="both")

    if not include_REG:
        DF_DICT["REG-rent"] = comb_df[["Wellington", "Auckland"]]
        comb_df.drop(["Wellington", "Auckland"], 1, inplace=True)

    if not include_TA:
        DF_DICT["TA-rent"] = comb_df[reader.TA_edit_FILTERS]
        comb_df.drop(reader.TA_edit_FILTERS, 1, inplace=True)

    return comb_df


def merge(df1, df2, on):
    unique_cols = df1.columns.difference(df2.columns)
    return pd.merge(df1[unique_cols], df2, on=on, how="outer", suffixes=("_x", ""))


def read_into_df(path, key, date_index, filter_col, sdary_process=None):
    DF_DICT[key] = reader.df_clean(
        df=reader.df_setup(csv=path, date_index=date_index), filter_cols=filter_col
    )

    if sdary_process:
        DF_DICT[key] = sdary_process(DF_DICT[key])


def get_test_data():
    return DF_DICT["REG-rent"]


if __name__ == "__main__":
    create_combined_df(False, False)
