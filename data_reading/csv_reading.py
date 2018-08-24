import pandas as pd

TA_FILTERS = ['Wellington', 'Auckland', 'Lower Hutt', 'Upper Hutt',
              'Porirua', 'Kapiti Coast District', 'National Total']

REG_FILTERS = ['Wellington', 'Auckland', 'National Total']


def df_setup(csv, date_index):
    df = pd.read_csv(csv)
    df[date_index] = pd.to_datetime(df[date_index])
    df.set_index([date_index], inplace=True)
    return df


def df_clean(df, filter_cols):
    cleaned_df = df
    if filter_cols:
        cleaned_df = df.loc[:, filter_cols]
    cleaned_df.dropna(inplace=True)
    return cleaned_df


def ta_process(ta_df):
    cleaned_tadf = ta_df.rename(
        columns={'Wellington': 'WellingtonTA', 'Auckland': 'AucklandTA'})
    return cleaned_tadf


def vicsumm_process(vic_summ_df):
    return (vic_summ_df.iloc[:, 0:3],
            vic_summ_df.iloc[:, 3:6],
            vic_summ_df.iloc[:, 6:9])


def agg_by_year(monthly_df, date_index):
    yearly_df = monthly_df.groupby(pd.Grouper(freq='Y')).sum()
    yearly_df.index.rename(date_index, inplace=True)
    return yearly_df
