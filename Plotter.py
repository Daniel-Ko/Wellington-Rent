import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Preprocess
import DataReader

sns.set()
sns.set_style("whitegrid")
sns.set(color_codes=True)

df = DataReader.create_combined_df()
pipeline = Preprocess.process()
regressor = pipeline.named_steps["regressor"]
# print(regressor.mse_path_)

# sns.lmplot(x="Year", y=df.columns.values, data=df.values, fit_reg=True)
