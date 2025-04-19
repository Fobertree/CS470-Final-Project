# %%
import pandas as pd
import numpy as np
import os
import ta
import sklearn.feature_selection

# %%
os.getcwd()

# %%
os.listdir(".")

# %%
df = pd.read_csv("./Models/Data/data.csv", index_col="timestamp")
df.drop("otc", inplace=True,  axis=1)
df.head()

# %%
df.values[0]

# %%
def augment_with_ta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True
    )
    return df

# %%
df = augment_with_ta(df)

# %%
df

# %%
df.columns

# %% [markdown]
# For feature selection, generate y labels
# 
# For $X_t$, we want to predict $\mathbb I (X_{t+1} - X_t)$, where $\mathbb I (\cdot)$ is Indicator function/Heaviside step function

# %%
close = df["close"]
close = close.values.flatten()
close.shape

# %%
y = np.array([close[i + 1] - close[i] for i in range(len(close)-1)])

# %%
y = np.where(y < 0, False, True)

# %%
np.append(y, np.nan);

# %%
df["Y"] = y

# %%
df.head()

# %%
from sklearn.feature_selection import SelectKBest, f_classif

# %%
raw_data = df.values[:-1] # have to drop last row

# %%
y = raw_data[:,-1]

# %%
X = raw_data[:,:-1]

# %%
X.shape, y.shape

# %%
X_new = SelectKBest(f_classif, k=10).fit_transform(X,y)
X_new.shape

# %%
X_new[5]

# %% [markdown]
# Alt ways

# %%
# Create and fit selector
selector = SelectKBest(f_classif, k=13)
selector.fit(df.iloc[:-1], y)
# Get columns to keep and create new dataframe with those only
cols_idxs = selector.get_support(indices=True)
features_df_new = df.iloc[:,cols_idxs]

# %%
features_df_new.columns

# %%



