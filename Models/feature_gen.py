# import pandas as pd
# import numpy as np
# import os
# import ta
# import sklearn.feature_selection
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.feature_selection import VarianceThreshold
# # import warnings

# # warnings.filterwarnings("ignore", category=RuntimeWarning)

# os.getcwd()
# os.listdir(".")

# df = pd.read_csv("./Models/Data/data.csv", index_col="timestamp")
# df.drop("otc", inplace=True,  axis=1)
# df.head()
# df.values[0]

# def augment_with_ta(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df = ta.add_all_ta_features(
#         df,
#         open="open",
#         high="high",
#         low="low",
#         close="close",
#         volume="volume",
#         fillna=True
#     )
#     return df


# df = augment_with_ta(df)


# # For feature selection, generate y labels
# # 
# # For $X_t$, we want to predict $\mathbb I (X_{t+1} - X_t)$, where $\mathbb I (\cdot)$ is Indicator function/Heaviside step function

# close = df["close"]
# close = close.values.flatten()
# close.shape

# y = np.array([close[i + 1] - close[i] for i in range(len(close)-1)])
# y = np.where(y < 0, False, True)

# # y = np.append(y, np.nan)
# df = df.iloc[:-1]
# df["Y"] = y

# raw_data = df.values[:-1] # have to drop last row
# y = raw_data[:,-1]
# X = raw_data[:,:-1]

# selector = VarianceThreshold(threshold=1e-8)
# X_cleaned = selector.fit_transform(X)

# X_cleaned.shape, y.shape

# X_new = SelectKBest(f_classif, k=10).fit_transform(X_cleaned,y)
# print(X_new.shape)

# print(X_new[5])

# # # Alt ways

# # # need to split on y as well here but another method for getting column names
# # # Create and fit selector
# # selector = SelectKBest(f_classif, k=13)
# # selector.fit(df.iloc[:-1], y)
# # # Get columns to keep and create new dataframe with those only
# # cols_idxs = selector.get_support(indices=True)
# # features_df_new = df.iloc[:,cols_idxs]

# # print(features_df_new.columns)

# feature_gen.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import ta

def generate_features(csv_path="./Models/Data/data.csv", k=10):
    df = pd.read_csv(csv_path, index_col="timestamp")
    df.drop("otc", inplace=True, axis=1)

    df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

    # Create binary labels
    close = df["close"].values.flatten()
    y = np.where(np.diff(close) < 0, 0.0, 1.0)
    df = df.iloc[:-1]  # drop last row to align with y
    df["Y"] = y

    # Prepare features and labels
    raw_data = df.values[:-1]  # drop last row to avoid label mismatch
    y = raw_data[:, -1]
    X = raw_data[:, :-1]

    # Variance thresholding
    vt = VarianceThreshold(threshold=1e-8)
    X_cleaned = vt.fit_transform(X)
    retained_cols = df.drop(columns=["Y"]).columns[vt.get_support()]

    # Feature selection
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X_cleaned, y)
    selected_cols = retained_cols[selector.get_support()]

    return X_new.astype(np.float32), y.astype(np.float32).reshape(-1, 1), selected_cols.tolist()

if __name__ == "__main__":
    X_new, y, selected_features = generate_features(k=10)

    print("✅ Feature generation successful!")
    print("X_new shape:", X_new.shape)
    print("y shape:", y.shape)
    print("Selected features:", selected_features)

    # Optional: inspect a sample row
    print("X_new[5]:", X_new[5])




