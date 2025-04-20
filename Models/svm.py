from sklearn.svm import SVC # C- support Vector Classification
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
from feature_gen import generate_features

# import warnings
# from sklearn.exceptions import DataConversionWarning

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)

param_grid = {
    "C": [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8], # L2 reg, inv. reg strength. Increased C means risk of overfitting
    "kernel": ["rbf", "sigmoid", "poly"], # try sigmoid as well since we are doing binary classification
    "degree": [1, 2, 3, 5, 7]
}

model = SVC()

'''
Our kernel function's best degree for the SVM is 1 so for some reason our SVM thinks kernelization is worse?
It can't learn anything from our data at all
'''

if __name__ == "__main__":
    # arr = np.array()  # PLACEHOLDER
    X, y, feature_names = generate_features(k=10)

    y = y.ravel()
    # X, y = arr[:, :-1], arr[:, -1]

    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=5)  
    grid = GridSearchCV(model, param_grid, scoring='f1', cv = tscv, verbose=1)
    split_idx = int(0.8 * len(X))

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    grid.fit(X_train, y_train)

    print("Best SVM Params:", grid.best_params_)

    gd_best = grid.best_estimator_
    y_pred_est = gd_best.predict(X_test)
    y_pred_binary = (y_pred_est > 0.5).astype(int) 

    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_binary, digits=4))