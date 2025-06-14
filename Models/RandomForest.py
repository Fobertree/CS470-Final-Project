from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
from feature_gen import generate_features

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}

rf = RandomForestClassifier()


if __name__ == "__main__":
    # arr = np.array()  # PLACEHOLDER
    X, y, feature_names = generate_features(k=10)
    # X, y = arr[:, :-1], arr[:, -1]

    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=5)  
    grid = GridSearchCV(rf, param_grid, scoring='f1', cv = tscv)
    split_idx = int(0.8 * len(X))

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    grid.fit(X_train, y_train)

    print("Best RF Params:", grid.best_params_)

    gd_best = grid.best_estimator_
    y_pred_est = gd_best.predict(X_test)
    y_pred_binary = (y_pred_est > 0.5).astype(int) 

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_binary, digits=4))