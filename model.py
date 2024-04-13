from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR 
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np

class HousePrice():
    def __init__(self):
        self.ridge_model = Ridge(alpha=1.0)
        self.lasso_model = Lasso(alpha=1.0)
    
    def train_ridge(self, x_train, y_train):
        self.ridge_model.fit(x_train, y_train)

    def pred_ridge(self, x):
        y_pred = self.ridge_model.predict(x)

        return y_pred
    
    def val_ridge(self, x_train, y_train_gt):
        y_train_pred = self.ridge_model.predict(x_train)
        train_mse = mean_squared_error(y_train_gt, y_train_pred)
        train_rmse = np.sqrt(train_mse)

        return train_mse, train_rmse
    
    def train_lasso(self, x_train, y_train):
        self.lasso_model.fit(x_train, y_train)
    
    def pred_lasso(self, x):
        y_pred = self.lasso_model.predict(x)

        return y_pred
    
    def val_lasso(self, x_train, y_train_gt):
        y_train_pred = self.lasso_model.predict(x_train)
        train_mse = mean_squared_error(y_train_gt, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train_gt, y_train_pred)

        return train_mse, train_rmse, train_r2



    




