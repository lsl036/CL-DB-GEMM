import numpy as np
import matplotlib.pyplot as plt
import datetime

import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization

from sklearn import datasets
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from scipy.stats import uniform
from xgboost import XGBRegressor

# Load the diabetes dataset (for regression)
X, Y = datasets.load_diabetes(return_X_y=True)

# Instantiate an XGBRegressor with default hyperparameter settings
xgb = XGBRegressor()

# and compute a baseline to beat with hyperparameter optimization 
baseline = cross_val_score(xgb, X, Y, scoring='neg_mean_squared_error').mean()

print('baseline =',baseline)

# Optimization objective 优化对象
def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                XGBRegressor(learning_rate=parameters[0],
                            gamma=int(parameters[1]),
                            max_depth=int(parameters[2]),
                            n_estimators=int(parameters[3]),
                            min_child_weight = parameters[4]), 
                X, Y, scoring='neg_mean_squared_error').mean()
    score = np.array(score)
    return score

if __name__ == "__main__":
    # -------------- #
    #  Random search #
    # -------------- #
    # Hyperparameters to tune and their ranges
    param_dist = {"learning_rate": uniform(0, 1),
                "gamma": uniform(0, 5),
                "max_depth": range(1,50),
                "n_estimators": range(1,300),
                "min_child_weight": range(1,10)}

    rs = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                            scoring='neg_mean_squared_error', n_iter=25)
    
    # Run random search for 25 iterations
    rs.fit(X,Y)

    # -------------- #
    #  GPy Bayesian  #
    # -------------- #
    bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
            {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
            {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
            {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]
    
    
    optimizer = BayesianOptimization(f=cv_score, 
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True, 
                                 maximize=True)
    
    # Only 20 iterations because we have 5 initial random points
    optimizer.run_optimization(max_iter=20)

    y_rs = np.maximum.accumulate(rs.cv_results_['mean_test_score'])
    y_bo = np.maximum.accumulate(-optimizer.Y).ravel()

    print(f'Baseline neg. MSE = {baseline:.2f}')
    print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
    print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

    plt.plot(y_rs, 'ro-', label='Random search')
    plt.plot(y_bo, 'bo-', label='Bayesian optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Neg. MSE')
    plt.ylim(-5000, -3000)
    plt.title('Value of the best sampled CV score');
    plt.legend();

    plt.savefig("XGBRegressor_Bayesian")
    




