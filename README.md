# Housing Dataset

# Dataset description:

  California Housing dataset with 10 attributes: longitude,	latitude,	housing_median_age,	total_rooms	,total_bedrooms,
  population, households,	median_income,	median_house_value,	ocean_proximity; and 20641 readings with varied number of records for each class of ocean_proximity.

# Methodology:
  
  Bagging and Random Forest is used for ensembling.

  ## GridSearchCV:
   Approach: It exhaustively searches through a predefined set of hyperparameter values.
   Method: It evaluates the model's performance for every possible combination of hyperparameter values specified in a grid.
   Pros: Guarantees finding the best combination if it exists within the specified grid.
   Cons: Can be computationally expensive, especially when the hyperparameter space is large.

  ## RandomizedSearchCV:
   Approach: It randomly samples hyperparameter values from a distribution (specified by the user) over a fixed number of iterations.
   Method: It evaluates the model's performance for a random subset of hyperparameter combinations.
   Pros: More efficient for large hyperparameter spaces, as it doesn't need to explore all possibilities.
   Cons: It might not find the optimal combination, but it can often discover good configurations more quickly.

# Conclusion:
  RandomizedSearchCV is a more efficient alternative to GridSearchCV, particularly in scenarios where the hyperparameter space is large, and an exhaustive search becomes impractical.    
      


# Bagging and Random Forest Regressor on California Housing Dataset
# Imports
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
    ​
    from sklearn.metrics import mean_absolute_error, confusion_matrix,\
    ConfusionMatrixDisplay, classification_report
    ​
    from sklearn.model_selection import train_test_split,\
    cross_validate, cross_val_score, ShuffleSplit, \
    RandomizedSearchCV
    ​
    from sklearn.tree import DecisionTreeRegressor
    np.random.seed(306) 
    Let's use ShuffleSplit as cv with 10 splits and 20% e.g. set aside as test examples.
    
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
 # Let's download the data and split it into training and test sets.

    # fetch dataset
    features, labels = fetch_california_housing(as_frame=True, return_X_y=True)
    labels *=100
    ​
    # train-test split
    com_train_features, test_features, com_train_labels, test_labels = \
        train_test_split(features, labels, random_state=42)
    ​
    # train --> train + dev split
    train_features, dev_features, train_labels, dev_labels = \
        train_test_split(com_train_features, com_train_labels, random_state=42)
    ​
 # Training different regressors
## Let's train different regressors:

    def train_regressor(estimator, X_train, y_train, cv, name):
        cv_results = cross_validate(estimator,
                                   X_train,
                                   y_train,
                                   cv=cv,
                                   scoring="neg_mean_absolute_error",
                                   return_train_score=True,
                                   return_estimator=True)
        
        cv_train_error = -1 * cv_results['train_score']
        cv_test_error = -1 * cv_results['test_score']
        
        print(f"On an average, {name} makes an error of "
              f"{cv_train_error.mean():.3f}k +/- {cv_train_error.std():.3f}k on the training set.")
        print(f"On an average, {name} makes an error of "
              f"{cv_test_error.mean():.3f}k +/- {cv_test_error.std():.3f}k on the test set.")
# Decision Tree Regressor
    train_regressor(
        DecisionTreeRegressor(), com_train_features,
        com_train_labels, cv, 'decision tree regressor')
        
  On an average, decision tree regressor makes an error of 0.000k +/- 0.000k on the training set.
  On an average, decision tree regressor makes an error of 47.259k +/- 1.142k on the test set.

# Bagging Regressor
    train_regressor(
        BaggingRegressor(), com_train_features,
        com_train_labels, cv, 'bagging regressor')
  On an average, bagging regressor makes an error of 14.377k +/- 0.196k on the training set.
  On an average, bagging regressor makes an error of 35.217k +/- 0.608k on the test set.
# RandomForest regressor
    train_regressor(
        RandomForestRegressor(), com_train_features,\
        com_train_labels, cv, 'random forest regressor')
  On an average, random forest regressor makes an error of 12.642k +/- 0.071k on the training set.
  On an average, random forest regressor makes an error of 33.198k +/- 0.717k on the test set.
 # Parameter search for random forest regressor
    param_distributions = {
        "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
        "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    }
    ​
    search_cv = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=2), param_distributions=param_distributions,
        scoring="neg_mean_absolute_error", n_iter=10, random_state=0, n_jobs=2,)
    ​
    search_cv.fit(com_train_features, com_train_labels)
    ​
    columns = [f"param_{name}" for name in param_distributions.keys()]
    columns += ["mean_test_error", "std_test_error"]
    cv_results = pd.DataFrame(search_cv.cv_results_)
    cv_results["mean_test_error"] = -cv_results["mean_test_score"]
    cv_results["std_test_error"] = cv_results["std_test_score"]
    cv_results[columns].sort_values(by="mean_test_error")
    param_n_estimators	param_max_leaf_nodes	mean_test_error	std_test_error
  0	500	100	40.641923	0.733708
  2	10	100	41.081103	0.921070
  7	100	50	43.872041	0.802726
  8	1	100	45.717665	1.180473
  6	50	20	49.465011	1.167198
  1	100	20	49.480914	1.021785
  9	10	20	50.056112	1.445609
  3	500	10	55.022199	1.076063
  4	5	5	61.822161	1.052154
  5	5	2	73.288226	1.257658
  
        error = -search_cv.score(test_features, test_labels)
        print(f"On average, our random forest regressor makes an error of {error:.2f} k$")
        
  On average, our random forest regressor makes an error of 40.46 k$
  
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    ​
    # Define the hyperparameter grid
    param_grid = {
        "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
        "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    }
    ​
    # Instantiate the GridSearchCV object
    grid_cv = GridSearchCV(
        RandomForestRegressor(n_jobs=2),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
    )
    ​
    # Fit the model with the training data
    grid_cv.fit(com_train_features, com_train_labels)
    ​
    # Extract and display the results
    columns = [f"param_{name}" for name in param_grid.keys()]
    columns += ["mean_test_error", "std_test_error"]
    cv_results = pd.DataFrame(grid_cv.cv_results_)
    cv_results["mean_test_error"] = -cv_results["mean_test_score"]
    cv_results["std_test_error"] = cv_results["std_test_score"]
    result_table = cv_results[columns].sort_values(by="mean_test_error")
    ​
    # Display the best parameters and corresponding mean test error
    print("Best Parameters:")
    print(grid_cv.best_params_)
    print("\nBest Mean Test Error:", -grid_cv.best_score_)
    print("\nResults Table:")
    print(result_table)
    ​
    Best Parameters:
    {'max_leaf_nodes': 100, 'n_estimators': 100}

  Best Mean Test Error: 40.55061229196913
  
  Results Table:
     param_n_estimators param_max_leaf_nodes  mean_test_error  std_test_error
  51                100                  100        40.550612        0.756845
  53                500                  100        40.610522        0.693956
  52                200                  100        40.613473        0.802481
  50                 50                  100        40.722904        0.728099
  49                 20                  100        41.035598        0.882256
  48                 10                  100        41.446081        0.773629
  47                  5                  100        41.921071        0.773260
  46                  2                  100        43.729194        0.683593
  44                500                   50        43.773841        0.832582
  41                 50                   50        43.796824        0.819423
  42                100                   50        43.834062        0.729886
  40                 20                   50        43.881667        1.109900
  43                200                   50        43.882336        0.826883
  39                 10                   50        44.414069        0.690507
  38                  5                   50        44.834047        1.034235
  37                  2                   50        46.830572        1.274405
  45                  1                  100        47.206125        1.232793
  35                500                   20        49.459799        1.083064
  31                 20                   20        49.493348        1.013789
  33                100                   20        49.513996        1.032586
  34                200                   20        49.516585        1.032485
  36                  1                   50        49.590287        1.082783
  32                 50                   20        49.606791        1.144723
  30                 10                   20        49.945249        1.098132
  29                  5                   20        50.422304        1.093535
  28                  2                   20        51.625285        1.073311
  27                  1                   20        53.554794        1.286804
  25                200                   10        55.026410        1.015200
  24                100                   10        55.028791        1.054578
  23                 50                   10        55.035263        1.060605
  26                500                   10        55.049547        1.028975
  21                 10                   10        55.232050        1.126444
  22                 20                   10        55.299195        1.114165
  20                  5                   10        55.521721        1.298371
  19                  2                   10        56.847811        1.292363
  18                  1                   10        58.161148        0.900437
  17                500                    5        61.209271        1.054021
  12                 10                    5        61.211613        1.209678
  16                200                    5        61.232972        1.026902
  13                 20                    5        61.242690        1.052412
  15                100                    5        61.256161        1.077360
  11                  5                    5        61.326887        0.998476
  14                 50                    5        61.391745        1.005879
  10                  2                    5        61.842120        0.951427
  9                   1                    5        62.560931        0.869451
  8                 500                    2        72.965381        1.046559
  5                  50                    2        73.008599        1.089444
  4                  20                    2        73.015276        1.061588
  7                 200                    2        73.024375        1.036095
  6                 100                    2        73.026407        1.026133
  2                   5                    2        73.112776        1.008032
  3                  10                    2        73.283499        1.169455
  1                   2                    2        74.227340        0.881880
  0                   1                    2        74.292386        0.884477

      
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    ​
    # Define the hyperparameter distribution for RandomizedSearchCV
    param_dist = {
        "regressor__n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
        "regressor__max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    }
    ​
    # Instantiate the RandomForestRegressor
    rf_reg = RandomForestRegressor(n_jobs=2)
    ​
    # Create a pipeline with feature selection and regression
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(rf_reg)),
        ('scaler', StandardScaler()),  # You can add other preprocessing steps here
        ('regressor', rf_reg)
    ])
    ​
    # Instantiate the RandomizedSearchCV object
    randomized_cv = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        scoring="neg_mean_absolute_error",
        n_iter=10,  # Number of parameter settings sampled
        n_jobs=2,
    )
    ​
    # Fit the model with the training data
    randomized_cv.fit(com_train_features, com_train_labels)
    ​
    # Extract and display the results
    columns = [f"param_{name}" for name in param_dist.keys()]
    columns += ["mean_test_error", "std_test_error"]
    cv_results = pd.DataFrame(randomized_cv.cv_results_)
    cv_results["mean_test_error"] = -cv_results["mean_test_score"]
    cv_results["std_test_error"] = cv_results["std_test_score"]
    result_table = cv_results[columns].sort_values(by="mean_test_error")
    ​
    # Display the best parameters and corresponding mean test error
    print("Best Parameters:")
    print(randomized_cv.best_params_)
    print("\nBest Mean Test Error:", -randomized_cv.best_score_)
    print("\nResults Table:")
    print(result_table)
    ​
    ​
    Best Parameters:
    {'regressor__n_estimators': 50, 'regressor__max_leaf_nodes': 100}

  Best Mean Test Error: 54.1448868729796

  Results Table:
    param_regressor__n_estimators param_regressor__max_leaf_nodes  \
  4                            50                             100   
  8                           200                              20   
  6                             2                              50   
  2                             2                              20   
  0                           500                              10   
  7                            50                              10   
  3                            10                              10   
  9                            50                               5   
  5                            50                               2   
  1                             5                               2   
  
   mean_test_error  std_test_error  
  4        54.144887        1.103991  
  8        54.705657        1.145223  
  6        54.929819        1.162768  
  2        55.294425        1.085424  
  0        56.933379        1.070925  
  7        57.027777        1.087156  
  3        57.054679        1.118944  
  9        61.234983        1.094016  
  5        73.075147        1.059267  
  1        73.258277        1.373636  

  
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    ​
    # Define the hyperparameter distribution for RandomizedSearchCV
    param_dist = {
        "regressor__n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
        "regressor__max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    }
    ​
    # Instantiate the RandomForestRegressor
    rf_reg = RandomForestRegressor(n_jobs=2)
    ​
    # Create a pipeline with feature selection and regression
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(rf_reg)),
        ('scaler', StandardScaler()),  # You can add other preprocessing steps here
        ('regressor', rf_reg)
    ])
    ​
    # Instantiate the RandomizedSearchCV object
    randomized_cv = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        scoring="neg_mean_absolute_error",
        n_iter=10,  # Number of parameter settings sampled
        n_jobs=2,
    )
    ​
    # Fit the model with the training data
    randomized_cv.fit(com_train_features, com_train_labels)
    ​
    # Extract and display the results
    columns = [f"param_{name}" for name in param_dist.keys()]
    columns += ["mean_test_error", "std_test_error"]
    cv_results = pd.DataFrame(randomized_cv.cv_results_)
    cv_results["mean_test_error"] = -cv_results["mean_test_score"]
    cv_results["std_test_error"] = cv_results["std_test_score"]
    result_table = cv_results[columns].sort_values(by="mean_test_error")
    ​
    # Display the best parameters and corresponding mean test error
    print("Best Parameters:")
    print(randomized_cv.best_params_)
    print("\nBest Mean Test Error:", -randomized_cv.best_score_)
    print("\nResults Table:")
    print(result_table)
    ​
    Best Parameters:
    {'regressor__n_estimators': 10, 'regressor__max_leaf_nodes': 100}

  Best Mean Test Error: 54.28954231618407

  Results Table:
    param_regressor__n_estimators param_regressor__max_leaf_nodes  \
  6                            10                             100   
  1                           100                              10   
  7                            20                              10   
  3                             1                              10   
  9                            50                               5   
  8                             2                               5   
  4                            10                               2   
  2                            50                               2   
  5                           500                               2   
  0                             5                               2   
  
  mean_test_error  std_test_error  
  6        54.289542        1.127776  
  1        56.894332        1.054914  
  7        57.062584        1.058496  
  3        58.529653        0.635195  
  9        61.270648        1.140876  
  8        62.221819        1.262790  
  4        72.941899        1.130796  
  2        72.985166        0.930381  
  5        73.005012        1.073371  
  0        73.275123        1.024800  
  ​
    
