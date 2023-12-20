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
      

  
