# Part 3 — Advanced Modeling with Ensembles and Tuning (Bike Sharing Dataset)

## Decision Trees: Unconstrained vs Controlled

**Unconstrained tree (no limits):**

* Train accuracy: **1.0**
* Test accuracy: **0.912**

The unconstrained decision tree achieved perfect accuracy on the training data, but its test accuracy is lower. This difference clearly shows **overfitting**. The model has learned the training data almost perfectly, including patterns that are actually just noise, so it does not perform as well on unseen data.

Decision trees are considered **high-variance models** because they choose the best split at each node without considering whether earlier decisions were globally optimal. Every split is made based on the current node only, which makes deep trees very likely to memorize the training data.

**Controlled tree (max_depth=5, min_samples_split=20):**

* Train accuracy: **0.914**
* Test accuracy: **0.871**

Adding restrictions such as **max_depth** and **min_samples_split** keeps the tree from becoming too complex. As expected, the training accuracy decreases slightly because the tree cannot learn every detail in the training data. The gap between training and testing performance is also smaller, meaning the model is less likely to overfit.

In this dataset, however, the unconstrained tree still performs slightly better on the test set, suggesting that allowing the tree to capture more detailed patterns was beneficial despite the increased risk of overfitting.

---

## Gini vs Entropy

**Gini impurity formula:**

**1 − Σ pi²**

**Entropy formula:**

**−Σ pi log₂(pi)**

A Gini impurity of **0** means that every sample in the node belongs to the same class, making it completely pure.

Both Gini and Entropy are measures of node impurity. Gini measures impurity using squared class probabilities, while Entropy uses logarithms. In practice, both usually produce very similar decision trees.

* Gini test accuracy: **0.9048**
* Entropy test accuracy: **0.9048**

Since both methods achieved exactly the same accuracy on this dataset, there is no noticeable difference in performance between them.

---

## Random Forest

* Train accuracy: **0.9897**
* Test accuracy: **0.9048**
* AUC: **0.9717**

### Top 5 features by importance

* atemp: **0.1957**
* yr: **0.1772**
* temp: **0.1715**
* hum: **0.0860**
* windspeed: **0.0742**

Random Forest calculates feature importance based on how much each feature reduces impurity across all trees in the forest. Features that consistently create better splits receive higher importance scores.

This is different from coefficients in Linear Regression. A regression coefficient explains how the target changes when a feature increases by one unit, while feature importance only indicates how useful that feature is for making splits. It does not indicate whether the relationship is positive or negative.

### Bagging Concept

Random Forest uses **bagging**, where every tree is trained on a different bootstrap sample of the training data. At each split, only a random subset of features is considered instead of all available features.

Because every tree is slightly different, their prediction errors are also different. Averaging the predictions from many trees reduces the overall variance of the model and generally leads to better generalization than using a single decision tree.

---

## Gradient Boosting

* Test accuracy: **0.9184**
* AUC: **0.9776**

Gradient Boosting builds trees one after another instead of independently. Each new tree tries to correct the mistakes made by the previous trees, gradually improving the model.

The **learning_rate** controls how much each new tree contributes to the final prediction. A smaller learning rate makes learning slower but usually reduces the risk of overfitting.

The higher AUC (**0.9776**) compared to Random Forest (**0.9717**) shows that Gradient Boosting performs slightly better at distinguishing between high-demand and low-demand bike rental days.

---

## Feature Ablation Study

Lowest-importance features identified:

* holiday

* season_label_Fall

* workingday

* season_label_Summer

* season_label_Spring

* Full model AUC: **0.9717**

* Reduced model AUC: **0.9745**

Interestingly, removing these five low-importance features slightly improved the model's AUC. This suggests that these features were contributing very little useful information and may have been adding unnecessary noise.

Using fewer features also makes the model easier to maintain in production because less data needs to be collected and processed. In this case, reducing the number of features improved performance without any noticeable downside.

---

## Cross-validated comparison (5-fold AUC)

| Model               | Mean AUC | Std AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.9464   | 0.0089  |
| Decision Tree       | 0.9233   | 0.0111  |
| Random Forest       | 0.9638   | 0.0053  |
| Gradient Boosting   | 0.9592   | 0.0095  |

Cross-validation provides a more reliable estimate of model performance than using only one train-test split. Since the model is evaluated on multiple folds, the results are less dependent on a lucky or unlucky split.

The mean AUC represents the average performance across all folds, while the standard deviation shows how consistent the model is.

Among all models, **Random Forest** achieved the highest average AUC (**0.9638**) and the lowest standard deviation (**0.0053**), making it both the strongest and the most consistent model during cross-validation.

---

## Hyperparameter Tuning with GridSearchCV

**Parameter grid tested**

* n_estimators: **[50, 100, 200]**
* max_depth: **[5, 10, None]**
* min_samples_leaf: **[1, 5]**

Total configurations:

**3 × 3 × 2 = 18 configurations × 5 folds = 90 model fits**

| Parameter           | Value  |
| ------------------- | ------ |
| max_depth           | None   |
| min_samples_leaf    | 5      |
| n_estimators        | 50     |
| Best CV Score (AUC) | 0.9594 |

GridSearchCV checks every possible combination of the selected hyperparameters and chooses the one with the best cross-validation score. This approach is thorough, although it becomes slower when the search space is large.

RandomizedSearchCV would evaluate only a random selection of combinations, making it faster but with a chance of missing the best combination.

For this project, only **18 configurations** were tested, so GridSearchCV was a practical choice.

The best-performing model used **50 trees**, **unlimited depth**, and **min_samples_leaf = 5**, showing that increasing the number of trees was not necessary for achieving the best result.

---

## Learning Curve (Data Quantity vs Capacity)

| Fraction | Training AUC | Test AUC |
| -------- | ------------ | -------- |
| 0.2      | 0.9803       | 0.9476   |
| 0.4      | 0.9836       | 0.9441   |
| 0.6      | 0.9856       | 0.9643   |
| 0.8      | 0.9873       | 0.9743   |
| 1.0      | 0.9897       | 0.9713   |

The training AUC remains above **0.98** throughout, showing that the model fits the training data very well regardless of the amount of training data used.

The test AUC generally improves as more training data is added, reaching its highest value at **80%** of the data before dropping slightly at **100%**. This small decrease is most likely due to normal variation rather than a significant issue.

Overall, the learning curve suggests that adding more data improves model performance, although the improvements become smaller as the dataset grows.

---

## Model Serialization

The best pipeline was saved as **best_model.pkl** using **joblib.dump**. The saved model was loaded again using **joblib.load**, and it successfully generated predictions without any issues.

---

## Summary Comparison Table (All Models from Parts 2 and 3)

| Model                        | 5-Fold CV Mean AUC | 5-Fold CV Std | Test Set AUC |
| ---------------------------- | ------------------ | ------------- | ------------ |
| Logistic Regression (Part 2) | 0.9464             | 0.0089        | 0.9607       |
| Decision Tree (max_depth=5)  | 0.9233             | 0.0111        | 0.9048       |
| Random Forest (Part 3)       | 0.9638             | 0.0053        | 0.9717       |
| Gradient Boosting (Part 3)   | 0.9592             | 0.0095        | 0.9776       |

**Recommendation**

Based on the overall results, **Gradient Boosting** is the best model to deploy. It achieved the highest test-set AUC (**0.9776**), showing that it performs the best at distinguishing between high-demand and low-demand rental days.

Although Random Forest achieved a slightly higher average cross-validation AUC, Gradient Boosting performed better on the final test set, which better represents real-world performance. Its standard deviation is also reasonably low, indicating stable performance across different data splits.

Another advantage is that Gradient Boosting is efficient to retrain and is well suited for production environments.

---

## Files in this repo

* **cleaned_data.csv** – Input dataset produced in Part 1.
* **part3.ipynb** – Contains all code for ensemble models, hyperparameter tuning, feature ablation, learning curves, and model serialization.
* **best_model.pkl** – Saved Gradient Boosting pipeline that can be loaded directly for making predictions.
