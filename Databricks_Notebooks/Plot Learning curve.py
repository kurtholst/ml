# Databricks notebook source
# MAGIC %md
# MAGIC # Plotting Learning Curves
# MAGIC ![Miljøstyrelsen_Logo](https://mst.dk/media/113238/logo-mst.png)
# MAGIC ![LearningCurve](https://chrisalbon.com/images/machine_learning_flashcards/Learning_Curve_print.png)
# MAGIC 
# MAGIC Reference: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

# COMMAND ----------

# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve #


# COMMAND ----------

# Load data
digits = load_digits()

# Create feature matrix and target vector
X = digits.data
y = digits.target

# COMMAND ----------

# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(),
                                                        X, 
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv = 10,
                                                        # Evaluation metric
                                                        scoring = 'accuracy',
                                                        # Use all computer cores
                                                        n_jobs = -1,
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# COMMAND ----------

# Plotting Learning Curve
# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

# COMMAND ----------

