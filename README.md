# Prediction-of-Fraud-Transaction-Python-
Prediction of Fraud Transactions with Machine Learning Models and with python libraries.

1. Data Loading:
The initial step involves loading a dataset containing information about financial transactions. The dataset is stored in a CSV file ('Fraud.csv.csv') and is read into a Pandas DataFrame named data.

2. Exploratory Data Analysis (EDA):
Exploratory Data Analysis is performed to understand the characteristics of the dataset:

head() and tail() show a glimpse of the first and last few rows.
info() provides information about data types and missing values.
shape indicates the number of rows and columns in the dataset.
isna().sum() reveals the count of missing values in each column.
value_counts() examines the distribution of values in the 'isFraud' and 'isFlaggedFraud' columns.

3. Data Preprocessing:
Unnecessary columns ('nameOrig' and 'nameDest') are dropped from the dataset.
The 'type' column, representing a categorical variable, is encoded using Label Encoding to convert it into a numerical format suitable for machine learning models.

5. Splitting Data:
The dataset is split into features (x) and the target variable (y). Then, a further split is performed to create training and testing sets using the train_test_split function from scikit-learn.

6. Feature Scaling:
Standard scaling is applied to normalize the feature values. This ensures that all features contribute equally to the machine learning models.

7. Model Building and Evaluation:
Gaussian Naive Bayes:
A Gaussian Naive Bayes model is trained using the GaussianNB class from scikit-learn. This model assumes that features are normally distributed. After training, predictions are made on the test set, and the accuracy of the model is evaluated using metrics.accuracy_score.

Logistic Regression:
A Logistic Regression model is trained using the LogisticRegression class. Logistic Regression is a linear model suitable for binary classification tasks. Similar to the Naive Bayes model, predictions are made, and accuracy is calculated.

Conclusion:
The script concludes by providing the accuracy of both the Gaussian Naive Bayes and Logistic Regression models on the test set. Accuracy is a common metric for classification problems, representing the proportion of correctly predicted instances. However, it's essential to consider other metrics and perform a more in-depth analysis based on the specific requirements and characteristics of the fraud detection problem, as accuracy alone may not be sufficient, especially in imbalanced datasets where fraudulent transactions are rare.
