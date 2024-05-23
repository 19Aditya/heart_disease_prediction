# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Collection
data = pd.read_csv("C:/Users/asus/Downloads/heart.csv")

# Exploratory Data Analysis
print(data.head())
print(data.tail())
print(data.info())
print(data.shape)

# Correlation between every attribute
plt.figure(figsize=(20, 12))
sns.set_context('notebook', font_scale=1.3)
sns.heatmap(data.corr(), annot=True, linewidth=2)
plt.tight_layout()
plt.show()

sns.set_context('notebook', font_scale=2.3)
data.drop('target', axis=1).corrwith(data.target).plot(kind='bar', grid=True, figsize=(20, 10), 
                                                       title="Correlation with the target feature")
plt.tight_layout()
plt.show()

# Age analysis
Young = data[(data.age >= 29) & (data.age < 40)]
Middle = data[(data.age >= 40) & (data.age < 55)]
Elder = data[(data.age > 55)]

plt.figure(figsize=(23, 10))
sns.set_context('notebook', font_scale=1.5)
sns.barplot(x=['young ages', 'middle ages', 'elderly ages'], y=[len(Young), len(Middle), len(Elder)])
plt.tight_layout()
plt.show()

# Sex Analysis
plt.figure(figsize=(12, 9))
sns.set_context('notebook', font_scale=1.5)
sns.countplot(x='sex', data=data)
plt.tight_layout()
plt.show()

# Chest Pain Analysis
plt.figure(figsize=(18, 9))
sns.set_context('notebook', font_scale=1.5)
sns.countplot(x='cp', data=data)
plt.tight_layout()
plt.show()

# Feature engineering
categorical_val = []
continuous_val = []
for column in data.columns:
    print("--------------------")
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continuous_val.append(column)

# Data Splitting
from sklearn.model_selection import train_test_split

X = data.drop('target', axis=1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# KNN Model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
print('Accuracy of knn model: ', accuracy_score(y_test, y_pred1))
print('Classification report of knn model:\n', classification_report(y_test, y_pred1))

# Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred2 = lr.predict(X_test)
print('Accuracy of logistic regression model: ', accuracy_score(y_test, y_pred2))
print('Classification report of logistic regression model:\n')
print(classification_report(y_test, y_pred2))

# Random Forest Model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred3 = rf.predict(X_test)
print('Accuracy of random forest model: ', accuracy_score(y_test, y_pred3))
print('Classification report of random forest model:\n')
print(classification_report(y_test, y_pred3))

# Predictive System
input_data = (3, 61, 1, 0, 148, 203, 0, 1, 161, 0, 2, 1, 3)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

for model in [knn, lr, rf]:
    prediction = model.predict(input_data_reshaped)
    print(model)
    print(prediction)
    if prediction[0] == 0:
        print('The Person does not have a Heart Disease')
    else:
        print('The Person has Heart Disease')
