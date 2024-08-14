# SVM Kernel Implementation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.io as pio

# Setting Plotly renderer to browser
pio.renderers.default = 'browser'

# Generate data points for two circles
x = np.linspace(-5.0, 5.0, 100)
y = np.sqrt(10**2 - x**2)
y = np.hstack([y, -y])
x = np.hstack([x, -x])

x1 = np.linspace(-5.0, 5.0, 100)
y1 = np.sqrt(5**2 - x1**2)
y1 = np.hstack([y1, -y1])
x1 = np.hstack([x1, -x1])

# Scatter plot of the data points
plt.scatter(y, x, label='Class 0')
plt.scatter(y1, x1, label='Class 1')
plt.legend()
plt.title("Data Points for Two Classes")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Creating DataFrames for the two classes
df1 = pd.DataFrame(np.vstack([y, x]).T, columns=['X1', 'X2'])
df1['Y'] = 0
df2 = pd.DataFrame(np.vstack([y1, x1]).T, columns=['X1', 'X2'])
df2['Y'] = 1

# Merging the DataFrames
df = pd.concat([df1, df2], ignore_index=True)

# Displaying the first and last 5 rows of the DataFrame
print("First 5 rows of the dataset:")
print(df.head())
print("\nLast 5 rows of the dataset:")
print(df.tail())

# Independent and Dependent features
X = df.iloc[:, :2]
y = df['Y']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training the SVM with RBF Kernel
classifier = SVC(kernel="rbf")
classifier.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with RBF Kernel: {accuracy:.2f}")

# Creating additional features for Polynomial Kernel
df['X1_Square'] = df['X1']**2
df['X2_Square'] = df['X2']**2
df['X1*X2'] = df['X1'] * df['X2']

# Independent and Dependent features for Polynomial Kernel
X = df[['X1', 'X2', 'X1_Square', 'X2_Square', 'X1*X2']]
y = df['Y']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 3D Scatter plot of the features
fig = px.scatter_3d(df, x='X1', y='X2', z='X1*X2', color='Y', title="3D Scatter Plot of X1, X2, and X1*X2")
fig.show()

fig = px.scatter_3d(df, x='X1_Square', y='X2_Square', z='X1*X2', color='Y', title="3D Scatter Plot of X1^2, X2^2, and X1*X2")
fig.show()

# Training the SVM with Linear Kernel
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Linear Kernel: {accuracy:.2f}")
