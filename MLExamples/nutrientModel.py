"""
Original file is located at
    https://colab.research.google.com/drive/1D8bMHPGFQu8z2dia_c01GgQOdoKjVFKG
"""

# Importing libraries and packages for basic statistics
import os # To change working directory
import pandas as pd # to read and manipulating data 
import numpy as np # to calculate mean and standard deviations

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#Load dataset to start EDA
#Changing working directory

# To read 'csv' file with panda library
df = pd.read_csv('/content/Fertilizer_Prediction.csv')

# To display the first 10 rows of dataset
display(df.head(10))

# To find Column name
df.columns

# To find the number of rows and columns
print(df.shape)

# check for the data types, memory usage, etc
display(df.info())

# checking the no. of missing values in the dataset
df.isnull().sum()

# statistics of the numerical variables
display(df.describe().T)

# statistics of the category variables
display(df.describe(include='object'))

from sklearn.preprocessing import MinMaxScaler # to normalize data
from sklearn.preprocessing import LabelEncoder # to encode object variable to numeric
from sklearn.model_selection import train_test_split # to split data into training

X = df.drop(['Fertilizer'], axis=1) #feature variables
y = df[['Fertilizer']] #Target variable
print('The shape of feature set, X is ' , X.shape)
print('The shape of target, y is ' , y.shape)

#Label Encoding 
le = LabelEncoder()
df['Fertilizer']= le.fit_transform(df['Fertilizer'])

display(df.head())

# normalize the feature(X) columns 
scaler = MinMaxScaler()

for col in X.columns:
    X[col] = scaler.fit_transform(X[[col]])

display(X.sample(10))

# Create train and test set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)

print('Shape of X_train is', X_train.shape)
print('Shape of X_test is', X_test.shape)
print('Shape of y_train is', y_train.shape)
print('Shape of y_test is',  y_test.shape)

# Importing libraries for classification and performance evaluation
from sklearn.neighbors import KNeighborsClassifier #to build KNeighbors model
from sklearn.ensemble import GradientBoostingClassifier #to build GradientBoosting model
from sklearn.ensemble import RandomForestClassifier #to build RandomForest model
from sklearn.tree import DecisionTreeClassifier #to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import GridSearchCV # to best select hyperparameter

from sklearn.metrics import accuracy_score, classification_report # to calcutate accuracy of model
from sklearn.metrics import classification_report #to calculte precision, recall, f1-score
#from sklearn.metrics import plot_confusion_matrix # to draw confusion_matrix

#Random Forest model
model_RF = RandomForestClassifier(random_state=42)

# Train the model using the training sets
model_RF = model_RF.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_RF = model_RF.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print('Accuracy: ', accuracy_score(y_test, y_pred_RF))

#Classification report
print(classification_report(y_test, y_pred_RF))

print("""Random Forest\t\t\t {:.4f}""".format(accuracy_score(y_test, y_pred_RF)))

df # Displaying dataset again

#For Random Forest model
#inputs: pH, N, P, K
# values = (curent - goal)
model = DecisionTreeClassifier()
X = X.values # conversion of X  into array
data = np.array([[(13-6.5), (0-110), (45-30), (23-110)]])
prediction = model_RF.predict(data)
print(prediction)