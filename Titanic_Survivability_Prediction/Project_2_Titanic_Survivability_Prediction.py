# Importing Dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Titanic_Survivability_Prediction/train.csv')  # Loading the training data into the dataframe.
test = pd.read_csv('Titanic_Survivability_Prediction/test.csv')   # Loading the testing data into the dataframe.
print(data.head())
print(data.isnull().sum())  # Checking for missing values

# Dropping the unrequited columns from the dataset
drop_colmns = ['Name','Cabin','Fare','Embarked','PassengerId','Ticket']
data = data.drop(drop_colmns, axis = 1)

# Handling the missing values and the categorical columns
sns.displot(data['Age'])  # Checking if the data is screwed or not .
plt.show()

data['Sex'].replace(['male','female'],[0,1],inplace = True)
data['Age'].fillna(data['Age'].mean(),inplace = True)
print(data.head())

# Splitting the features and labels

feat = data.drop('Survived',axis = 1)
tgt = data['Survived']

# Training both the models
# Logistic Regression
model = RandomForestClassifier()
model.fit(feat, tgt)

# Accuracy of Logistic Regression Model
pred = model.predict(feat)
print('Accuracy on training data of Logistic Regression Model :: ',accuracy_score(tgt,pred))

# After prediction our Support Vector Model performs much better than our Logistic Regression Model , So will be using it for prediction .

test['Sex'].replace(['male','female'],[0,1],inplace = True)
test['Age'].fillna(data['Age'].mean(),inplace = True)
feat2 = test.drop(drop_colmns, axis = 1)

Test_Predictions = model.predict(feat2)
print(Test_Predictions)
