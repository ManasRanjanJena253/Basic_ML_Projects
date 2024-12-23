import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import csv
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset into a pands dataframe
train_data = pd.read_csv('../Important Datasets 2/Spaceship titanic train data.csv')
test_data = pd.read_csv('../Important Datasets 2/Spaceship titanic test data.csv')

# Getting information about the data
print('Info about the training data :: ')
print('------------------------------------------------------------------------------')
print(train_data.info())
print(train_data.isnull().sum())
print(train_data['Transported'].value_counts())
print('------------------------------------------------------------------------------')
print('Info about the test data :: ')
print(test_data.info())


# Removing unimportant data and managing missing values
train_data.drop(['PassengerId','Name'],axis = 1, inplace = True)

train_data['HomePlanet'].fillna(train_data['HomePlanet'].mode()[0],  inplace = True)
train_data['CryoSleep'].fillna(train_data['CryoSleep'].mode()[0], inplace = True)
train_data['Cabin'].fillna(train_data['Cabin'].mode()[0], inplace = True)
train_data['Destination'].fillna(train_data['Destination'].mode()[0], inplace = True)
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
train_data['RoomService'].fillna(train_data['RoomService'].mean(), inplace = True)
train_data['FoodCourt'].fillna(train_data['FoodCourt'].mean(), inplace = True)
train_data['ShoppingMall'].fillna(train_data['ShoppingMall'].mean(), inplace = True)
train_data['Spa'].fillna(train_data['Spa'].mean(), inplace = True)
train_data['VIP'].fillna(train_data['VIP'].mode()[0], inplace = True)
train_data['VRDeck'].fillna(train_data['VRDeck'].mean(), inplace = True)
train_data.replace(['True','False'],[1,0], inplace = True)
test_data.replace(['True','False'],[1,0], inplace = True)

categorical_colmns = ['HomePlanet','CryoSleep','Cabin','VIP','Destination']
label_encode = LabelEncoder()
# List of categorical columns
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for i in categorical_columns:
        train_data[i] = label_encode.fit_transform(train_data[i])
        test_data[i] = label_encode.fit_transform(test_data[i])
# Verify there are no NaN values and that all categorical columns are strings
print(train_data[categorical_columns].info())
print(train_data[categorical_columns].isnull().sum())


# Combining similar columns
# Ensure all numerical columns are of type float
expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for k in expense_columns:
        train_data['Expenses'] = ' '+str(train_data[k])
        test_data['Expenses'] = ' '+str(test_data[k])
train_data['Expenses'] = label_encode.fit_transform(train_data['Expenses'])
test_data['Expenses'] = label_encode.fit_transform(test_data['Expenses'])

# Drop the original individual expense columns after creating 'Expenses'
train_data.drop(columns=expense_columns, inplace=True)
test_data.drop(columns=expense_columns, inplace=True)


print('------------------------------------------------------------------------------')
print(train_data.info())
print(train_data['Transported'])
print(train_data.isnull().sum())

# Splitting the data into features and labels
x = train_data.drop('Transported', axis = 1)
y = train_data['Transported']

# Hyperparameter tuning
param_grid = {
        'num_leaves': [31, 50, 100],               # Controls the complexity of the model
        'max_depth': [-1, 10, 20],                 # Depth of the trees (-1 means no limit)
        'learning_rate': [0.1, 0.01, 0.05],        # Step size for each iteration
        'n_estimators': [100, 200, 500],           # Number of boosting iterations
        'min_child_samples': [20, 30, 50],         # Minimum data per leaf, to prevent overfitting
        'subsample': [0.8, 1.0],                   # Fraction of data to use for each tree
        'colsample_bytree': [0.8, 1.0],            # Fraction of features to use for each tree
        'reg_alpha': [0, 0.1, 1],                  # L1 regularization term
        'reg_lambda': [0, 0.1, 1]                  # L2 regularization term
}
# Training the model
model = GridSearchCV(LGBMClassifier(), param_grid)
model.fit(x, y)

train_pred = model.predict(x)
print('The accuracy of a model :: ', accuracy_score(y,train_pred))

print(test_data.shape)

# Writing the test_data predictions into a csv file
f = open('SpaceShip Titanic Survivability.csv', 'w+')
writer = csv.writer(f)
l = []
header = ['PassengerId','Transported']
writer.writerow(header)
test_predictions = model.predict(test_data.drop(['PassengerId','Name'], axis = 1, inplace = False))
print(test_data.drop(['PassengerId','Name'], axis = 1, inplace = False).isnull().sum())
print(test_predictions)
for k in tqdm(test_predictions):
        if k == 1:
                l.append('True')
        else:
                l.append('False')
print(l)
for k in tqdm (range (4277)):
        i = []
        i.append(test_data['PassengerId'][k])
        i.append(l[k])
        writer.writerow(i)
f.close()