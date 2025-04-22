import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load Dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Handle Missing Data (Impute missing values)
imputer = SimpleImputer(strategy='median')
train['Age'] = imputer.fit_transform(train[['Age']])

imputer = SimpleImputer(strategy='most_frequent')
train['Embarked'] = imputer.fit_transform(train[['Embarked']])

# Handle Categorical Data
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

# Feature Selection
X = train.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = train['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Save predictions for test data
test['Age'] = imputer.transform(test[['Age']])
test_predictions = model.predict(test.drop(['Name', 'Ticket', 'Cabin'], axis=1))

# Save submission file
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)
