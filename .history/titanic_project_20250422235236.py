import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv('titanic.csv')

# Handle Missing Data for 'Age' and 'Embarked' (Impute missing values)
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])

imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer.fit_transform(df[['Embarked']]).ravel()  # Flatten to 1D

# Handle Categorical Data (Convert categorical features into numeric using get_dummies)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Feature Selection (Remove non-relevant columns)
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

# Train-Test Split (Splitting data for model training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Using RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and Evaluation (Calculating accuracy)
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Get the PassengerId from the original dataframe for the test set
test_passenger_ids = df.loc[X_test.index, 'PassengerId']

# Save predictions for test data (if required)
test_predictions = model.predict(X_test)

# Save results into a submission file
submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)
