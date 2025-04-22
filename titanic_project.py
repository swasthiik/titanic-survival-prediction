import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv('titanic.csv')

# Handle Missing Data
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])

imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer.fit_transform(df[['Embarked']]).ravel()

# Convert Categorical Features
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Prepare Features and Labels
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f' Model Accuracy: {acc:.2f}')

# Save submission file
test_passenger_ids = df.loc[X_test.index, 'PassengerId']
submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': y_pred})
submission.to_csv('submission.csv', index=False)
print(" Submission file saved as 'submission.csv'")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
