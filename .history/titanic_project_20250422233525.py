import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# Step 2: Data Cleaning (Drop unnecessary columns and handle missing values)
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
df.dropna(subset=['Embarked'], inplace=True)  # Handle missing 'Embarked'
df['Age'].fillna(df['Age'].median(), inplace=True)

# Step 3: Encoding categorical variables (Embarked, Sex)
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Step 4: Features and Labels (X - Features, y - Target)
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# Step 5: Train/Test Split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Saving the Model (Optional)
joblib.dump(model, 'titanic_model.pkl')
print("Model saved as titanic_model.pkl")

# Step 9: Testing with the saved model (load model and make prediction)
loaded_model = joblib.load('titanic_model.pkl')
y_loaded_pred = loaded_model.predict(X_test)
print("Loaded Model Accuracy:", accuracy_score(y_test, y_loaded_pred))
