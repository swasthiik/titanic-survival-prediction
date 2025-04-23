import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("titanic.csv")

# Handle missing values
df['Age'] = SimpleImputer(strategy='median').fit_transform(df[['Age']])
df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Embarked']]).ravel()

# NEW: Drop any rows that still have NaN
df.dropna(inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Features and target
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=3)
}

# Compare models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2),
        'F1 Score': round(f1, 2)
    })

# Print results
result_df = pd.DataFrame(results)
print("\n🔍 Model Comparison Summary:")
print(result_df.to_string(index=False))
