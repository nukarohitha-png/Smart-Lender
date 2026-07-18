import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier   # pip install xgboost

# ---------- 1. Load data ----------
data = pd.read_csv('Dataset/loan_prediction.csv')

# ---------- 2. Handle missing values ----------
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    data[col] = data[col].fillna(data[col].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data = data.drop_duplicates()

# ---------- 3. Encode categorical columns ----------
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
data = data.drop('Loan_ID', axis=1)

# ---------- 4. Balance the classes ----------
majority = data[data['Loan_Status'] == 1]
minority = data[data['Loan_Status'] == 0]
minority_upsampled = minority.sample(n=len(majority), replace=True, random_state=42)
data_balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# ---------- 5. Split features/target and scale ----------
X = data_balanced.drop('Loan_Status', axis=1)
y = data_balanced['Loan_Status']

FEATURE_ORDER = list(X.columns)   # this order must match Flask app.py

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_ORDER)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 6. Train all 4 models ----------
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              eval_metric='logloss', random_state=42),
}

print(f"{'Model':<20}{'Train Acc':>12}{'Test Acc':>12}")
best_name, best_model, best_acc = None, None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:<20}{train_acc:>12.3f}{test_acc:>12.3f}")
    if test_acc > best_acc:
        best_name, best_model, best_acc = name, model, test_acc

# ---------- 7. Save the best model + scaler ----------
pickle.dump(best_model, open('Flask/rdf.pkl', 'wb'))
pickle.dump(scaler, open('Flask/scale.pkl', 'wb'))
print(f"\nBest model: {best_name} (test acc {best_acc:.3f}) saved to Flask/rdf.pkl")
