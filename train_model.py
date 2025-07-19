# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("studentsperformance.csv")

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['pass_fail'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(['math score', 'reading score', 'writing score', 'average_score', 'pass_fail'], axis=1)
y = df_encoded['pass_fail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 4]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print("Best Params:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(best_model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
