{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "95f5f28f-308c-4d95-9a1a-e9ee94d83242",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Params: {'max_depth': 10, 'min_samples_split': 4, 'n_estimators': 50}\n",
      "Accuracy: 0.855\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.00      0.00      0.00        27\n",
      "           1       0.86      0.99      0.92       173\n",
      "\n",
      "    accuracy                           0.85       200\n",
      "   macro avg       0.43      0.49      0.46       200\n",
      "weighted avg       0.75      0.85      0.80       200\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['features.pkl']"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train_model.py\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "import joblib\n",
    "\n",
    "df = pd.read_csv(\"studentsperformance.csv\")\n",
    "\n",
    "# Create a binary target: pass if average >= 50\n",
    "df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)\n",
    "df['pass_fail'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)\n",
    "\n",
    "# Optional: encode categorical columns\n",
    "df_encoded = pd.get_dummies(df, drop_first=True)\n",
    "\n",
    "# Select features\n",
    "X = df_encoded.drop(['math score', 'reading score', 'writing score', 'average_score', 'pass_fail'], axis=1)\n",
    "y = df_encoded['pass_fail']\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Grid search with Random Forest\n",
    "model = RandomForestClassifier(random_state=42)\n",
    "param_grid = {\n",
    "    'n_estimators': [50, 100],\n",
    "    'max_depth': [None, 10],\n",
    "    'min_samples_split': [2, 4]\n",
    "}\n",
    "\n",
    "grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')\n",
    "grid.fit(X_train, y_train)\n",
    "\n",
    "# Best model\n",
    "best_model = grid.best_estimator_\n",
    "\n",
    "# Evaluate\n",
    "y_pred = best_model.predict(X_test)\n",
    "print(\"Best Params:\", grid.best_params_)\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# Save model and features\n",
    "joblib.dump(best_model, \"model.pkl\")\n",
    "joblib.dump(X.columns.tolist(), \"features.pkl\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
