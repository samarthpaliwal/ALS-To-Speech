# train_model.py
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = [], []
for letter in os.listdir('data'):
    files = os.listdir(f"data/{letter}")
    for fname in files:
        vec = pickle.load(open(f"data/{letter}/{fname}", 'rb'))
        X.append(vec)
        y.append(letter)

clf = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

pickle.dump(clf, open('asl_rf_model.pkl', 'wb'))
