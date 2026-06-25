import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

print("Original Class Distribution:")
print(train['is_fraud'].value_counts())

fraud = train[train['is_fraud'] == 1]
legit = train[train['is_fraud'] == 0]

legit_sample = legit.sample(n=len(fraud), random_state=42)

train = pd.concat([fraud, legit_sample])
train = train.sample(frac=1, random_state=42)

print("\nBalanced Class Distribution:")
print(train['is_fraud'].value_counts())

train['trans_date_trans_time'] = pd.to_datetime(train['trans_date_trans_time'])
test['trans_date_trans_time'] = pd.to_datetime(test['trans_date_trans_time'])

train['hour'] = train['trans_date_trans_time'].dt.hour
test['hour'] = test['trans_date_trans_time'].dt.hour

cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']

for col in cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.fit_transform(test[col].astype(str))

drop_cols = [
    'Unnamed: 0',
    'trans_date_trans_time',
    'cc_num',
    'first',
    'last',
    'street',
    'trans_num',
    'dob',
    'is_fraud'
]

X_train = train.drop(columns=drop_cols, errors='ignore')
y_train = train['is_fraud']

X_test = test.drop(columns=drop_cols, errors='ignore')
y_test = test['is_fraud']

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nSample Prediction:")

if y_pred[0] == 1:
    print("Fraud Transaction")
else:
    print("Legitimate Transaction")