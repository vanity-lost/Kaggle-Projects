import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


def preprocessing_train(train_data):
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    X = X.fillna(X.mean())

    X = X.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X.values[:] = imp.fit_transform(X)

    X = pd.get_dummies(X)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y


def preprocessing_test(test_data):
    X = test_data

    X = X.fillna(X.mean())

    X = X.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X.values[:] = imp.fit_transform(X)

    X = pd.get_dummies(X)

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X


train_data = pd.read_csv('train.csv')
X_train, y_train = preprocessing_train(train_data)
sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)

xgboost = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=3, random_state=0).fit(X_train, y_train)

test_data = pd.read_csv('test.csv')
X_test = preprocessing_test(test_data)
preds = xgboost.predict(X_test)

sub = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": preds.astype(int)})
sub.to_csv('submission.csv', index=False)
