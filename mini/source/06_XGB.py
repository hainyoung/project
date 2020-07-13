import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from sklearn.multioutput import MultiOutputClassifier

x = np.load('./data/x_data.npy')
y = np.load('./data/y_data.npy')
x_pred = np.load('./data/x_pred.npy')

print("x.shape :", x.shape)
print("y.shape :", y.shape)
print("x_pred.shape :", x_pred.shape)


x = x.reshape(x.shape[0], 64*64*3)

x_pred = x_pred.reshape(x_pred.shape[0], 64*64*3)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 77, shuffle = True
)

# model = XGBClassifier()
model = MultiOutputClassifier(XGBRFClassifier())

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

print("acc :", acc)

y_pred = model.predict(x_pred)