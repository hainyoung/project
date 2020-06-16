import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

x = np.load('./data/x_data.npy')
y = np.load('./data/y_data.npy')
x_pred = np.load('./data/x_pred.npy')

print("x.shape :", x.shape) # (200, 64, 64, 3)
print("y.shape :", y.shape) # (200, 2)
print("x_pred.shape :", x_pred.shape) #(20, 64, 64, 3)

# RandomForestClassifier : 2차원모델
x = x.reshape(x.shape[0], 64*64*3)
x_pred = x_pred.reshape(x_pred.shape[0], 64*64*3)


# train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 77, shuffle = True
)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)


# 2. 모델 구성
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_pred)
acc = model.score(x_test, y_test)

print("y_pred.shape :", y_pred.shape)
print("y_test.shape :", y_test.shape)

print("acc :", acc)

print(y_pred)
