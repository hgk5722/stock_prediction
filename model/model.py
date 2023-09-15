from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import MeanAbsoluteError

import numpy as np
import matplotlib.pyplot as plt


def lstm(train_x, train_y):
    # 데이터 reshape
    train_x = train_x.reshape(train_x.shape[0], -1, train_x.shape[-1])

    # 모델 선언
    model = Sequential()
    model.add(LSTM(20, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1)) # y값은 Close 하나의 컬럼: 1차원

    model.compile(optimizer='SGD', loss='mean_squared_error', metrics = [MeanAbsoluteError()])

    model.summary()

    # 모델 훈련하기
    model.fit(train_x, train_y, epochs=100)
    return model


def ShowGraph(model, test_x, test_y):
    pred_y = model.predict(test_x)
    pred_y = np.mean(pred_y, axis=1)  # 각 시퀀스의 평균을 사용하여 1D 배열로 변환

    plt.rcParams['font.family'] = 'NanumGothic'
    plt.figure()
    plt.plot(test_y, color='red', label='실제 데이터')
    plt.plot(pred_y, color='blue', label='예측 데이터')
    plt.legend()
    plt.show()


def lstm_forced(train_x, train_y):
    # 데이터 reshape
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2])

    # 모델 선언
    model2 = Sequential()
    model2.add(LSTM(40, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
    model2.add(LSTM(60, activation='relu', return_sequences=True))
    model2.add(LSTM(50, activation='relu', return_sequences=True))
    model2.add(Dense(1)) # y값은 Close 하나의 컬럼: 1차원
    
    model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model2.summary()

    model2.fit(train_x, train_y, epochs=100)
    return model2

def ShowGraph2(model1, model2, test_x, test_y):
    # 첫 번째 모델의 예측
    pred_y1 = model1.predict(test_x)
    pred_y1 = np.mean(pred_y1, axis=1)  # 각 시퀀스의 평균을 사용하여 1D 배열로 변환

    # 두 번째 모델의 예측
    pred_y2 = model2.predict(test_x)
    pred_y2 = np.mean(pred_y2, axis=1)  # 각 시퀀스의 평균을 사용하여 1D 배열로 변환

    plt.rcParams['font.family'] = 'NanumGothic'
    # 그래프 출력
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # 1x2 그리드의 첫 번째 subplot
    plt.plot(test_y, color='red', label='실제 데이터')
    plt.plot(pred_y1, color='blue', label='예측 데이터')
    plt.legend()

    plt.subplot(1, 2, 2)  # 1x2 그리드의 두 번째 subplot
    plt.plot(test_y, color='red', label='실제 데이터')
    plt.plot(pred_y2, color='green', label='예측 데이터')
    plt.legend()

    plt.show()
