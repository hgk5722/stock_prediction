from data import load
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def Preprocessing(company, start_date, end_date):
    # 데이터 불러오기
    df = load.LoadStockData(company, start_date, end_date)

    # 원하는 컬럼만 추출
    # Close를 마지막에 배치
    dfx = df[['Open','High','Low','Volume', 'Close']]

    for col in dfx.columns:
        scaler = MinMaxScaler() # 정규화: 최소값0, 최댓값1
        dfx[col] = scaler.fit_transform(dfx[[col]])
    dfy = dfx[['Close']]    # 주식 종가
    dfx = dfx[['Open','High','Low','Volume']]

    x = dfx.values.tolist()
    y = dfy.values.tolist()

    return x, y

def InputDataSet(x, y):
    window_size = 5
    data_x, data_y = [], []

    for i in range(len(y) - window_size):
        _x = x[i:i+window_size]
        _y = y[i+window_size]
        data_x.append(_x)
        data_y.append(_y)

    train_size = int(len(data_y) * 0.8)
    train_x = np.array(data_x[0 : train_size])
    train_y = np.array(data_y[0 : train_size])

    test_size = len(data_y) - train_size
    test_x = np.array(data_x[train_size : len(data_x)])
    test_y = np.array(data_y[train_size : len(data_y)])

    print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
    print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)

    return (train_x, train_y), (test_x, test_y)