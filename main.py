from data import preprocess
from model import model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import MeanAbsoluteError

# 데이터 로드 & 전처리

## 매개변수: 회사번호, 시작일자, 종료일자
x, y = preprocess.Preprocessing('005930', '2018-05-04', '2020-01-22')

(train_x, train_y), (test_x, test_y) = preprocess.InputDataSet(x, y)

# 모델 훈련하기
mesg = """
        어떤 모델을 훈련시키겠습니까? ╰(*°▽°*)╯
        1번: layer가 적은 모델
        2번: layer 추가 모델
        3번: 1, 2번 모두 출력
        4번: 제작자 정보보기
        10번: 종료하기
        """

while 1:
    command = int(input(f'{mesg} 숫자만 입력하기 :) '))
    if command == 1:
        model1 = model.lstm(train_x, train_y)
        print("모델 학습을 끝냈습니다! 결과를 출력합니다 (～￣▽￣)～ ")
        model.ShowGraph(model1, test_x, test_y)
    elif command == 2:
        model2 = model.lstm_forced(train_x, train_y)
        print("모델 학습을 끝냈습니다! 결과를 출력합니다 (～￣▽￣)～ ")
        model.ShowGraph(model2, test_x, test_y)
    elif command == 3:
        model1 = model.lstm(train_x, train_y)
        model2 = model.lstm_forced(train_x, train_y)
        print("모델 학습을 끝냈습니다! 결과를 출력합니다 (～￣▽￣)～ ")
        model.ShowGraph2(model1, model2, test_x, test_y)
    elif command == 4:
        print("\n개인 이메일: hgk5722@naver.com")
        print("개인 깃허브: https://github.com/hgk5722")
        print("개인 블로그: https://hgk5722.tistory.com/")
    elif command == 10:
        print("프로그램을 종료하겠습니다! (●'◡'●)")
        break
    else:
        print("잘못된 입력입니다. 다시 입력해주세요!")


# 훈련된 모델 확인


