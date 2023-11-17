# stock_prediction
FinanceDataReader에서 가져온 데이터를 RNN의 종류인 LSTM 모델을 이용하여 삼성전자 주가 예측 프로그램 제작

<br><br>

# 사용시 유의사항

*main.py 파일의 Preprocessing()함수 매개변수를 수정해 주세요!

순서대로 회사번호, 시작일자, 종료일자를 의미합니다!

```
## 매개변수: 회사번호, 시작일자, 종료일자
x, y = preprocess.Preprocessing('005930', '2018-05-04', '2020-01-22')
```

---

프로젝트 학습 및 기간: 5일

사용언어: 파이썬

사용모델: LSTM, 활성함수: RELU 

컴파일 시 optimizer: adam, 손실함수: mse, 평가지표: mae

epochs는 100으로 지정!

---
![image](https://github.com/hgk5722/stock_prediction/assets/95735511/1eacc67a-8d8f-4b51-9bfa-e50ee9bcc2df)
