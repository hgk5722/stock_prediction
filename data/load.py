import FinanceDataReader as fdr

def LoadStockData(company, start_date, end_date):
    df = fdr.DataReader(company, start_date, end_date)
    
    print('\n\t 불러온 데이터의 일부를 출력합니다! \n\n', df.head(5))
    return df

