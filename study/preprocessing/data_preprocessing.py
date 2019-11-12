import pandas as pd
from io import StringIO

csv_data = \
'''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
'''
# 누락된 데이터 판별
df =pd.read_csv(StringIO(csv_data))
# print(df)
# print(df.isnull().sum())
# # values 사용하여 배열 획득 가능
# print(df.values)

# na 제거 0은 행, 1은 열
# print(df.dropna(axis=0))
# print(df.dropna(axis=1))

# 모든 열이 NaN 일 떄만 행을 삭제
df.dropna(how='all')
# 실수값이 4개 이하인 작은 행 삭제
df.dropna(thresh=4)
# 특정 열에 NaN 이 있는 행 삭제
# print(df.dropna(subset=['C']))
import numpy as np
from sklearn.impute import SimpleImputer

# 누락된 곳을 열의 평균으로 채워넣음, strategy 에 constant를 사용한뒤 fill_value 매개변수에 채우려는 값을 지정할수 있음
simr = SimpleImputer(missing_values=np.nan, strategy='mean')
simr = simr.fit(df.values)
imputed_data = simr.transform(df.values)
# print(imputed_data)

# 누락된 행의 평균으로 채워넣음
from sklearn.preprocessing import FunctionTransformer
ftr_simr = FunctionTransformer(lambda X: simr.fit_transform(X.T).T, validate=False)
# print(ftr_simr)
imputed_data = ftr_simr.fit_transform(df.values)
# print(imputed_data)

# 범주형 데이터

import pandas as pd
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

# 순서 특성 매핑하기
size_mapping = {'XL': 3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
# print(df)
# ket value 뒤집기
inv_size_mapping = {v: k for k,v in size_mapping.items()}
# print(inv_size_mapping)
# print(df['size'].map(inv_size_mapping))

## 문자열 클래스를 정수로 변환 
# 매핑 딕셔너리를 생성
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# print(class_mapping)
# 문자열 클래스 레이블을 정수로 변환
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# print(df)

# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
# print(y)
# print(class_le.inverse_transform(y))

# 순서가 없는 특성에 원-핫 인코딩 적용하기
X = df[['color', 'size', 'price']].values
# print(X)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)

# 범주형 데이터를 정수로 인코딩
from sklearn.compose import ColumnTransformer
# 열마다 다른 변환을 적용하도록 도와주는 클래스
from sklearn.preprocessing import OrdinalEncoder
# 위 두개의 클래스를 활용하면 여러개의 열을 한번에 정수로 변환가능
ord_enc = OrdinalEncoder(dtype=np.int)
# ColumnTransformer(트랜스포머의 리스트) 
# 트랜스포머의 리스트(튜플) :[이름, 변환기, 변환할 열의 리스트]
col_trans = ColumnTransformer([('ord_enc', ord_enc, ['color'])])
print(df)
X_trans = col_trans.fit_transform(df)
print(X_trans)
