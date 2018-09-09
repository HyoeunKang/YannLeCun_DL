import Perceptron
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

# iris 데이터를 활용한 퍼셉트론
# x0은 바이어스로 지정할 것임
# x1 꽃받침 길이, x2 꽃받침 너비, x3 꽃잎 길이, x4 꽃잎 너비, y 품종
# 단층 퍼셉트론은 바이너리 결과를 가지므로 3개 이상의 푸종을 동시에 구분할 수 없음
# 현재 예제에서는 Setosa 품종과 Versicolor 2 개의 품종을 구분할 수 있도록 학습
# 데이터에서 1~50까지가 setosa(-1), 51~100까지가 versicolor(1)
style.use('seaborn-talk')

#한글 표현
kfront = {'family':'NanumGothic', 'weight':'bold', 'size':10}
matplotlib.rc('font', **kfront)
matplotlib.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    data_frame = pd.read_csv('iris.data', header=None)

    # 붖꽃 데이터를 저장한 데이터프레임에서 0~99라인까지 5번째 컬럼의 값을 numpy 배열로 리턴받아 y에 대입
    y = data_frame.iloc[0:100, 4].values
    # 저장된 품종 문자가 setosa일 경우 -1
    # y배열을 숫자 배열로 바꿈
    y = np.where(y=='Iris-setosa', -1, 1)
    # 데이터 세트는 꽃받침 길이, 꽃잎 길이 사용
    X = data_frame.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1], color='r', marker='o', label="setosa")
    plt.scatter(X[50:100,0], X[50:100,1], color='b', marker='x', label="versicolor")
    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃받침 길이')
    plt.legend(loc=4)
    plt.show()

    perceptron = Perceptron.Perceptron(eta=0.1)
    perceptron.fit(X,y)
    print(perceptron.erros_)
    print(perceptron.w_[0],"+",perceptron.w_[1],"X 꽃받침 길이 +",perceptron.w_[2],"X 꽃잎 길이")