import numpy as np
# AND 연산에 대한 퍼셉트론을 수
# 퍼셉트론을 구현하는 클래스
class Perceptron():
    # 퍼셉트론 클래스 생성. 임계값, learning rate, 학습 횟수
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
    # 트레이닝 데이터 X와 실제 결과값 y를 인자로 받아 머신러닝 수해
    # 일반적으로 트레이닝 데이터는 대문자 X, 실제 결과값은 y
    def fit(self, X, y):
        # 가중치를 numpy 배열로 지정.
        # X.shape[1]은 트레이닝 데이터의 입력값 갯수
        # 예를 들어 X가 4*2 배열인 경우, X.shape의 값은 (4,2), X.shape[1]값은 2
        # 바이어스 때문에 1개 더 더함
        self.w_ = np.zeros(1+X.shape[1]) # 0, 0, 0

        # 머신러닝 학습 반복 횟수에 따라 퍼셉트론의 예측값과 실제 결과값이 다른 오류 횟수 저장
        self.erros_ = []

        # _은 단순히 반복문을 특정 횟수만큼 반복하고자 할 때 관습적으로 사용하는 변수
        for _ in range(self.n_iter):
            # 초기 오류 횟수 정의
            err = 0

            # 트레이닝 데이터 세트 X와 결과값 y를 하나씩 꺼집어내서 xi, target 변수에 대입
            for xi, target in zip(X, y):
                # 가중치 업데이트를 위한 수식. learning rate * (target-predict)
                # 결과값과 예측값의 활성함수리턴값이 같을 경우 변화 없고, 다를 경우 유효한 값이 더해져서 가중치 업데이트
                update = self.eta * (target-self.predict(xi))

                self.w_[1:] += update * xi
                # x0는 바이어스여서 단순히 1로 보존
                self.w_[0] += update

                # 업데이트 값이 0이 아닌 경우 -> 오차가 발생한 경우 -> 에러 값을 1 증가시키고 다음 트레이닝 데이터로 넘어감
                # 모든 트레이닝 데이터에 대해 1회 학습이 끝나면 발생한 오류 횟수를 추가한 후, 가중치 화면에 출력
                err += int(update!=0.0)
            self.erros_.append(err)
            print(self.w_)

        return self

    def net_input(self, X):
        # 벡터 x,y의 내적 또는 행렬 x,y의 곲을 리턴
        # 트레이팅 데이터 각 입력값과 그에 따른 가중치를 곱한 총 합 리턴
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # 순입력 함수 결과값이 임계값보다 크다면 1, 그렇지 않으면 -1 리턴.
    # 활성함수 구현 (step fucntion)?
    def predict(self, X):
        return np.where(self.net_input(X)>self.thresholds, 1, -1)

if __name__ == '__main__':
    X = np.array([  [0,0],
                    [0,1],
                    [1,0],
                    [1,1]
                ])
    y = np.array([-1,-1,-1,1])

    perceptron = Perceptron(eta=0.1)
    perceptron.fit(X,y)
    print(perceptron.erros_)