import numpy as np


# Simple Perceptron Process
# inputs = [1, x1, x2, xn]
# weights = [w0, w1, w2, wn]
# a = w0 + x1w1 + x2w2 + xnwn
# output = 1 if a>1.0
#          else 0

#Training Process
# Initialize the weights to 0 or to a small random vaue
# For each example j in our training set D, perform the following steps over the input xj
# and the desired output dj : Calculate the outpt -> update the weight

# example AND

threshold = 0
learning_rate = 0.05

def __main__():
    #        bias    x1  x2  y
    data = [    [-1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [-1.0, 1.0, 1.0, 1.0]
            ]
    #           w0  w1  w2  bias
    weights = [0.3, 0.4, 0.1]
    #print(data[0][:3])
    train(data[2], weights)

def train(data, weights):
    bias = -1
    res = 0.0
    answer=data[3]
    print(answer)

    net = weights[0]*data[0]+ weights[1]*data[1] + weights[2]*data[2]

    print('net result : ',net)

   # step function
    if(net>=threshold):
        f_net = 1
    else :
        f_net = 0

    # learning rule
    for index in range(len(weights)):
        weights[index] = weights[index] + learning_rate*data[index]*(data[index]-f_net)

    print(weights)

if __name__ == '__main__':
    __main__()


