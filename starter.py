import numpy as np
import json

# Implementation of a neural network using only Numpy
# trained using gradient descent with momentum

# Load data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Flatten data
def flattenData(data):
    return np.reshape(data, newshape=(data.shape[0], -1))

# One hot
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

# Shuffle data
def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


# x: N * 1000
# return: N * 1000
def relu(x):
    return x * (x > 0)

def gradRelu(x):
    return x > 0

# x: N * 10
# return: N * 10
def softmax(x):
    ex = np.exp(x - x.max(axis=1)[:, np.newaxis])
    row_sums = np.sum(ex, axis=1)
    return ex / row_sums[:, np.newaxis]

# s: np array
# return np array
def gradSoftmax(s):
    ex = np.exp(s - s.max())
    row_sums = np.sum(ex)
    row_square_sums = row_sums ** 2
    a = ex / row_square_sums
    b = row_sums - ex
    return a * b

# X: N * F, W: F * K, b: 1 * K or 1-d array with K entries
def compute(X, W, b):
    return np.dot(X, W) + b

# average CE
# targets, predictions: N * K
# return: a number
def averageCE(targets, predictions):
    epsilon = 1e-9 # avoid log(0)
    N = predictions.shape[0]
    return -np.sum(targets * np.log(predictions + epsilon)) / N

# targets, predictions: N * 10
# w.r.t probabilities of each 10 classes
# return np array of 10 dimensions
def gradCE(targets, predictions):
    epsilon = 1e-9  # avoid 0
    return -np.mean(targets / (predictions + epsilon), axis=0)

def XaiverInit(unitin, unitout, size):
    return np.random.normal(0, np.sqrt(2 / (unitin + unitout)), size)

def forward(inputs, W_h, b_h, W_o, b_o):
    S_1 = compute(inputs, W_h, b_h)
    X_1 = relu(S_1)
    S_2 = compute(X_1, W_o, b_o)
    X_2 = softmax(S_2)
    return [S_1, X_1, S_2, X_2]

def accuracy(targets, predictions):
    a = np.zeros_like(predictions)
    a[np.arange(len(predictions)), predictions.argmax(1)] = 1
    return np.sum(np.all(a == targets, axis=1)) / predictions.shape[0]

if __name__ == '__main__':
    ''' data processing '''
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData, validData, testData = flattenData(trainData), flattenData(validData), flattenData(testData)
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

    ''' back propagation '''
    # configuration
    epochs = 200
    hidden_units = 100
    classes = trainTarget.shape[1]
    features = trainData.shape[1]
    gama, alpha = 0.9, 0.001
    log_name = ""
    log = []
    log.append({
        "hidden_units": hidden_units,
        "learning_rate": alpha
    })

    # initialize the weights and bias
    W_h = XaiverInit(features, hidden_units, (features, hidden_units))
    v_Wh = np.full((features, hidden_units), 1e-5)
    b_h = XaiverInit(features, hidden_units, (1, hidden_units))
    v_bh = np.full((1, hidden_units), 1e-5)

    W_o = XaiverInit(hidden_units, classes, (hidden_units, classes))
    v_Wo = np.full((hidden_units, classes), 1e-5)
    b_o = XaiverInit(hidden_units, classes, (1, classes))
    v_bo = np.full((1, classes), 1e-5)

    # training
    for i in range(epochs):
        # caculate all X, S, losses and accuracy with forward propagation
        _, _, _, X_2 = forward(testData, W_h, b_h, W_o, b_o)
        testCE = averageCE(testTarget, X_2)
        testAcc = accuracy(testTarget, X_2)
        _, _, _, X_2 = forward(validData, W_h, b_h, W_o, b_o)
        validCE = averageCE(validTarget, X_2)
        validAcc = accuracy(validTarget, X_2)
        S_1, X_1, S_2, X_2 = forward(trainData, W_h, b_h, W_o, b_o)
        trainCE = averageCE(trainTarget, X_2)
        trainAcc = accuracy(trainTarget, X_2)
        log.append({
            'iterations': i,
            "trainloss": trainCE,
            "validloss": validCE,
            "testloss": testCE,
            "trainacc": trainAcc,
            "validacc": validAcc,
            "testacc": testAcc
        })
        # print('iterations {}: trainloss {}, validloss {}, testloss {}'.format(i, trainCE, validCE, testCE))
        # print('iterations {}: trainacc {}, validacc {}, testacc {}'.format(i, trainAcc, validAcc, testAcc))

        # calculate gradWo, gradbo, gradWh, gradbh
        trainNum = trainData.shape[0]
        gradWo, gradbo, gradWh, gradbh = 0, 0, 0, 0
        for j in range(trainNum):
            Delta_2 = (-trainTarget[j, :] / (X_2[j, :] + 1e-5)) * gradSoftmax(S_2[j, :])
            gradWo += np.dot(X_1[j, :].reshape(-1, 1), Delta_2.reshape(1, -1))
            gradbo += Delta_2
            Delta_1 = np.dot(Delta_2, W_o.T) * gradRelu(S_1[j, :])
            gradWh += np.dot(trainData[j, :].reshape(-1, 1), Delta_1.reshape(1, -1))
            gradbh += Delta_1
        gradWo /= trainNum
        gradbo /= trainNum
        gradWh /= trainNum
        gradbh /= trainNum

        # gd with momentum
        v_Wo = gama * v_Wo + alpha * gradWo
        W_o -= v_Wo
        v_bo = gama * v_bo + alpha * gradbo
        b_o -= v_bo
        v_Wh = gama * v_Wh + alpha * gradWh
        W_h -= v_Wh
        v_bh = gama * v_bh + alpha * gradbh
        b_h -= v_bh

    if not log_name:
        print("no log_name")
    else:
        json.dump(log, open(log_name, 'w'))
        print("write {} done".format(log_name))


