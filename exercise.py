from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()  #导入数据包
#print(iris.data.shape) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#print(iris.data)  #查看iris的数据，以2、3列的数据为横纵坐标
speciesName = iris.target_names
fig , ax = plt.subplots(figsize=(12,8))
ax.scatter(iris.data[: , 2][iris.target==0],iris.data[:,3][iris.target==0],s=50,c='b',marker='o',label=speciesName[0])
ax.scatter(iris.data[: , 2][iris.target==1],iris.data[:,3][iris.target==1],s=50,c='r',marker='x',label=speciesName[1])
ax.scatter(iris.data[: , 2][iris.target==2],iris.data[:,3][iris.target==2],s=50,c='g',marker='D',label=speciesName[2])
ax.legend()
ax.set_xlabel(iris.feature_names[2])
ax.set_ylabel(iris.feature_names[3])
plt.title('Iris Data')
plt.savefig('IrisData.png',dpi=200)
plt.show()

iris_data = pd.DataFrame(data=iris.data,columns=iris.feature_names)
print(iris_data.head())
#增加常数列1，用来适配Δwj0的更新
tmp = pd.Series(np.ones(iris_data.shape[0]),name='ones')
iris_data = pd.concat([tmp,iris_data],join='outer',axis=1)
print(iris_data.head())

#标准化
#iris_data = (iris_data-iris_data.mean())/iris_data.std()
#print(iris_data.head())

target = pd.Series(data=iris.target)
#将数据随机分离为测试数据和训练数据，其中测试数据占30%
x_train,x_test,y_train,y_test = train_test_split(iris_data,target,test_size=0.3)
x_train = np.matrix(x_train.values)
y_train = np.matrix(y_train.values).T  #转化为列向量
x_test = np.matrix(x_test.values)
y_test = np.matrix(y_test.values).T
print(x_train.shape)
print(y_train.shape)
#获得种类的数量
K = iris.target_names.shape[0]
print(K)
def sigmod(z):
    return 1/(1+np.exp(-z))
#计算代价函数
def computerCost(theta,x,y):
    theta = np.matrix(theta)
    first = np.multiply(-y,np.log(sigmod(x*(theta.T))))
    second = np.multiply(1-y,np.log(1-sigmod(x*theta.T)))
    return np.sum(first-second)/x.shape[0]
#梯度
def gradient(theta,x,y):
    theta = np.matrix(theta)
    error = sigmod(x*theta.T) - y
    inner_cost = (error.T * x)/x.shape[0]
    return inner_cost
#法1：使用逻辑斯谛判别式算法来进行梯度下降
def softmax(all_theta,x,t):
    # all_theta = np.matrix(all_theta)
    # oi = all_theta*x.T   #为K*N维矩阵，K为种类,N为样本数
    # #print(oi.shape)
    # #print(oi[:,0]/np.sum(oi[:,0]))
    # yi = np.zeros((oi.shape))
    # yi = np.matrix(yi)
    # for i in range(x.shape[0]):
    #     yi[:,i] = oi[:,i]/np.sum(oi[:,i])
    # return yi
    # ret = []
    # for i in range(K):
    #     tmp = all_theta
    yi = []
    oi = []
    for i in range(K):
        tmp = 0
        for j in range(x.shape[1]):
            tmp += all_theta[i,j]*x[t,j]
        oi.append(tmp)
    for i in range(K):
        yi.append(np.exp(oi[i])/np.sum(np.exp(oi)))
    return yi

import random 
def gradientDescent(x,y,K,learningRate=0.01,epoch=500):
    all_theta = np.zeros((K,x.shape[1]))
    grad = np.zeros((K,x.shape[1]))
    cost_list = []
    #生成范围在[-0.01,0.01]范围内的theta
    for i in range(K):
        for j in range(x.shape[1]):
            all_theta[i,j] = random.uniform(-0.01,0.01)
    #print(all_theta)
    for _ in range(epoch):
        for i in range(K):
            for j in range(x.shape[1]):
                grad[i,j] = 0
        cost = 0   #计算损失
        for t in range(x.shape[0]):
            yi = softmax(all_theta,x,t)
            for i in range(K):
                for j in range(x.shape[1]):
                    tmp = 0
                    if y[t,0] == i:
                        tmp = 1
                    grad[i,j] = grad[i,j] + (tmp-yi[i])*x[t,j]
                    cost += -tmp*np.log(yi[i])
        for i in range(K):
            for j in range(x.shape[1]):
                all_theta[i,j] = all_theta[i,j] + learningRate*grad[i,j]
        #y_i = [1 if label==i else 0 for label in y]
        cost_list.append(cost)
    return all_theta,cost_list

#法2：使用minimize函数的TNC来梯度下降优化
from scipy.optimize import minimize
def one_vs_rest(x,y,K):
    all_theta = np.zeros((K,x.shape[1]))
    for i in range(0,K):
        theta = np.zeros(x.shape[1])
        y_i = np.array([1 if label==i else 0 for label in y])
        y_i = np.reshape(y_i,(x.shape[0],1))
        #自动拟合梯度下降
        ret = minimize(fun=computerCost,x0=theta,args=(x,y_i),method='TNC',
        jac=gradient,options={'disp':True})
        #更新最终参数
        all_theta[i,:] = ret.x
    return all_theta
def predict_all(x,all_theta):
    h = sigmod(x * all_theta.T)
    h_argmax = np.argmax(h,axis=1)
    return h_argmax
def computeAccuracy(all_theta,x_test,y_test):
    y_test_res = predict_all(x_test,all_theta)
    correct = [1 if a==b else 0 for (a,b) in zip(y_test_res,y_test)]
    accuracy = (sum(map(int,correct))) / float(len(correct))
    print("accuracy= %f%%" %(accuracy*100))
    
all_theta1 = one_vs_rest(x_train,y_train,3)
computeAccuracy(all_theta1,x_test,y_test)


all_theta2,cost_list = gradientDescent(x_train,y_train,3,0.0003,1000)
computeAccuracy(all_theta2,x_test,y_test)

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(1000),cost_list,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
plt.savefig('IrisCost.png',dpi=200)
plt.show()

