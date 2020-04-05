import numpy as np
import sys as sys
import Tools
from svm import svm
from perceptron import perceptron
from pa import pa

#read all the files
with open(sys.argv[1], 'r') as X:
    X_train = [[num for num in line.split(',')] for line in X]
with open(sys.argv[2], 'r') as Y:
    Y_train = [int(line) for line in Y]
with open(sys.argv[3], 'r') as Test_x:
    X_Test = [[num for num in line.split(',')] for line in Test_x]


#change the first column from letter to number and all the string to float
X_train=Tools.change_for_matrix(X_train)
X_Test=Tools.change_for_matrix(X_Test)
#normalization of the data
X_train = Tools.z_score_normalization(X_train)
#hyperparameters:
epoch=20
eta=0.001
lambda_=0.05

# run the perceptron algorithm
perceptron_p = perceptron(X_train, Y_train, epoch, eta)
w_perceptron=perceptron_p.train()
# run the svm algorithm
svm_s=svm(X_train,Y_train,epoch,lambda_,eta)
w_svm=svm_s.train()
# run the pa algorithm
pa_p=pa(X_train, Y_train, epoch)
w_pa=pa_p.train()



m=len(X_Test)
for t in range(0, m ):
    perceptron_yhat= np.argmax(np.dot(w_perceptron,X_Test[t]))
    svm_yhat=np.argmax(np.dot(w_svm,X_Test[t]))
    pa_yhat=np.argmax(np.dot(w_pa,X_Test[t]))
    print(f"perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}")
