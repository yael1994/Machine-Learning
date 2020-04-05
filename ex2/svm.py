import numpy as np

class svm:
    #constructor
    def __init__(self, data_x, data_y, epochs, lamda, eta):
        self.train_x = data_x
        self.train_y = data_y
        self.epochs = epochs
        self.w = np.zeros((3, self.train_x.shape[1]))
        self.eta = eta #Learning rate
        self.lamda = lamda

    def train(self):
        for t in range(self.epochs):
            np.random.shuffle(list(zip(self.train_x,self.train_y)))
            for x,y in zip(self.train_x,self.train_y):
                # predict
                y_hat = np.argmax(np.dot(self.w, x))
                loss=max(1-np.dot(self.w[y,:],x)+np.dot(self.w[y_hat,:],x),0)
                # update
                if loss != 0:
                    self.w[y, :] = (1-self.lamda*self.eta)*self.w[y, :] + self.eta * x
                    self.w[y_hat, :] = (1-self.lamda*self.eta)*self.w[y_hat, :] - self.eta * x
                    for r in range(3):
                        if r!=y and r!=y_hat:
                            self.w[r,:]=(1-self.lamda*self.eta)*self.w[r, :]
                            break
        return self.w