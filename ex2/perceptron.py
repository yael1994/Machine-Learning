import numpy as np

class perceptron:
    #constructor
    def __init__(self, data_x, data_y, epochs, eta):
        self.train_x = data_x
        self.train_y = data_y
        self.epochs = epochs
        self.w = np.zeros((3, self.train_x.shape[1]))
        self.eta = eta #Learning rate

    def train(self):
        for t in range(self.epochs):
            np.random.shuffle(list(zip(self.train_x, self.train_y)))
            for x, y in zip(self.train_x, self.train_y):
                #predict
                y_hat=np.argmax(np.dot(self.w,x))
                #update
                if y!=y_hat:
                    self.w[y,:]=self.w[y,:]+self.eta*x
                    self.w[y_hat,:]=self.w[y_hat,:]-self.eta*x
            self.eta /= t + 1
        return self.w