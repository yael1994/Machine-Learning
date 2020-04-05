import numpy as np

class pa:
    # constructor
    def __init__(self, data_x, data_y, epochs):
        self.train_x = data_x
        self.train_y = data_y
        self.epochs = epochs
        self.w = np.zeros((3, self.train_x.shape[1]))

    def train(self):
        for t in range(self.epochs):
            np.random.shuffle(list(zip(self.train_x, self.train_y)))
            for x, y in zip(self.train_x, self.train_y):
                # predict
                y_hat = np.argmax(np.dot(self.w, x))
                loss=max(0, 1 - np.dot(self.w[y], x) + np.dot(self.w[y_hat], x))
                teta = loss / (2*np.linalg.norm(x) ** 2)
                # update
                if loss!=0:
                    self.w[y, :] =  self.w[y, :] + teta * x
                    self.w[y_hat, :] = self.w[y_hat, :] - teta * x
        return self.w