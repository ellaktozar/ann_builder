import numpy as np

class MLPClassifier():
    def __init__(self, epochs=10000, early_stopping=False, learning_rate=10e-5):
        self.costs = []
        self.input_layer = None
        self.output_layer = None
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.has_weights = False

    def add(self, new_layer):
        if self.input_layer is None:
            self.input_layer = new_layer
            self.output_layer = new_layer
        else:
            self.output_layer.output_layer = new_layer
            new_layer.input_layer = self.output_layer
            self.output_layer = new_layer

    def fit(self, X, Y):
        self.output_layer.T = Y.copy()
        self.input_layer.Z = X.copy()
        if not self.has_weights:
            self.input_layer.output_shape = X.shape[1]
            self.input_layer.init_weights()
            self.has_weights = True

        for e in range(self.epochs):
            self.input_layer.output_layer._forward()
            cost = self._cost(
                self.output_layer.T,
                self.output_layer.Z
            )
            self.costs.append(cost)
            if self.early_stopping and (len(self.costs) > 2):
                if abs(self.costs[-1] - self.costs[-2]) <= 10e-6:
                    print('Early stopping')
                    print('Epoch:', e, 'Cost:', cost, 'Accuracy:', self.score(self.output_layer.T, self.output_layer.Z))
                    break
            if e % 100 == 0:
                print('Epoch:', e, 'Cost:', cost, 'Accuracy:', self.score(self.output_layer.T, self.output_layer.Z))

            self.output_layer._backprop()
            self.output_layer._update_weights(self.learning_rate)

    def _cost(self, T, Y):
        return (T * np.log(Y)).sum()

    def predict(self, X):
        self.input_layer.Z = X
        self.input_layer.output_layer._forward()
        return np.argmax(self.output_layer.Z, axis=1)


    def score(self, T, Y):
        return np.mean(np.argmax(T, axis=1) == np.argmax(Y, axis=1))
        

            