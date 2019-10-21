import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class R_L_M:

    def __init__(self, wine):
        print(wine)
        data = self.elimination(wine)
        len_ = len(data)
        len_c = len(data.columns)
        len_train = int((len_ * 80) / 100)
        len_test = int(len_ - len_train)
        self.xtrain = data.iloc[:len_train, 0:len_c - 1]
        self.xtest = data.iloc[len_train + 1:, 0:len_c - 1]
        self.ytrain = data.iloc[:len_train, len_c - 1:]
        self.ytest = data.iloc[len_train + 1:, len_c - 1:]

        regressor = LinearRegression()
        regressor.fit(self.xtrain, self.ytrain)  # training the algorithm
        # To retrieve the intercept:
        print(regressor.intercept_)
        # For retrieving the slope:
        thetask = regressor.coef_
        print(regressor.coef_)
        y_pred = regressor.predict(self.xtest)
        self.df = pd.DataFrame({'Actual': np.array(self.ytest).flatten(), 'Predicted': np.array(y_pred).flatten()})

        wine_norm = self.normalization(self.xtrain, "zscore")
        self.wine_test = self.normalization(self.xtest, "zscore")
        theta, theta_history, cost_history = self.m_l_r(wine_norm, self.ytrain)
        self.test(thetask)
        self.testtest(theta, self.wine_test, self.ytest)
        # self.new_feature(wine_x,wine_y)

    def elimination(self, wine):
        # verificare che non vi siano valori nulli o ripetuti del dataset

        wine_1 = wine.drop_duplicates()

        wine_2 = wine_1.dropna()

        return (wine_2)

    def normalization(self, wine, type):
        # applicachiamo la normalizzazione z-score che equivale a (x-medi)/std o la minmax
        if type == "zscore":
            wine_n = (wine - wine.mean()) / wine.std()
        else:
            wine_n = (wine - wine.min()) / (wine.max() - wine.min())
        return wine_n

    def m_l_r(self, wine, y):
        n_iter = 10000
        alpha = 0.02

        x_1 = np.append(np.ones((len(wine), 1)), wine, axis=1)
        x_len = len(x_1[0])
        x_1 = np.matrix(x_1)

        theta = np.matrix(np.random.randn(x_len, 1))
        cost_history = np.zeros(n_iter)
        theta_history = []
        y = np.matrix(y)
        for i in range(0, n_iter):
            error = np.dot(x_1, theta) - y
            delta = np.dot(x_1.T, error) / len(y)
            theta = (theta - (alpha * delta))
            theta_history.append(theta)
            #cost_history[i] = self.cost_function(x_1, y, theta)

        return theta, theta_history, cost_history

    def cost_function(self, x, y, theta):
        cost = (np.sum(np.square(np.dot(x, theta) - y.T))) / (2 * len(y))
        return cost

    def test(self, theta):
        array_test = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]
        array_test = self.normalization(np.array(array_test), "zscore")
        prediction = np.dot(theta, array_test)
        print("Predizione con l`array di test: " + str(prediction))

    def test2(self, theta):
        array_test = [1, 6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0, 2.7]
        prediction = np.dot(array_test, theta)
        print("Predizione con l`array di test2: " + str(prediction))

    def new_feature(self, wine_x, wine_y):
        # adding a new feature in the dataset
        new_feature = np.array(wine_x[["fixed acidity"]]) * np.array(wine_x[["pH"]])
        wine_new = np.append(wine_x, new_feature, axis=1)
        wine_norm = self.normalization(wine_new, "minmax")
        prova = len(wine_norm[0])
        theta, theta_history, cost_history = self.m_l_r(wine_norm, wine_y)
        self.test2(theta)

    def testtest(self, theta, xnp, ynp):
        diffsom = 0
        mones = np.ones((len(xnp), 1))
        x_1 = np.append(mones, xnp, axis=1)
        xnp = np.matrix(xnp)
        ynp = np.matrix(ynp)
        y_pre = np.dot(x_1, theta)
        self.dfmio = pd.DataFrame({'Actual': np.array(ynp).flatten(), 'Predicted': np.array(y_pre).flatten()})
        for i in range(0, len(xnp)):
            test1 = xnp[i]
            monesv = np.ones((1))
            test1 = np.append(monesv, test1)
            result = np.matmul(test1, theta)
            diff = ynp[i] - result
            print(str(i) + "Il valore che ci aspettiamo: " + str(ynp[i]) + " Invece abbiamo:" + str(
                result) + "con differenza: " + str(diff))
            diffsom += diff
        mediaerro = diffsom / len(xnp)
        print("La media di errore è : " + str(mediaerro))
        print("Errore quadratico medio : " + str(self.RMSE(theta, x_1, ynp)))
        print("Mean square error : " + str(self.MSE(theta, x_1, ynp)))
        print("Mean absolute error  : " + str(self.MAE(theta, x_1, ynp)))
        print("Coefficiente di determinazione : " + str(self.R2(theta, x_1, ynp)))

    def RMSE(self, theta, X, y):
        yp = np.dot(X, theta)
        loss = np.power((yp - y), 2)
        loss = np.sum(loss)
        rmse = (loss / len(y)) ** (0.5)
        return rmse

    def MSE(self, theta, X, y):
        yp = np.dot(X, theta)
        loss = np.power((yp - y), 2)
        loss = np.sum(loss)
        mse = loss / len(y)
        return mse

    def MAE(self, theta, X, y):
        yp = np.dot(X, theta)
        loss = (yp - y)
        loss = np.sum(loss)
        mae = loss / len(y)
        return mae

    def R2(self, theta, X, y):
        yp = np.dot(X, theta)
        yavg = np.mean(y)
        ess = np.sum(np.power((yp - yavg), 2))
        tss = np.sum(np.power((y - yavg), 2))
        r2 = ess / tss
        return r2


if __name__ == '__main__':
    wine = pd.read_csv("wine.csv", names=["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                                          "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                                          "density", "pH", "sulphates", "alcohol", "quality"])
    R_L_M(wine)
