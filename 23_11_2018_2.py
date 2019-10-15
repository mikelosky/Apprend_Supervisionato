import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class L_R:
    def __init__(self,candy):
        data=self.elimination(candy)

        len_ = len(data)
        len_c = len(data.columns)
        len_train = int((len_ * 80) / 100)
        len_test = int(len_ - len_train)
        self.xtrain = data.iloc[:len_train, 0:len_c - 1]
        self.xtest = data.iloc[len_train + 1:, 0:len_c - 1]
        self.ytrain = data.iloc[:len_train, len_c - 1:]
        self.ytest = data.iloc[len_train + 1:, len_c - 1:]

        # all parameters not specified are set to their defaults
        logisticRegr = LogisticRegression()
        logisticRegr.fit(self.xtrain, self.ytrain)
        predictions = logisticRegr.predict(self.xtest)
        # Use score method to get accuracy of model

        score = logisticRegr.score(self.xtest, self.ytest)
        print(score)
        thetask = logisticRegr.coef_
        candy_x_norm = self.normalization(self.xtest)
        candy_y = self.ytest
        theta,theta_history,cost_history=self.l_r(candy_x_norm,candy_y)
        self.test(thetask)


    def elimination(self,candy):
        #trattamento dei dati elinimare entry con dati nulli ed eliminare i duplicati
        candy=candy.drop_duplicates()
        candy=candy.dropna()
        return candy

    def normalization(self,candy):
        #nomalizzazione dei dati con z-score
        prova = candy
        candy=(candy-candy.mean())/candy.std()
        return candy

    def l_r(self,candy_x,candy_y):
        alpha=0.5
        iteration=1000

        candy_1=np.append(np.ones((len(candy_x),1)),candy_x,axis=1)
        lenght=len(candy_1[0])
        candy_1=np.matrix(candy_1)
        candy_y=np.matrix(candy_y)

        theta=np.matrix(np.random.rand(lenght,1))
        print(theta)
        cost_history=np.zeros(iteration)
        theta_history=[]
        for i in range(0,iteration):
            z=np.dot(candy_1,theta)
            sig=self.sigmoide(z)
            temp=candy_y-sig
            gradient=np.dot(temp.T,candy_1)
            theta=theta-((alpha/len(candy_y))*gradient).T
            theta_history.append(theta)
            #cost_history[i]=

        return theta,theta_history,cost_history

    def sigmoide(self,z):
        z_p = 1/(1+np.exp(-z))
        return z_p

    def prediction(self,pred):
        if self.sigmoide(pred)>=0.5:
            return 1
        else:
            return 0

    def test(self,theta):
        print(theta)
        array_test1=np.matrix([1, 0, 0, 0, 1, 0, 0, 0.186, 0.26699999, 41.904308])
        array_test2=np.matrix([0, 0, 0, 1, 0, 0, 1, 0.87199998, 0.84799999, 49.524113])
        array_test1 = self.normalization(array_test1)
        array_test2 = self.normalization(array_test2)
        array_test1 = self.sigmoide(array_test1)
        array_test2 = self.sigmoide(array_test2)

        result1=np.dot(array_test1,theta.T)
        result2=np.dot(array_test2,theta.T)

        print("I rusultati sul primo array sono: "+ str(self.prediction(result1)))
        print("I rusultati sul secondo array sono: " + str(self.prediction(result2)))


if __name__ == '__main__':
    candy = pd.read_csv("candy_1.csv",names=["caramel", "peanutyalmondy", "nougat", "crispedricewafer", "hard", "bar", "pluribus",
                                                "sugarpercent", "pricepercent", "winpercent","cioccolatino"])
    L_R(candy)
