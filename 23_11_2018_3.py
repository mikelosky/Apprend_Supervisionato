import numpy as np
import pandas as pd

class L_R_M:
    def __init__(self,candy):
        candy_clean = self.elimination(candy)
        candy_x = candy_clean.iloc[:, 0:10]
        candy_y = candy_clean.iloc[:, 10:13]
        candy_x_norm = self.normalization(candy_x)

        theta_all, theta_history, cost_history = self.l_r_m(candy_x, candy_y)
        self.test(theta_all)

    def elimination(self,candy):
        #trattamento dei dati elinimare entry con dati nulli ed eliminare i duplicati
        candy=candy.drop_duplicates()
        candy=candy.dropna()
        return candy

    def normalization(self,candy):
        #nomalizzazione dei dati con z-score
        prova=candy.iloc[:,7:10]
        candy.iloc[:,7:10]=(candy.iloc[:,7:10]-candy.iloc[:,7:10].mean())/candy.iloc[:,7:10].std()
        return candy

    def l_r_m(self,candy_x,candy_y):
        alpha=0.5
        iteration=1000

        candy_1=np.append(np.ones((len(candy_x),1)),candy_x,axis=1)
        lenght=len(candy_1[0])
        candy_1=np.matrix(candy_1)
        candy_y=np.matrix(candy_y)

        prova=candy_y.shape[1]
        theta=np.matrix(np.random.rand(lenght,1))
        theta_all=[]
        print(theta)
        cost_history=np.zeros(iteration)
        theta_history=[]
        for y in range(0,candy_y.shape[1]):
            for i in range(0,iteration):
                z=np.dot(candy_1,theta)
                sig=self.sigmoide(z)
                temp=candy_y[:,y]-sig
                gradient=np.dot(temp.T,candy_1)
                theta=theta-((alpha/len(candy_y))*gradient).T
                theta_history.append(theta)
                #cost_history[i]=
            theta_all.append(theta)
            theta = np.matrix(np.random.rand(lenght, 1))

        return theta_all,theta_history,cost_history

    def sigmoide(self,z):
        z_p = 1/(1+np.exp(-z))
        return z_p

    def prediction(self,array,theta_all):
        result=[]
        test=len(theta_all)
        for i in range(0,len(theta_all)):
            prova=np.dot(array, theta_all[i])
            prova1=self.sigmoide(np.dot(array, theta_all[i]))
            thetha=theta_all[i]
            if self.sigmoide(np.dot(array, theta_all[i]))>=0.5:
                result.append(1)
            else:
                result.append(0)
        return result


    def test(self,theta_all):
        array_test1=[1,0,0,0,0,0,0,1,0.31299999,0.31299999,44.375519]
        array_test2=[1,1,0,0,0,1,0,0,0.186,0.26699999,41.904308]
        array_test3=[1,0,0,0,1,0,0,1,0.87199998,0.84799999,49.524113]


        print("I rusultati sul primo array sono: "+ str(self.prediction(array_test1,theta_all)))
        print("I rusultati sul secondo array sono: " + str(self.prediction(array_test2,theta_all)))
        print("I rusultati sul terzo array sono: " + str(self.prediction(array_test3,theta_all)))

if __name__=="__main__":
    candy=pd.read_csv("candy_2.csv",names=["caramel", "peanutyalmondy", "nougat", "crispedricewafer", "hard", "bar", "pluribus",
                                                "sugarpercent", "pricepercent", "winpercent","cioccolato","frutta","altro" ])
    L_R_M(candy)