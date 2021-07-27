import pandas as pd
import sklearn as sk
import string
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import json
np.random.seed(333)


class Dataset():
    def __init__(self):
        self.dataset=pd.DataFrame()

        self.X=pd.DataFrame()
        self.Y=pd.DataFrame()
        self.Y0=None
    def load(self,path):
        self.dataset = pd.read_csv(path)
        self.Y = self.dataset.iloc[:, 0]
        self.X = self.dataset.iloc[:, 1:15]
        self.Y0 = list(set(self.Y))
    def get_train_test(self):
        encoder = LabelEncoder()
        Y = encoder.fit_transform(self.Y)
        X_train, X_test, y_train, y_test = train_test_split(self.X, Y, test_size=0.2, random_state=0, stratify=Y)
        X_train=X_train.values.tolist()
        X_test=X_test.values.tolist()
        return (X_train, X_test, y_train, y_test)
    def get_names_of_class(self):
        return self.Y0

class Bayes():
    def __init__(self):
        self.frequency_table=[]
        self.likelihood_table=[]
        self.answers=0

    def fit(self,X,Y):
        self.answers=list(set(Y))
        for xi in range(len(X[0])):
            self.frequency_table.append(dict())
            self.likelihood_table.append(dict())
        for x in X:
            for xi in range(len(x)):
                if x[xi] in self.frequency_table[xi]:
                    continue
                else:
                    self.frequency_table[xi][x[xi]]= [1 for x in self.answers]
                    self.likelihood_table[xi][x[xi]] = [0 for x in self.answers]

        for i in range(len(X)):
            for y in range(len(X[i])):
                self.frequency_table[y][X[i][y]][Y[i]]+=1

        for i in range(len(self.frequency_table)):
            for x in self.frequency_table[i]:
                total=sum(self.frequency_table[i][x])
                for c in range(len(self.frequency_table[i][x])):
                    self.likelihood_table[i][x][c]=self.frequency_table[i][x][c]/total

    def predict(self, inp):
        res1=[0 for x in range(2)]
        for i in range(len(res1)):
            for y in range(len(inp)):
                if res1[i]==0:
                    res1[i] = self.likelihood_table[y][inp[y]][i]
                else:
                    res1[i] *= self.likelihood_table[y][inp[y]][i]

        return np.argmax(res1)

    def test(self,X,Y):
        self.answers = list(set(Y))
        good=0
        all=0
        for x in range(len(X)):
            if self.predict(X[x])==Y[x]:
                good+=1
            all+=1
        accuracy=good/all
        return accuracy
    def load(self,path):
        with open(path+".json", "r") as read_file:
            self.likelihood_table = json.load(read_file)


    def save(self,name):
        with open(name+".json", 'w') as file:
            file.write(json.dumps(self.likelihood_table, indent=4))


if __name__ == '__main__':
    dataset=Dataset()
    dataset.load("mushrooms.csv")
    all=dataset.get_train_test()
    X_train=all[0]
    X_test=all[1]
    y_train=all[2]
    y_test=all[3]
    Y1=dataset.get_names_of_class()
    model=Bayes()
    model.fit(X_train,y_train)
    model.save("model")
    y_pred = [model.predict(x) for x in X_test]
    data="b,s,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,n,m".split(',')
    print(model.predict(data))

    from sklearn.metrics import classification_report, confusion_matrix

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))



