import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

class KNN:

    def __init__(self,k):
        self.mnist = datasets.fetch_openml('mnist_784')
        self.data,self.target = self.mnist.data, self.mnist.target
        self.indx = np.random.permutation(70000)
        self.classifier = KNeighborsClassifier(n_neighbors=k)

#    def mk_dataset(self,size):
#        train_img = [self.data[i] for i in self.indx[0:size]]
#        train_img = np.array(train_img)
#        train_target = [self.target[i] for i in self.indx[0:size]]
#        train_target = np.array(train_target)
#
#        return train_img, train_target
    
    def skl_knn(self):
        x1,y1 = self.data.loc[self.indx],self.target.loc[self.indx]
        x1.reset_index(drop=True, inplace=True)
        y1.reset_index(drop=True, inplace=True)
        x_train, x_test = x1[:60000], x1[60000:]
        y_train, y_test = y1[:60000], y1[60000:]
        self.classifier.fit(x_train,y_train)

        y_pred = self.classifier.predict(x_test)
        pickle.dump(self.classifier, open('knn.sav', 'wb'))
        print(classification_report(y_test,y_pred))
        print("KNN Classifier model saved as knn.sav!")