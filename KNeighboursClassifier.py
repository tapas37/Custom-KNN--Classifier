import numpy as np
from collections import Counter
class knn:

    def __init__(self,k=5):
        self.n_neighbours=k
        self.X_train=None
        self.y_train=None

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train

    def predict(self,X_test):
        y_pred=[]



        for i in X_test:
            # calculate distance with each training point
            distance=[]
            for j in self.X_train:
                distance.append(self.calculate_distance(i,j))
            # enumerate will show with which training point we find the distance of x test
            n_neighbours=sorted(list(enumerate(distance)),key=lambda x:x[1])[0:self.n_neighbours]
            label=self.majority_count(n_neighbours)
            y_pred.append(label)

        return np.array(y_pred)

    def calculate_distance(self,point_A,point_B):
        return np.linalg.norm(point_A-point_B)
        # no matter how many dimension e.g -[4,5,6][7,8,9] or mort than that it find
        #  euclidian distance between it

    def majority_count(self,neighbours):
        votes=[]
        for i in neighbours:
            # print(i,self.y_train[i[0]])--> here you will get the value of 238th of y train e.g (238, 0.058729713167986475) 0
            votes.append(self.y_train[i[0]])

        # finding most common element in a list
        votes=Counter(votes)
        return votes.most_common()[0][0]






