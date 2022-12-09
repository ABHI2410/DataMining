import sys
from random import randrange
import time 
# import tqdm 

class KMeanClassifier():
    def __init__(self,path,k,i):
        self.path = path
        self.points = []
        self.current_cluster ={}
        self.class_val = []
        self.clusters = k
        self.iterations = i
        self.centriods = {x:[0,0] for x in range(k)}
        self.error = {x:0 for x in range (i)}
    
    def distance(self,p1,p2):
        sum_of_square_differance = 0
        for i,j in zip(p1,p2):
            sum_of_square_differance += (i-j)**2
        return sum_of_square_differance**0.5
    
    def closest_centriod(self, point):
        min_distance = sys.maxsize
        centriod = None
        for key, value in self.centriods:
            dist = self.distance(value,point)
            if min_distance> dist:
                min_distance = dist
                centriod = key
        return centriod
    
    def get_data(self):
        self.points = []
        self.class_val = []
        with open (self.path) as file:
            for count, line in enumerate(file):
                point = line.split(" ")
                while("" in point):
                    point.remove("")
                self.points.append(point[:-1])
                self.class_val.append(int(point[-1].strip("\n")))
                self.closest_centriod[count] = randrange(0,self.clusters)
    




if __name__ == "__main__":
    path = sys.argv[1]
    k = int(sys.argv[2])
    i = int(sys.argv[3])
    kmean = KMeanClassifier(path, k, i)
    kmean.get_data()

