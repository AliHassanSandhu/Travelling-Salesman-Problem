import numpy as np
import pandas as pd



df = pd.read_csv("data.csv",names=["x","y"])

number_of_cities = 10
population_size = 20
mutation_rate = 0.1
no_of_generations =  10000


class TSP:
    def __init__(self,data,population_size,mutation_rate,no_of_generations,number_of_cities):
        self.data = data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.no_of_generations = no_of_generations
        self.no_of_cities = number_of_cities 

    def prepare_data(self):
        data = self.data[:self.no_of_cities]
        index = [i for i in range(self.no_of_cities)]
        data.insert(loc=0,
        column = "index",
        value = index
        )

        return data
        
    @staticmethod
    def euclidean_distance(x,y):
        return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5
    
    def prepare_distance_matrix(self):
        data  = self.prepare_data()
        n = self.no_of_cities
        self.distance_matrix = np.zeros((n,n))

        for i in range(n):
            for j in range(i + 1, n):

                city_1 = data.iloc[i][["x", "y"]].values
                city_2 = data.iloc[j][["x", "y"]].values

                distance = self.euclidean_distance(city_1, city_2)

                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance

a = TSP(df,population_size,mutation_rate,no_of_generations,number_of_cities)
a.prepare_data()
a.prepare_distance_matrix()
print(a.distance_matrix)