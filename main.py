import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class AsymmetricTSP:
    def __init__(self, number_of_cities, population_size, mutation_rate, no_of_generations, connectivity_rate=0.8):
        self.no_of_cities = number_of_cities 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.no_of_generations = no_of_generations
        self.connectivity_rate = connectivity_rate
        self.df = self.generate_city_data()
        self.prepare_distance_matrix()

    def generate_city_data(self):
        
        rng = np.random.default_rng(seed=42)
        df = pd.DataFrame({
            "index": np.arange(self.no_of_cities),
            "x": rng.integers(10, 100, size=self.no_of_cities),
            "y": rng.integers(10, 100, size=self.no_of_cities)
        })
        return df

    @staticmethod
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def prepare_distance_matrix(self):
        n = self.no_of_cities
        self.distance_matrix = np.zeros((n, n), dtype=float)
        rng = np.random.default_rng(seed=42)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue 

                if rng.random() > self.connectivity_rate:
                    self.distance_matrix[i][j] = np.inf
                    continue

                city_1 = self.df.iloc[i][["x", "y"]].values
                city_2 = self.df.iloc[j][["x", "y"]].values
                
                base_dist = self.euclidean_distance(city_1, city_2)

                asymmetric_factor = rng.uniform(0.8, 1.5)
                
                self.distance_matrix[i][j] = round(base_dist * asymmetric_factor, 1)

    def create_population(self):
       
        population = np.zeros((self.population_size, self.no_of_cities + 1), dtype=int)
        intermediate_cities = np.arange(1, self.no_of_cities)
        rng = np.random.default_rng(seed=42)
        
        for i in range(self.population_size):
            population[i, 1:-1] = rng.permutation(intermediate_cities)

        self.population = population
        return self.population

    def plot_graph(self):

        G = nx.DiGraph()
        n = self.no_of_cities

        positions = {}
        for _, row in self.df.iterrows():
            city_id = int(row["index"])
            G.add_node(city_id)
            positions[city_id] = (row["x"], row["y"])

        
        for i in range(n):
            for j in range(n):
                if i != j and not np.isinf(self.distance_matrix[i][j]):
                    G.add_edge(i, j, weight=self.distance_matrix[i][j])

        plt.figure(figsize=(11, 8))

        
        nx.draw_networkx_nodes(G, positions, node_color='skyblue', node_size=800)
        nx.draw_networkx_labels(G, positions, font_size=12, font_weight='bold')

        
        nx.draw_networkx_edges(
            G, positions, 
            edge_color='#4A4A4A', 
            alpha=0.75, 
            arrows=True, 
            arrowsize=22, 
            arrowstyle='-|>', 
            connectionstyle="arc3,rad=0.15",
            min_source_margin=18, 
            min_target_margin=18
        )

        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, positions, 
            edge_labels=edge_labels, 
            font_size=8, 
            label_pos=0.3,  
            rotate=False
        )

        plt.title("Asymmetric TSP Graph (Clear Directed Paths)", fontsize=14, pad=15)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('on')
        plt.tight_layout()
        plt.show()


# Execution Parameters
number_of_cities = 5
population_size = 10
mutation_rate = 0.1
no_of_generations = 10000

tsp = AsymmetricTSP(number_of_cities, population_size, mutation_rate, no_of_generations, connectivity_rate=0.8)

print("--- Coordinates DataFrame ---")
print(tsp.df)

print("\n--- Asymmetric Distance Matrix ---")
print(tsp.distance_matrix)


tsp.plot_graph()