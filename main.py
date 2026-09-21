from networkx.generators import spectral_graph_forge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class AsymmetricTSP:
    def __init__(self, number_of_cities, population_size, mutation_rate, no_of_generations, connectivity_rate=0.8, seed=42):
        self.no_of_cities = number_of_cities 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.no_of_generations = no_of_generations
        self.connectivity_rate = connectivity_rate
        
        self.rng = np.random.default_rng(seed=seed)
        
        self.df = self.generate_city_data()
        self.prepare_distance_matrix()
        #self.prepare_symmetric_distance_matrix()

    def generate_city_data(self):
        df = pd.DataFrame({
            "index": np.arange(self.no_of_cities),
            "x": self.rng.integers(10, 100, size=self.no_of_cities),
            "y": self.rng.integers(10, 100, size=self.no_of_cities)
        })
        return df

    @staticmethod
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def prepare_symmetric_distance_matrix(self):
        n = self.no_of_cities
        self.distance_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):  # Only calculate upper triangle
                city_1 = self.df.iloc[i][["x", "y"]].values
                city_2 = self.df.iloc[j][["x", "y"]].values
                
                base_dist = self.euclidean_distance(city_1, city_2)
                
                # Optional asymmetric factor omitted for pure symmetry
                dist = round(base_dist, 1)
                
                # Mirror the distance to make it symmetric
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist


    def prepare_distance_matrix(self):
        n = self.no_of_cities
        self.distance_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue 

                if self.rng.random() > self.connectivity_rate:
                    self.distance_matrix[i][j] = np.inf
                    continue

                city_1 = self.df.iloc[i][["x", "y"]].values
                city_2 = self.df.iloc[j][["x", "y"]].values
                
                base_dist = self.euclidean_distance(city_1, city_2)
                asymmetric_factor = self.rng.uniform(0.8, 1.5)
                
                self.distance_matrix[i][j] = round(base_dist * asymmetric_factor, 1)

    def create_population(self):
        """Generates a population of UNIQUE routes (may include inf costs)."""
        intermediate_cities = np.arange(1, self.no_of_cities)
        unique_routes = set()
        population_list = []

        while len(population_list) < self.population_size:
            perm = self.rng.permutation(intermediate_cities)
            route = np.concatenate(([0], perm, [0]))
            
            # Convert to tuple to check uniqueness
            route_tuple = tuple(route)

            if route_tuple not in unique_routes:
                unique_routes.add(route_tuple)
                population_list.append(route)

        self.population = np.array(population_list, dtype=int)
        return self.population

    def calculate_route_cost(self, route: list or np.ndarray):
        """Calculates total cost using an explicit for loop."""
        cost = 0.0
        for i in range(self.no_of_cities):
            from_city = route[i]
            to_city = route[i + 1]
            cost += self.distance_matrix[from_city][to_city]

        return cost     

    def population_cost(self):
        """Prints route and cost for each individual in the population."""
        print("\n--- Population Cost Summary ---")
        for idx, individual in enumerate(self.population):
            cost = self.calculate_route_cost(individual)
            print(f"Individual {idx:2d} | Route: {individual} | Cost: {cost}")

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
number_of_cities = 20
population_size = 50
mutation_rate = 0.1
no_of_generations = 10000

tsp = AsymmetricTSP(number_of_cities, population_size, mutation_rate, no_of_generations, connectivity_rate=0.98)

print("--- Coordinates DataFrame ---")
print(tsp.df)

print("\n--- Asymmetric Distance Matrix ---")
print(tsp.distance_matrix)

print(tsp.create_population())
tsp.population_cost()

#tsp.plot_graph()