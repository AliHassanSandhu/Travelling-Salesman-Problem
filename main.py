import external_input as ex_input
import atsp_read  
from networkx.generators import spectral_graph_forge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys


class AsymmetricTSP:    
    @ex_input.LogExecutionTime
    def __init__(self, number_of_cities, population_size, mutation_rate, no_of_generations,generaton_gap=1,elitism=2, connectivity_rate=0.8, seed=42):
        self.no_of_cities = number_of_cities 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.no_of_generations = no_of_generations
        self.connectivity_rate = connectivity_rate
        self.elitism = elitism
        self.generation_gap = generaton_gap
        self.rng = np.random.default_rng(seed=seed)

        #self.prepare_distance_matrix()
        #self.prepare_symmetric_distance_matrix()
        self.distance_matrix = atsp_read.read_tsplib("usa13509.tsp", self.no_of_cities)
        self.is_symmetric = np.array_equal(self.distance_matrix, self.distance_matrix.T)

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

    
    def calculate_fitness(self,route: list or np.ndarray):

        valid_cost = 0
        inf_count = 0
        for  i in range(self.no_of_cities):
            edge_cost = self.distance_matrix[route[i]][route[i+1]]
            if np.isinf(edge_cost):
                inf_count += 1
                valid_cost += 500
            else:
                valid_cost += edge_cost


        fitness = 1.0/(valid_cost+inf_count)
        return fitness, inf_count            

    def tournament_selection(self, k=3):
        selected_samples = self.rng.choice(self.population,size=k,replace=False)
        print(f"selected Samples:\n {selected_samples}")
        print("\n")
        
        candidates = []
        for i in selected_samples:
            candidates.append(self.calculate_fitness(i)[0])

        print(candidates)    

        max, idx = 0,None
        for index,value in enumerate(candidates):
            if max < value:
                max = value
                idx = index

        return np.array(selected_samples[idx])        

    @ex_input.LogExecutionTime
    def select_parents(self,k=3):
        parent1 = self.tournament_selection(k)
        parent2 = self.tournament_selection(k)

        while np.array_equal(parent1,parent2):
            parent2 = self.tournament_selection(k)

        return parent1,parent2

    def ordered_crossover(self, parent1, parent2):
        """Ordered Crossover (OX). City 0 stays fixed at start/end, crossover happens on the intermediate cities."""
        p1 = parent1[1:-1]
        p2 = parent2[1:-1]
        size = len(p1)

        cut1, cut2 = np.sort(self.rng.choice(size, size=2, replace=False))

        def make_child(segment_parent, order_parent):
            child = np.full(size, -1, dtype=int)
            child[cut1:cut2+1] = segment_parent[cut1:cut2+1]
            used = set(child[cut1:cut2+1])

            # Fill remaining positions (after cut2, wrapping around) with the other parent's order
            order = np.concatenate((order_parent[cut2+1:], order_parent[:cut2+1]))
            fill = [city for city in order if city not in used]

            positions = list(range(cut2+1, size)) + list(range(0, cut1))
            for pos, city in zip(positions, fill):
                child[pos] = city

            return np.concatenate(([0], child, [0]))

        child1 = make_child(p1, p2)
        child2 = make_child(p2, p1)

        return child1, child2

    def swap_mutation(self, route):
        """Swaps two random intermediate cities with probability mutation_rate."""
        route = route.copy()
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.choice(np.arange(1, self.no_of_cities), size=2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    def inversion_mutation(self, route):
        """Reverses a random segment of intermediate cities with probability mutation_rate."""
        route = route.copy()
        if self.rng.random() < self.mutation_rate:
            i, j = np.sort(self.rng.choice(np.arange(1, self.no_of_cities), size=2, replace=False))
            route[i:j+1] = route[i:j+1][::-1]
        return route

    def insertion_mutation(self, route):
        """Removes a random intermediate city and reinserts it at another position with probability mutation_rate."""
        route = route.copy()
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.choice(np.arange(1, self.no_of_cities), size=2, replace=False)
            city = route[i]
            route = np.delete(route, i)
            route = np.insert(route, j, city)
        return route

    def mutate(self, route):
        """Applies a randomly chosen mutation operator.

        Symmetric instances: swap, inversion or insertion.
        Asymmetric instances: swap or insertion (inversion flips edge
        directions, which is too disruptive for asymmetric costs).
        """
        if self.is_symmetric:
            operators = [self.swap_mutation, self.inversion_mutation, self.insertion_mutation]
        else:
            operators = [self.swap_mutation, self.insertion_mutation]

        operator = operators[self.rng.integers(len(operators))]
        return operator(route)

    def population_fitnes(self):
        """Prints route and cost for each individual in the population."""
        print("\n--- Fitness Summary ---")
        for idx, individual in enumerate(self.population):
            fitness, penalty = self.calculate_fitness(individual)
            print(f"Individual {idx:2d} | Route: {individual} | fitness: {fitness} | penalty: {penalty}")

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
arg_names = ['number_of_cities', 'population_size', 'mutation_rate', 'no_of_generations', 'elitism']
args = dict(zip(arg_names, ex_input.get_external_input([10, 10, 0.1,1000,2])))
number_of_cities = 10 if type(args['number_of_cities']) != int else  args['number_of_cities']
population_size = 10 if type(args['population_size']) != int else  args['population_size']
mutation_rate = 0.1 if type(args['mutation_rate']) != float else  args['mutation_rate']
no_of_generations = 10000 if type(args['no_of_generations']) != int else  args['no_of_generations']
elitism = 2 if type(args['elitism']) != int else  args['elitism']
generation_gap = 1

tsp = AsymmetricTSP(number_of_cities, population_size, mutation_rate, no_of_generations, connectivity_rate=0.8)


print("\n--- Asymmetric Distance Matrix ---")
print(tsp.distance_matrix)

print(tsp.create_population())
#tsp.population_fitnes()
parent1, parent2 = tsp.select_parents()
print(f"parent1: {parent1}")
print(f"parent2: {parent2}")

child1, child2 = tsp.ordered_crossover(parent1, parent2)
print("\n--- Ordered Crossover (OX) ---")
print(f"child1:  {child1}")
print(f"child2:  {child2}")

print("\n--- Mutation ---")
print(f"swap:      {tsp.swap_mutation(child1)}")
print(f"inversion: {tsp.inversion_mutation(child2)}")
print(f"insertion: {tsp.insertion_mutation(child1)}")
print(f"mutate:    {tsp.mutate(child2)}")

#tsp.plot_graph()