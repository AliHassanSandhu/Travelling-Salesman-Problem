import atsp_read  
from networkx.generators import spectral_graph_forge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class AsymmetricTSP:
    def __init__(self, number_of_cities, population_size, mutation_rate, no_of_generations,generaton_gap=1,elitism=2, connectivity_rate=0.8, seed=42):
        self.no_of_cities = number_of_cities 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.no_of_generations = no_of_generations
        self.connectivity_rate = connectivity_rate
        self.elitism = elitism
        self.generation_gap = generaton_gap
        self.rng = np.random.default_rng(seed=seed)

        self.distance_matrix = atsp_read.read_tsplib("rbg403.atsp", self.no_of_cities)

    @staticmethod
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


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

    def select_parents(self,k=3):
        parent1 = self.tournament_selection(k)
        parent2 = self.tournament_selection(k)

        while np.array_equal(parent1,parent2):
            parent2 = self.tournament_selection(k)

        return parent1,parent2    

    def ordered_crossover(self,parent1,parent2):
        print(parent1)
        print(parent2)
        max_idx = len(parent1) - 1
        mid = self.no_of_cities//2
        three_forth = mid + self.no_of_cities//4
        cp1 = int(self.rng.choice([i for i in range(mid//2, mid) ]))
        cp2 = int(self.rng.choice([i for i in range(mid, three_forth)]))
        def make_child(p1, p2):
            child = np.zeros(len(p1), dtype=int)
            child[cp1:cp2+1] = p1[cp1:cp2+1]
            copied_set = set(p1[cp1:cp2+1])

            remaining = [city for city in p2 if city != 0 and city not in copied_set]
            rem_idx = 0
            for i in range(1, max_idx):
                if i < cp1 or i > cp2:
                    child[i] = remaining[rem_idx]
                    rem_idx += 1
                    
            return child

        child1 = make_child(parent1, parent2)
        child2 = make_child(parent2, parent1)
        return child1, child2   


    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.mutation_rate:
            max_idx = len(chromosome) - 1
            idx1, idx2 = self.rng.choice(range(1, max_idx), size=2, replace=False)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
            
        return chromosome



    def evaluate_population(self):
        """Calculates fitness and inverse cost for the entire population."""
        fitnesses = []
        penalties = []
        for ind in self.population:
            fit, pen = self.calculate_fitness(ind)
            fitnesses.append(fit)
            penalties.append(pen)
        return np.array(fitnesses), np.array(penalties)

    def run(self):
        """Main Genetic Algorithm Execution Loop."""
        self.create_population()

        best_fitness_history = []
        avg_fitness_history = []
        best_cost_history = []

        best_overall_individual = None
        best_overall_fitness = -1

        for gen in range(self.no_of_generations):
            fitnesses, penalties = self.evaluate_population()
            
            # Identify elitism champions
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            current_best_fit = fitnesses[sorted_indices[0]]
            current_best_ind = self.population[sorted_indices[0]]

            if current_best_fit > best_overall_fitness:
                best_overall_fitness = current_best_fit
                best_overall_individual = current_best_ind.copy()

            # Track metrics
            best_fitness_history.append(current_best_fit)
            avg_fitness_history.append(np.mean(fitnesses))
            best_cost_history.append(1.0 / current_best_fit)

            # Build next generation starting with elite individuals
            next_population = [self.population[idx].copy() for idx in sorted_indices[:self.elitism]]

            # Fill remaining population using crossover and mutation
            while len(next_population) < self.population_size:
                p1, p2 = self.select_parents()
                c1, c2 = self.ordered_crossover(p1, p2)
                
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                next_population.append(c1)
                if len(next_population) < self.population_size:
                    next_population.append(c2)

            self.population = np.array(next_population)

            if gen % max(1, self.no_of_generations // 10) == 0:
                print(f"Gen {gen:4d} | Best Cost Value: {1.0 / current_best_fit:.4f} | Best Fitness: {current_best_fit:.6f}")

        return best_overall_individual, best_overall_fitness, best_fitness_history, avg_fitness_history, best_cost_history

    def plot_convergence(self, best_fitness_history, avg_fitness_history, best_cost_history):
        """Plots convergence curves for fitness and overall cost over generations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Fitness Evolution Plot
        ax1.plot(best_fitness_history, label='Best Fitness', color='#1f77b4', linewidth=2)
        ax1.plot(avg_fitness_history, label='Avg Fitness', color='#ff7f0e', linestyle='--', alpha=0.8)
        ax1.set_title('Fitness Convergence Over Generations', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (1 / Total Penalty Cost)')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # Route Cost Plot
        ax2.plot(best_cost_history, label='Best Route Cost Value', color='#2ca02c', linewidth=2)
        ax2.set_title('Best Route Cost Reduction', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Cost / Penalty Score')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.show()


# Execution Parameters
number_of_cities = 20
population_size = 20
mutation_rate = 0.15
no_of_generations = 200
elitism = 2

tsp = AsymmetricTSP(
    number_of_cities=number_of_cities,
    population_size=population_size,
    mutation_rate=mutation_rate,
    no_of_generations=no_of_generations,
    elitism=elitism
)

# Run Optimization
best_route, best_fitness, best_fit_hist, avg_fit_hist, best_cost_hist = tsp.run()

print("\n================ Optimization Finished ================")
print(f"Optimal Route Found: {best_route}")
print(f"Best Fitness Value: {best_fitness:.6f}")
print(f"Calculated Path Cost: {1.0 / best_fitness:.2f}")

# Display Convergence Plots
tsp.plot_convergence(best_fit_hist, avg_fit_hist, best_cost_hist)