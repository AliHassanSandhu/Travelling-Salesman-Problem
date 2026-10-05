import external_input as ex_input
import atsp_read  
from networkx.generators import spectral_graph_forge
import numpy as np
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

        self.is_symmetric = np.array_equal(
            self.distance_matrix,
            self.distance_matrix.T
        )



    def create_population(self):
        """Generates a population of UNIQUE routes."""

        intermediate_cities = np.arange(1, self.no_of_cities)

        unique_routes = set()
        population_list = []

        while len(population_list) < self.population_size:

            perm = self.rng.permutation(intermediate_cities)

            route = np.concatenate(
                ([0], perm, [0])
            )

            route_tuple = tuple(route)

            if route_tuple not in unique_routes:

                unique_routes.add(route_tuple)
                population_list.append(route)

        self.population = np.array(
            population_list,
            dtype=int
        )

    
    def calculate_fitness(self,route: list or np.ndarray):

        valid_cost = 0
        inf_count = 0

        for i in range(self.no_of_cities):

            edge_cost = self.distance_matrix[
                route[i]
            ][
                route[i + 1]
            ]

            if np.isinf(edge_cost):

                inf_count += 1
                valid_cost += 500

            else:

                valid_cost += edge_cost

        fitness = 1.0 / (valid_cost + inf_count)

        return fitness, inf_count


    def evaluate_population(self):
        """Calculates fitness for the entire population."""

        fitnesses = []
        penalties = []

        for individual in self.population:

            fitness, penalty = self.calculate_fitness(
                individual
            )

            fitnesses.append(fitness)
            penalties.append(penalty)

        return (
            np.array(fitnesses),
            np.array(penalties)
        )


    def tournament_selection(self, k=3):

        selected_samples = self.rng.choice(
            self.population,
            size=k,
            replace=False
        )

        candidates = []

        for individual in selected_samples:

            fitness = self.calculate_fitness(
                individual
            )[0]

            candidates.append(fitness)

        best_index = np.argmax(candidates)

        return np.array(
            selected_samples[best_index]
        )
    
    def select_parents(self,k=3):
        parent1 = self.tournament_selection(k)
        parent2 = self.tournament_selection(k)

        while np.array_equal(parent1, parent2):

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

    def ordered_crossover(self, parent1, parent2):
        """
        Ordered Crossover (OX).

        City 0 remains fixed at the beginning and end.
        Crossover is performed only on intermediate cities.
        """

        p1 = parent1[1:-1]
        p2 = parent2[1:-1]

        size = len(p1)

        cut1, cut2 = np.sort(
            self.rng.choice(
                size,
                size=2,
                replace=False
            )
        )

        def make_child(segment_parent, order_parent):

            child = np.full(
                size,
                -1,
                dtype=int
            )

            # Copy selected segment
            child[
                cut1:cut2 + 1
            ] = segment_parent[
                cut1:cut2 + 1
            ]

            used = set(
                child[
                    cut1:cut2 + 1
                ]
            )

            # Remaining cities from other parent
            order = np.concatenate(
                (
                    order_parent[cut2 + 1:],
                    order_parent[:cut2 + 1]
                )
            )

            fill = [
                city
                for city in order
                if city not in used
            ]

            # Positions to fill
            positions = (
                list(range(cut2 + 1, size))
                +
                list(range(0, cut1))
            )

            for pos, city in zip(
                positions,
                fill
            ):
                child[pos] = city

            # Add starting and ending city 0
            return np.concatenate(
                ([0], child, [0])
            )

        child1 = make_child(
            p1,
            p2
        )

        child2 = make_child(
            p2,
            p1
        )

        return child1, child2



    def swap_mutation(self, route):
        """
        Swap two random intermediate cities.
        """

        route = route.copy()

        if self.rng.random() < self.mutation_rate:

            i, j = self.rng.choice(
                np.arange(1, self.no_of_cities),
                size=2,
                replace=False
            )

            route[i], route[j] = (
                route[j],
                route[i]
            )

        return route


    def inversion_mutation(self, route):
        """
        Reverse a random segment of intermediate cities.
        """

        route = route.copy()

        if self.rng.random() < self.mutation_rate:

            i, j = np.sort(
                self.rng.choice(
                    np.arange(1, self.no_of_cities),
                    size=2,
                    replace=False
                )
            )

            route[i:j + 1] = (
                route[i:j + 1][::-1]
            )

        return route


    def insertion_mutation(self, route):
        """
        Remove one intermediate city and
        insert it at another position.
        """

        route = route.copy()

        if self.rng.random() < self.mutation_rate:

            i, j = self.rng.choice(
                np.arange(1, self.no_of_cities),
                size=2,
                replace=False
            )

            city = route[i]

            route = np.delete(
                route,
                i
            )

            route = np.insert(
                route,
                j,
                city
            )

        return route


    def mutate(self, route):
        """
        Randomly selects a mutation operator.

        Symmetric TSP:
            swap
            inversion
            insertion

        Asymmetric TSP:
            swap
            insertion
        """

        if self.is_symmetric:

            operators = [
                self.swap_mutation,
                self.inversion_mutation,
                self.insertion_mutation
            ]

        else:

            operators = [
                self.swap_mutation,
                self.insertion_mutation
            ]

        operator = operators[
            self.rng.integers(
                len(operators)
            )
        ]

        return operator(route)

    # Main Loop
    @ex_input.LogExecutionTime
    def run(self):

        """Main Genetic Algorithm execution loop."""

        self.create_population()

        best_fitness_history = []
        avg_fitness_history = []
        best_cost_history = []

        best_overall_individual = None
        best_overall_fitness = -1

        for gen in range(
            self.no_of_generations
        ):

    

            fitnesses, penalties = (
                self.evaluate_population()
            )

    

            sorted_indices = np.argsort(
                fitnesses
            )[::-1]

            current_best_fit = (
                fitnesses[
                    sorted_indices[0]
                ]
            )

            current_best_ind = (
                self.population[
                    sorted_indices[0]
                ]
            )

            

            if (
                current_best_fit
                >
                best_overall_fitness
            ):

                best_overall_fitness = (
                    current_best_fit
                )

                best_overall_individual = (
                    current_best_ind.copy()
                )

            # ----------------------------------------
            # Store metrics
            # ----------------------------------------

            best_fitness_history.append(
                current_best_fit
            )

            avg_fitness_history.append(
                np.mean(fitnesses)
            )

            best_cost_history.append(
                1.0 / current_best_fit
            )

          

            next_population = [
                self.population[idx].copy()
                for idx in sorted_indices[
                    :self.elitism
                ]
            ]

         

            while (
                len(next_population)
                <
                self.population_size
            ):

                # Selection
                parent1, parent2 = (
                    self.select_parents()
                )

                # Crossover
                child1, child2 = (
                    self.ordered_crossover(
                        parent1,
                        parent2
                    )
                )

                # Mutation
                child1 = self.mutate(
                    child1
                )

                child2 = self.mutate(
                    child2
                )

                # Add children
                next_population.append(
                    child1
                )

                if (
                    len(next_population)
                    <
                    self.population_size
                ):

                    next_population.append(
                        child2
                    )

           

            self.population = np.array(
                next_population
            )

        

            if (
                gen
                %
                max(
                    1,
                    self.no_of_generations // 10
                )
                == 0
            ):

                print(
                    f"Gen {gen:4d} | "
                    f"Best Cost: "
                    f"{1.0 / current_best_fit:.4f} | "
                    f"Best Fitness: "
                    f"{current_best_fit:.6f}"
                )

        return (
            best_overall_individual,
            best_overall_fitness,
            best_fitness_history,
            avg_fitness_history,
            best_cost_history
        )


   

    def plot_convergence(
        self,
        best_fitness_history,
        avg_fitness_history,
        best_cost_history
    ):

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(14, 5)
        )

        # Fitness plot
        ax1.plot(
            best_fitness_history,
            label="Best Fitness",
            linewidth=2
        )

        ax1.plot(
            avg_fitness_history,
            label="Average Fitness",
            linestyle="--"
        )

        ax1.set_title(
            "Fitness Convergence Over Generations"
        )

        ax1.set_xlabel(
            "Generation"
        )

        ax1.set_ylabel(
            "Fitness"
        )

        ax1.grid(True)

        ax1.legend()


        # Cost plot
        ax2.plot(
            best_cost_history,
            label="Best Route Cost",
            linewidth=2
        )

        ax2.set_title(
            "Best Route Cost Reduction"
        )

        ax2.set_xlabel(
            "Generation"
        )

        ax2.set_ylabel(
            "Cost"
        )

        ax2.grid(True)

        ax2.legend()

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
'''
tsp = AsymmetricTSP(number_of_cities, population_size, mutation_rate, no_of_generations, connectivity_rate=0.8)


number_of_cities = (
    args["number_of_cities"]
    if type(args["number_of_cities"]) == int
    else 10
)

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

mutation_rate = (
    args["mutation_rate"]
    if type(args["mutation_rate"]) == float
    else 0.1
)

no_of_generations = (
    args["no_of_generations"]
    if type(args["no_of_generations"]) == int
    else 1000
)

elitism = (
    args["elitism"]
    if type(args["elitism"]) == int
    else 2
)
'''



tsp = AsymmetricTSP(
    number_of_cities=number_of_cities,
    population_size=population_size,
    mutation_rate=mutation_rate,
    no_of_generations=no_of_generations,
    elitism=elitism
)




(
    best_route,
    best_fitness,
    best_fit_hist,
    avg_fit_hist,
    best_cost_hist
) = tsp.run()

ex_input.save_execution_timelog(args, f"Calculated lowest path cost {1.0 / best_fitness:.2f}\nOptimal Route Found:{best_route}", f"{args['number_of_cities']}CitiesSolve{args['no_of_generations']}.txt")

print(
    "\n================ Optimization Finished ================"
)

print(
    f"Optimal Route Found: {best_route}"
)

print(
    f"Best Fitness Value: {best_fitness:.6f}"
)

print(
    f"Calculated Path Cost: "
    f"{1.0 / best_fitness:.2f}"
)

tsp.plot_convergence(
    best_fit_hist,
    avg_fit_hist,
    best_cost_hist
)