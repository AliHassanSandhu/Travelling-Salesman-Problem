import numpy as np
import atsp_read


class DynamicTSP:

    def __init__(
        self,
        number_of_cities,
        tsp_file="usa13509.tsp"
    ):

        self.number_of_cities = number_of_cities

        self.distance_matrix = atsp_read.read_tsplib(
            tsp_file,
            number_of_cities
        )

    def run(self):

        n = self.number_of_cities

        # Cities excluding the starting city 0
        cities = list(range(1, n))

        # --------------------------------------------------
        # dp[(subset, last_city)]
        #
        # Minimum cost to start at city 0,
        # visit all cities in subset,
        # and finish at last_city.
        # --------------------------------------------------

        dp = {}
        parent = {}

        # --------------------------------------------------
        # Base cases
        #
        # 0 -> city
        # --------------------------------------------------

        for city in cities:

            subset = frozenset([city])

            dp[(subset, city)] = (
                self.distance_matrix[0, city]
            )

            parent[(subset, city)] = 0

        # --------------------------------------------------
        # Build larger subsets
        # --------------------------------------------------

        for subset_size in range(2, n):

            for subset in self.generate_subsets(
                cities,
                subset_size
            ):

                for current_city in subset:

                    previous_cities = (
                        subset - {current_city}
                    )

                    best_cost = np.inf
                    best_previous = None

                    # Try every possible previous city
                    for previous_city in previous_cities:

                        previous_cost = dp[
                            (
                                previous_cities,
                                previous_city
                            )
                        ]

                        edge_cost = self.distance_matrix[
                            previous_city,
                            current_city
                        ]

                        candidate_cost = (
                            previous_cost
                            + edge_cost
                        )

                        if candidate_cost < best_cost:

                            best_cost = candidate_cost
                            best_previous = previous_city

                    dp[
                        (
                            subset,
                            current_city
                        )
                    ] = best_cost

                    parent[
                        (
                            subset,
                            current_city
                        )
                    ] = best_previous

        # --------------------------------------------------
        # Complete tour
        #
        # Find the best city to return to city 0 from.
        # --------------------------------------------------

        full_set = frozenset(cities)

        best_cost = np.inf
        best_last_city = None

        for last_city in cities:

            path_cost = dp[
                (
                    full_set,
                    last_city
                )
            ]

            return_cost = self.distance_matrix[
                last_city,
                0
            ]

            total_cost = (
                path_cost
                + return_cost
            )

            if total_cost < best_cost:

                best_cost = total_cost
                best_last_city = last_city

        # --------------------------------------------------
        # Reconstruct optimal route
        # --------------------------------------------------

        route = self.reconstruct_route(
            parent,
            full_set,
            best_last_city
        )

        return (
            np.array(route),
            best_cost
        )

    def generate_subsets(
        self,
        cities,
        size
    ):

        """
        Generate all subsets of a given size.
        """

        if size == 0:

            yield frozenset()

            return

        if len(cities) < size:

            return

        if size == 1:

            for city in cities:

                yield frozenset([city])

            return

        for i in range(len(cities)):

            current_city = cities[i]

            remaining_cities = cities[
                i + 1:
            ]

            for subset in self.generate_subsets(
                remaining_cities,
                size - 1
            ):

                yield frozenset(
                    {current_city} | subset
                )

    def reconstruct_route(
        self,
        parent,
        full_set,
        last_city
    ):

        route = []

        current_city = last_city
        subset = full_set

        while current_city != 0:

            route.append(current_city)

            previous_city = parent[
                (
                    subset,
                    current_city
                )
            ]

            subset = subset - {
                current_city
            }

            current_city = previous_city

        route.reverse()

        return (
            [0]
            + route
            + [0]
        )