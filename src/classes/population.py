from .individual import Individual
from typing import Tuple, List
import logging
import json


class Population:

    def __init__(self, json_path: str):
        self.current_population: List[Individual] = []
        self.prev_populations: List[Tuple[List[Individual], float]] = []
        self.best_fitness: float = 0
        self.best_indiv: Individual = None
        self.state: int = 0  # Keep track of the state (evolution step)
        self.json_path = json_path

    def add_indiv(self, indiv: Individual):
        self.current_population.append(indiv)
        self._update()

    def _update(self):
        # Find the best indiv
        for indiv in self.current_population:
            if indiv.fitness > self.best_fitness:
                self.best_fitness = indiv.fitness
                self.best_indiv = indiv

        # Update json file
        self.save_to_json()

    def step_evolution(self, new_population: List[Individual]):
        self.prev_populations.append((self.current_population[:], self.best_fitness))
        self.current_population = new_population[:]
        self._update()
        self.state += 1  # Update the state when a step is taken

    def save_to_json(self):
        try:
            with open(self.json_path, "w") as file:
                json.dump(
                    {
                        "current_population": [
                            indiv.to_dict() for indiv in self.current_population
                        ],
                        "prev_populations": [
                            ([indiv.to_dict() for indiv in pop], fitness)
                            for pop, fitness in self.prev_populations
                        ],
                        "best_fitness": self.best_fitness,
                        "best_indiv": self.best_indiv.to_dict() if self.best_indiv else None,
                        "state": self.state,  # Save the current state
                    },
                    file,
                    indent=4,
                    ensure_ascii=False,
                )
        except Exception as e:
            logging.error(f"Failed to save population to {self.json_path}: {e}")

    def load_from_json(self):
        try:
            with open(self.json_path, "r") as file:
                data = json.load(file)
                self.current_population = [
                    Individual.from_dict(indiv_data) for indiv_data in data["current_population"]
                ]
                self.prev_populations = [
                    ([Individual.from_dict(indiv_data) for indiv_data in pop], fitness)
                    for pop, fitness in data["prev_populations"]
                ]
                self.best_fitness = data["best_fitness"]
                self.best_indiv = (
                    Individual.from_dict(data["best_indiv"]) if data["best_indiv"] else None
                )
                self.state = data.get("state", 0)  # Load the state from JSON or default to 0
        except FileNotFoundError:
            logging.warning(
                f"No JSON file found at {self.json_path}. Starting with an empty population."
            )
        except Exception as e:
            logging.error(f"Failed to load population from {self.json_path}: {e}")

    def get_current_ideas(self) -> List[str]:
        return [indiv.chromosome["idea"] for indiv in self.current_population]

    def get_current_codes(self) -> List[str]:
        return [indiv.chromosome["added_cut"] for indiv in self.current_population]

    def get_all_ideas(self) -> List[str]:
        all_ideas = []
        for population, _ in self.prev_populations:
            all_ideas.extend([indiv.chromosome["idea"] for indiv in population])
        all_ideas.extend(self.get_current_ideas())
        return all_ideas

    def get_current_size(self):
        return len(self.current_population)

    def get_all_indivs(self):
        # return unique individuals from all populations based on their idea
        unique_indivs = {}
        # Process individuals from previous populations
        for population, _ in self.prev_populations:
            for indiv in population:
                idea = indiv.chromosome["idea"]
                if idea not in unique_indivs:
                    unique_indivs[idea] = indiv

        # Process individuals from the current population
        for indiv in self.current_population:
            idea = indiv.chromosome["idea"]
            if idea not in unique_indivs:
                unique_indivs[idea] = indiv

        return list(unique_indivs.values())

    @staticmethod
    def get_average_fintesses(population: List[Individual]) -> float:
        """
        Calculate the average fitness of a population.
        Args:
            population (List[Individual]): A list of individuals in the population.
        Returns:
            float: The average fitness of the population.
        """
        total_fitness = 0
        for indiv in population:
            total_fitness += indiv.fitness
        return total_fitness / len(population) if population else 0

    @staticmethod
    def get_std_dev_fintesses(population: List[Individual]) -> float:
        """
        Calculate the standard deviation of fitnesses in a population.
        Args:
            population (List[Individual]): A list of individuals in the population.
        Returns:
            float: The standard deviation of fitnesses in the population.
        """
        avg = Population.get_average_fintesses(population)
        variance = sum((indiv.fitness - avg) ** 2 for indiv in population) / len(population)
        return variance**0.5 if population else 0
