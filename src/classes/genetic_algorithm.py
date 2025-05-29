import copy
import logging
import os
import random
import sys
import json
from typing import Any, Dict, List, Optional

from .llm_handler import LlmHandler
from .solver import Solver
from .individual import Individual
from .population import Population
from .cut_verifier import CutVerifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.general_utils import (
    extract_code,
    adopt_code_to_fucntion,
    extract_json,
)


class GeneticAlgorithm:
    def __init__(
        self,
        solver: Solver,
        evaluation_dataset: List[Dict[str, Any]],
        verification_dataset: List[Dict[str, Any]],
        number_population: int,
        time_limit: float,
        population_file: str = "population.json",
        max_llm_attempts: int = 3,
    ):
        """
        A class that implements a Genetic Algorithm for generating and refining 'cuts' via LLM.
        """
        self.population = Population(population_file)
        self.solver = solver
        self.evaluation_dataset = evaluation_dataset
        self.verification_dataset = verification_dataset
        self.number_population = number_population
        self.time_limit = time_limit
        self.max_llm_attempts = max_llm_attempts
        self.stat_file = population_file.split(".json")[0] + "_stats.json"

        self.stats = {
            "initializer_generation": [],
            "crossover_generation": [],
            "mutation_generation": [],
        }
        self.load_stats()

    def load_stats(self):
        """
        Load the stats from the stat file if it exists.
        """
        if os.path.exists(self.stat_file):
            try:
                with open(self.stat_file, "r") as f:
                    self.stats = json.load(f)
                logging.info("Loaded stats from file.")
            except Exception as e:
                logging.error(f"Failed to load stats file: {e}")
        else:
            logging.info("No existing stats file found. Starting fresh.")

    def save_stats(self):
        """
        Save the current stats to the stat file.
        """
        try:
            with open(self.stat_file, "w") as f:
                json.dump(
                    self.stats,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            logging.info("Stats saved to file.")
        except Exception as e:
            logging.error(f"Failed to save stats file: {e}")

    def initialize_population(
        self,
        *,
        llm_handler: LlmHandler,
        prompt_initializer: list,
        model_name: str,
        true_code: str,
    ):
        """
        Load (or create) individuals in the population up to the specified number.
        """
        # Load existing population from file
        self.population.load_from_json()
        current_population_size = len(self.population.current_population)

        # Generate new individuals if the population is not full
        if current_population_size < self.number_population:
            logging.info(
                f"Population size is {current_population_size}, "
                f"filling up to {self.number_population}."
            )
            ideas = self.population.get_current_ideas()
            codes = self.population.get_current_codes()

            while len(self.population.current_population) < self.number_population:

                # Create system prompt: prompt_initializer + existing ideas
                prompt_system = copy.deepcopy(prompt_initializer) + "\n\n".join(ideas)
                messages = [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": true_code},
                ]

                new_individual = self._attempt_llm_individual_generation(
                    llm_handler=llm_handler,
                    model_name=model_name,
                    true_code=true_code,
                    messages=messages,
                    error_feedback=(
                        "Encountered an error during execution: {error}. "
                        "Please correct the code, try again, and output the same format."
                    ),
                    invalid_feedback=(
                        "Verification failed with details: The generated cut is not valid. "
                        "Please try to correct it and output in the same format."
                    ),
                    operation_type="initializer",
                    agent_name="Initializer",
                    parents=None,
                )

                if new_individual is not None:
                    self.population.add_indiv(new_individual)
                    ideas.append(new_individual.chromosome["idea"])

                else:
                    logging.info(
                        "Max attempts reached without a valid individual. Skipping this slot."
                    )

    def step(
        self,
        *,
        cross_overs_prompts: Dict[str, str],
        mutations_prompts: Dict[str, str],
        llm_handler: LlmHandler,
        model_name: str,
        true_code: str,
        mutation_rate: float = 0.4,
        elitism_ratio: float = 0.2,
        cross_over_ratio: float = 0.8,
    ):
        """
        Run one generation step of the genetic algorithm (selection, crossover, mutation).
        """
        new_population = []

        # Elitism: keep top performers
        elitism_num = int(elitism_ratio * self.number_population)
        elite = sorted(self.population.current_population, key=lambda x: x.fitness, reverse=True)[
            :elitism_num
        ]
        new_population.extend(elite)

        # Generate offspring via crossover and mutation
        while len(new_population) < self.number_population:
            logging.info(
                f"Generating new individual: {len(new_population) + 1}/{self.number_population}"
            )
            # Selection
            parents = self.tournament_selection(num_parents=2)

            # Crossover
            if random.random() < cross_over_ratio:  # Probability of crossover
                cross_over_agent = random.choice(list(cross_overs_prompts.keys()))
                logging.info(f"Crossing over with agent: {cross_over_agent}")
                logging.info(f"Parents' fitness are: {[parent.fitness for parent in parents]}")
                prompt_cross_over = cross_overs_prompts[cross_over_agent]

                child = self._crossover(
                    parent1=parents[0],
                    parent2=parents[1],
                    prompt=prompt_cross_over,
                    llm_handler=llm_handler,
                    model_name=model_name,
                    true_code=true_code,
                    agent_name=cross_over_agent,
                )
                if child:
                    child.add_parents(parents, generator=cross_over_agent)
                    new_population.append(child)
                    if len(new_population) >= self.number_population:
                        break

            # Mutation
            if random.random() < mutation_rate:
                mutate_agent = random.choice(list(mutations_prompts.keys()))
                logging.info(f"Mutating with agent: {mutate_agent}")
                logging.info(f"Parents' fitness are: {[parents[0].fitness]}")
                prompt_mutate = mutations_prompts[mutate_agent]
                # Mutate one of the parents
                mutant = self._mutate(
                    individual=parents[0],
                    prompt=prompt_mutate,
                    llm_handler=llm_handler,
                    model_name=model_name,
                    true_code=true_code,
                    agent_name=mutate_agent,
                )
                if mutant and len(new_population) < self.number_population:
                    mutant.add_parents([parents[0]], generator=mutate_agent)
                    new_population.append(mutant)

        # Update population with the new generation (trim to exact size)
        self.population.step_evolution(new_population[: self.number_population])

    def tournament_selection(self, num_parents: int, tournament_size: int = 2) -> List[Individual]:
        """
        Simple tournament selection: pick 'tournament_size' individuals at random
        and return the top 'num_parents' individuals with the highest fitness.
        Ensures that the same parent is not selected more than once.
        """
        selected_parents = []
        available_population = self.population.current_population[:]

        for _ in range(num_parents):
            participants = random.sample(available_population, tournament_size)
            winner = max(participants, key=lambda x: x.fitness)
            selected_parents.append(winner)
            available_population.remove(winner)  # Remove the selected parent to avoid duplicates

        return selected_parents

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        prompt: str,
        llm_handler: LlmHandler,
        model_name: str,
        true_code: str,
        agent_name: str,
    ) -> Optional[Individual]:
        """
        Cross two parents via LLM to produce a child.
        """
        messages = [
            {
                "role": "system",
                "content": prompt.format(true_code=true_code),
            },
            {
                "role": "user",
                "content": (
                    "Parent 1:\n    Idea:{idea1}\n    Code:\n{code1}\n    Fitness:{fitness1}\n\n"
                    "Parent 2:\n    Idea:{idea2}\n    Code:\n{code2}\n    Fitness:{fitness2}"
                ).format(
                    idea1=parent1.chromosome["idea"],
                    code1=parent1.chromosome["added_cut"],
                    fitness1=parent1.fitness,
                    idea2=parent2.chromosome["idea"],
                    code2=parent2.chromosome["added_cut"],
                    fitness2=parent2.fitness,
                ),
            },
        ]

        # Pass the parents into the generation step for stats
        return self._attempt_llm_individual_generation(
            llm_handler=llm_handler,
            model_name=model_name,
            true_code=true_code,
            messages=messages,
            error_feedback=(
                "Encountered an error during execution: {error}. "
                "Please correct the code, try again, and output the same format."
            ),
            invalid_feedback=(
                "Verification failed: The generated cut is not valid. "
                "Please try to correct it and output in the same format."
            ),
            operation_type="crossover",
            agent_name=agent_name,
            parents=[parent1, parent2],
        )

    def _mutate(
        self,
        individual: Individual,
        prompt: str,
        llm_handler: LlmHandler,
        model_name: str,
        true_code: str,
        agent_name: str,
    ) -> Optional[Individual]:
        """
        Mutate an individual by prompting LLM for a changed version.
        """
        messages = [
            {
                "role": "system",
                "content": prompt.format(true_code=true_code),
            },
            {
                "role": "user",
                "content": ("Individual:\n    Idea:{idea}\n    Code:\n{code}").format(
                    idea=individual.chromosome["idea"],
                    code=individual.chromosome["added_cut"],
                ),
            },
        ]

        return self._attempt_llm_individual_generation(
            llm_handler=llm_handler,
            model_name=model_name,
            true_code=true_code,
            messages=messages,
            error_feedback=(
                "Encountered an error during execution: {error}. "
                "Please correct the code, try again, and output the same format."
            ),
            invalid_feedback=(
                "Verification failed: The generated cut is not valid. "
                "Please try to correct it and output in the same format."
            ),
            operation_type="mutation",
            agent_name=agent_name,
            parents=[individual],
        )

    def _attempt_llm_individual_generation(
        self,
        *,
        llm_handler: LlmHandler,
        model_name: str,
        true_code: str,
        messages: List[Dict[str, str]],
        error_feedback: str,
        invalid_feedback: str,
        operation_type: str,  # "initializer", "crossover", or "mutation"
        agent_name: str,
        parents: Optional[List[Individual]] = None,
    ) -> Optional[Individual]:
        """
        Helper method that tries up to 'max_llm_attempts' times to generate a valid Individual
        from the LLM. If it fails, returns None.
        """
        attempt = 0

        # Prepare a small parent-info structure for stats
        parents_info = []
        if parents:
            parents_info = [{"id": p.id, "fitness": p.fitness} for p in parents]

        while attempt < self.max_llm_attempts:
            attempt += 1

            # A record of everything about this attempt
            record = {
                "attempt_number": attempt,
                "operation_type": operation_type,
                "agent_name": agent_name,
                "messages_so_far": [m for m in messages],
                "success": False,
                "error_reason": None,
                "verification_passed": False,
                "fitness": None,
                "response_snippet": None,
                "parents_info": parents_info,
                "child_id": None,
                "child_fitness": None,
            }

            try:
                response = llm_handler.chat(
                    model=model_name,
                    messages=messages,
                )
                json_response = extract_json(response)
                code = extract_code(json_response["code"])
                idea = json_response["idea"]
                record["response_snippet"] = code

                # Validate syntax
                full_code = adopt_code_to_fucntion(copy.deepcopy(true_code), code)
                exec(full_code, globals())

            except Exception as extraction_or_execution_error:
                error_msg = str(extraction_or_execution_error)
                logging.error(f"Attempt {attempt}: Error extracting/executing code: {error_msg}")
                record["error_reason"] = f"extraction_or_execution_exception: {error_msg}"
                if response:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response if "response" in locals() else "No output",
                        }
                    )
                messages.append({"role": "user", "content": error_feedback.format(error=error_msg)})

                # Save stats and continue to next attempt
                self.stats[operation_type + "_generation"].append(record)
                self.save_stats()
                continue  # Move to the next attempt

            # Check validity
            try:
                validity = CutVerifier.verify_score_cut(
                    verification_dataset=self.verification_dataset,
                    evaluation_dataset=self.evaluation_dataset,
                    solver=self.solver,
                    full_code=full_code,
                    true_code=true_code,
                )
            except Exception as verification_error:
                error_msg = str(verification_error)
                logging.error(f"Attempt {attempt}: Error during verification: {error_msg}")
                record["error_reason"] = f"verification_exception: {error_msg}"

                # Save stats
                self.stats[operation_type + "_generation"].append(record)
                self.save_stats()

                messages.append(
                    {
                        "role": "assistant",
                        "content": response if "response" in locals() else "No output",
                    }
                )
                messages.append({"role": "user", "content": error_feedback.format(error=error_msg)})
                continue  # Move to the next attempt

            if validity["verified"]:
                # We have a good individual; compute fitness
                record["verification_passed"] = True
                fitness = CutVerifier.cal_fitness(
                    solver_reports=validity["solver_reports"],
                    solver_reports_ref=[
                        data["true_score"]["solver_report"] for data in self.evaluation_dataset
                    ],
                    time_limit=self.time_limit,
                    solver_name=self.solver.solver_name,
                )
                record["success"] = True
                record["fitness"] = fitness

                # Create the child individual
                child = Individual(
                    chromosome={
                        "full_code": full_code,
                        "added_cut": code,
                        "idea": idea,
                    },
                    fitness=fitness,
                    solver_reports=validity["solver_reports"],
                )

                # Store child's ID/fitness in the record
                record["child_id"] = child.id
                record["child_fitness"] = child.fitness

                logging.info(f"Generated individual with fitness {fitness}")

                # Append stats and save
                self.stats[operation_type + "_generation"].append(record)
                self.save_stats()

                return child

            else:
                # Code is invalid according to CutVerifier
                logging.info(f"Attempt {attempt}: Verification failed.")
                record["error_reason"] = "verification_failed"

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": invalid_feedback})

                self.stats[operation_type + "_generation"].append(record)
                self.save_stats()

        # If all attempts have been exhausted without generating a valid individual:
        return None
