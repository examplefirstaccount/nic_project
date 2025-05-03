from typing import Callable

import numpy as np

from core.chromosome import Chromosome
from core.problem import SchedulingProblem


class GeneticAlgorithm:
    def __init__(self, problem: SchedulingProblem) -> None:
        self.problem = problem

    def tournament_selection(
            self,
            population: list[Chromosome],
            selection_size: int,
            tournament_size: int
    ) -> list[Chromosome]:
        costs = np.array([self.problem.evaluate(ind) for ind in population])
        selected = []
        for _ in range(selection_size):
            indices = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = indices[np.argmin(costs[indices])]
            selected.append(population[best_idx])
        return selected

    def crossover(
            self,
            parent1: Chromosome,
            parent2: Chromosome,
            prob: float = 0.2,
            allow_single_swap: bool = True,
            random_order: bool = True
    ) -> tuple[Chromosome, Chromosome]:

        child1, child2 = parent1.copy(), parent2.copy()

        group_indices = np.arange(parent1.num_groups)
        if random_order:
            np.random.shuffle(group_indices)

        for idx in group_indices:
            idx: int
            slot1, slot2 = parent1.assigned_slots[idx], parent2.assigned_slots[idx]

            valid1 = child1.is_swap_valid(idx, slot2, self.problem.group_sizes, self.problem.slots_min, self.problem.slots_max)
            valid2 = child2.is_swap_valid(idx, slot1, self.problem.group_sizes, self.problem.slots_min, self.problem.slots_max)

            if valid1 and valid2 and np.random.rand() < prob:
                child1.update_occupancy(idx, slot1, slot2, self.problem.group_sizes)
                child2.update_occupancy(idx, slot2, slot1, self.problem.group_sizes)
                child1.assigned_slots[idx], child2.assigned_slots[idx] = slot2, slot1
            elif allow_single_swap:
                if valid1 and np.random.rand() < prob:
                    child1.update_occupancy(idx, slot1, slot2, self.problem.group_sizes)
                    child1.assigned_slots[idx] = slot2
                elif valid2 and np.random.rand() < prob:
                    child2.update_occupancy(idx, slot2, slot1, self.problem.group_sizes)
                    child2.assigned_slots[idx] = slot1

        return child1, child2

    def mutation_random_swap(self, chromosome: Chromosome, mutation_rate: float = 0.1) -> None:
        """
        Performs a random valid mutation on a chromosome with given probability.

        With probability mutation_rate, selects a random family and moves it to a
        different valid day that satisfies capacity constraints. The new day is
        selected randomly from all possible valid alternatives. May not improve solution
        """

        if np.random.rand() >= mutation_rate:
            return

        idx = np.random.randint(chromosome.num_groups)
        current_slot = chromosome.assigned_slots[idx]

        possible_slots = np.delete(np.arange(1, chromosome.num_slots + 1), current_slot - 1)
        np.random.shuffle(possible_slots)

        for new_slot in possible_slots:
            if chromosome.is_swap_valid(idx, new_slot, self.problem.group_sizes, self.problem.slots_min, self.problem.slots_max):
                chromosome.update_occupancy(idx, current_slot, new_slot, self.problem.group_sizes)
                chromosome.assigned_slots[idx] = new_slot
                break

    def mutation_prefer_better_slot(self, chromosome: Chromosome, mutation_rate: float = 0.1) -> None:
        """
        Attempts to mutate a group's assigned slot to a better-ranked one from their preferences,
        and falls back to random mutation if no improvement is possible.
        """

        if np.random.rand() >= mutation_rate:
            return

        idx = np.random.randint(chromosome.num_groups)
        current_slot = chromosome.assigned_slots[idx]

        preferred_slots = self.problem.choice_matrix[idx]
        try:
            current_rank = np.where(preferred_slots == current_slot)[0][0]
        except IndexError:
            current_rank = len(preferred_slots)

        possible_slots = preferred_slots[:current_rank]
        np.random.shuffle(possible_slots)

        for new_slot in possible_slots:
            if chromosome.is_swap_valid(idx, new_slot, self.problem.group_sizes, self.problem.slots_min, self.problem.slots_max):
                chromosome.update_occupancy(idx, current_slot, new_slot, self.problem.group_sizes)
                chromosome.assigned_slots[idx] = new_slot
                return

        if np.random.rand() >= 0.07:
            return

        alt_slots = np.delete(np.arange(1, chromosome.num_slots + 1), current_slot - 1)
        np.random.shuffle(possible_slots)

        for new_slot in alt_slots:
            if chromosome.is_swap_valid(idx, new_slot, self.problem.group_sizes, self.problem.slots_min, self.problem.slots_max):
                chromosome.update_occupancy(idx, current_slot, new_slot, self.problem.group_sizes)
                chromosome.assigned_slots[idx] = new_slot
                break

    def reproduce(
            self,
            mutation_func: Callable,
            parents: list[Chromosome],
            crossover_proba=0.2,
            allow_single_swap=True,
            random_order=True,
            mutation_rate=0.1
    ):
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = self.crossover(parent1, parent2, prob=crossover_proba, allow_single_swap=allow_single_swap,
                                       random_order=random_order)
            mutation_func(child1, mutation_rate)
            mutation_func(child2, mutation_rate)
            next_generation.extend([child1, child2])
        return next_generation

    def epoch_optimal(self, population: list[Chromosome]):
        costs = np.array([self.problem.evaluate(ind) for ind in population])
        best_idx = np.argmin(costs)
        return population[best_idx], costs[best_idx]

    def run(
            self,
            pop_size=100,
            num_generations=200,
            tournament_size=5,
            crossover_proba=0.2,
            allow_single_swap=True,
            random_order=True,
            mutation_type="random",
            mutation_rate=0.1,
            elitism_ratio=0.1,
            verbose=True
    ) -> tuple[Chromosome, float]:

        if mutation_type == "random":
            mutation_func = self.mutation_random_swap
        elif mutation_type == "prefer_better":
            mutation_func = self.mutation_prefer_better_slot
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

        # Create starting population of random valid solutions
        population = [self.problem.initialize_chromosome() for _ in range(pop_size)]

        # Evaluate initial fitness (cost) for all individuals
        costs = [self.problem.evaluate(ind) for ind in population]

        # Sort population by fitness (ascending - lower cost is better)
        sorted_indices = np.argsort(costs)
        population = [population[i] for i in sorted_indices]
        costs = [costs[i] for i in sorted_indices]

        # Initialize global best tracking
        best_chromosome = population[0].copy()
        best_cost = costs[0]

        for generation in range(num_generations):
            # Elitism: preserve top individuals
            elite_size = max(1, int(elitism_ratio * pop_size))
            elites = [population[i].copy() for i in range(elite_size)]

            # Selection
            parents = self.tournament_selection(population, pop_size - elite_size, tournament_size)

            # Reproduction
            offspring = self.reproduce(mutation_func, parents, crossover_proba, allow_single_swap, random_order,
                                     mutation_rate)

            # Combine elites and offspring
            population = elites + offspring

            # Find the best in the current generation
            current_best, current_cost = self.epoch_optimal(population)

            # Update the best solution
            if current_cost < best_cost:
                best_chromosome, best_cost = current_best.copy(), current_cost

            if verbose:
                print(f"Generation {generation + 1}: Best Cost = {best_cost}")
        return best_chromosome, best_cost
