from typing import Callable

import numpy as np
from numba import njit

from core.chromosome import ChromosomeType
from core.problem import SchedulingProblem


class GeneticAlgorithm:
    """
    Genetic Algorithm class to solve scheduling problems using evolutionary techniques.

    Attributes:
        problem (SchedulingProblem): An instance of the scheduling problem to be solved.
        best_solution (ChromosomeType): The best solution found by the algorithm.
        best_cost (float): The cost of the best solution.
        best_costs_history (np.ndarray): History of best costs per generation.
    """
    def __init__(self, problem: SchedulingProblem) -> None:
        self.problem = problem
        self.best_solution = None
        self.best_cost = None
        self.best_costs_history = None

    def run(
            self,
            initialize_chromosome_func,
            evaluate_func,
            pop_size=100,
            num_generations=200,
            tournament_size=5,
            crossover_proba=0.2,
            allow_single_swap=True,
            random_order=True,
            mutation_type="random",
            mutation_rate=0.1,
            crazy_rate=0.05,
            elitism_ratio=0.1,
            verbose=True
    ) -> tuple[np.ndarray, any]:
        """
        Run the genetic algorithm optimization loop.

        Args:
            initialize_chromosome_func (Callable): Numba-compatible function to generate a chromosome.
            evaluate_func (Callable): Numba-compatible fitness evaluation function.
            pop_size (int): Size of the population.
            num_generations (int): Number of generations to evolve.
            tournament_size (int): Size of the tournament for selection.
            crossover_proba (float): Probability of crossover per gene.
            allow_single_swap (bool): Whether to allow one-way slot swaps.
            random_order (bool): Whether to shuffle gene order before crossover.
            mutation_type (str): Mutation strategy, either "random" or "prefer_better".
            mutation_rate (float): Probability of mutation per gene.
            crazy_rate (float): Probability of a "crazy" mutation.
            elitism_ratio (float): Fraction of population retained unmodified.
            verbose (bool): Whether to print progress.

        Returns:
            tuple: Best chromosome (assigned slots) and corresponding cost.
        """
        mutation_func_id = 0 if mutation_type == "random" else 1

        best_chromosome, best_costs = run_ga_numba(
            problem_data=(self.problem.choice_dict_items, self.problem.choice_matrix, self.problem.choice_rank,
                          self.problem.group_sizes, self.problem.penalties_array, self.problem.slots_min,
                          self.problem.slots_max),
            initialize_chromosome_numba=initialize_chromosome_func,
            evaluate_numba=evaluate_func,
            pop_size=pop_size,
            num_generations=num_generations,
            tournament_size=tournament_size,
            crossover_proba=crossover_proba,
            allow_single_swap=allow_single_swap,
            random_order=random_order,
            mutation_rate=mutation_rate,
            crazy_rate=crazy_rate,
            elitism_ratio=elitism_ratio,
            mutation_func_id=mutation_func_id
        )

        self.best_solution = best_chromosome
        self.best_cost = best_costs[-1]
        self.best_costs_history = best_costs

        if verbose:
            print(f"[GA] Finished after {num_generations} generations. Best Cost = {self.best_cost}")

        return self.best_solution[0], self.best_cost


@njit
def tournament_selection_numba(
        population: np.ndarray,
        costs: np.ndarray,
        selection_size: int,
        tournament_size: int
) -> np.ndarray:
    """
    Select chromosomes using tournament selection.

    Args:
        population (np.ndarray): Current population.
        costs (np.ndarray): Cost array for each individual.
        selection_size (int): Number of individuals to select.
        tournament_size (int): Number of candidates per tournament.

    Returns:
        np.ndarray: Selected indices from the population.
    """
    selected = np.empty(selection_size, dtype=np.int32)
    for i in range(selection_size):
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmin(costs[indices])]
        selected[i] = best_idx
    return selected


@njit
def crossover_numba(
        parent1_slots: np.ndarray,
        parent1_occ: np.ndarray,
        parent2_slots: np.ndarray,
        parent2_occ: np.ndarray,
        group_sizes: np.ndarray,
        slots_min: np.ndarray,
        slots_max: np.ndarray,
        prob: float = 0.2,
        allow_single_swap: bool = True,
        random_order: bool = True
) -> tuple[ChromosomeType, ChromosomeType]:
    """
    Performs uniform crossover between two parent chromosomes, producing two children.

    Respects slot capacity constraints. Supports full or single-gene swaps.

    Returns:
        Tuple of two new chromosomes.
    """
    num_groups = len(parent1_slots)

    child1_slots = parent1_slots.copy()
    child2_slots = parent2_slots.copy()
    child1_occ = parent1_occ.copy()
    child2_occ = parent2_occ.copy()

    group_indices = np.arange(num_groups)
    if random_order:
        np.random.shuffle(group_indices)

    for idx in group_indices:
        slot1 = parent1_slots[idx]
        slot2 = parent2_slots[idx]

        size = group_sizes[idx]

        # Check capacity constraints
        valid1 = (child1_occ[slot2 - 1] + size <= slots_max[slot2 - 1] and
                  child1_occ[slot1 - 1] - size >= slots_min[slot1 - 1])
        valid2 = (child2_occ[slot1 - 1] + size <= slots_max[slot1 - 1] and
                  child2_occ[slot2 - 1] - size >= slots_min[slot2 - 1])

        if valid1 and valid2 and np.random.rand() < prob:
            child1_occ[slot1 - 1] -= size
            child1_occ[slot2 - 1] += size
            child2_occ[slot2 - 1] -= size
            child2_occ[slot1 - 1] += size
            child1_slots[idx], child2_slots[idx] = slot2, slot1
        elif allow_single_swap:
            if valid1 and np.random.rand() < prob:
                child1_occ[slot1 - 1] -= size
                child1_occ[slot2 - 1] += size
                child1_slots[idx] = slot2
            elif valid2 and np.random.rand() < prob:
                child2_occ[slot2 - 1] -= size
                child2_occ[slot1 - 1] += size
                child2_slots[idx] = slot1

    return (child1_slots, child1_occ), (child2_slots, child2_occ)


@njit
def mutation_random_swap_numba(
        chromosome: ChromosomeType,
        group_sizes: np.ndarray,
        slots_min: np.ndarray,
        slots_max: np.ndarray,
        mutation_rate: float = 0.1,
        crazy_rate: float = 0.05
):
    """
    Applies random mutation by reassigning a group to a feasible new slot.

    Mutation occurs with given rate and respects slot bounds.
    """
    assigned_slots, slot_occupancy = chromosome
    num_groups = len(assigned_slots)
    num_slots = len(slot_occupancy)

    if np.random.rand() >= max(mutation_rate, crazy_rate):
        return

    idx = np.random.randint(num_groups)
    current_slot = assigned_slots[idx]
    size = group_sizes[idx]

    possible_slots = np.delete(np.arange(1, num_slots + 1), current_slot - 1)
    np.random.shuffle(possible_slots)

    for new_slot in possible_slots:
        if (slot_occupancy[new_slot - 1] + size <= slots_max[new_slot - 1] and
                slot_occupancy[current_slot - 1] - size >= slots_min[current_slot - 1]):
            assigned_slots[idx] = new_slot
            slot_occupancy[current_slot - 1] -= size
            slot_occupancy[new_slot - 1] += size
            break


@njit
def mutation_prefer_better_slot_numba(
        chromosome: ChromosomeType,
        group_sizes: np.ndarray,
        choice_matrix: np.ndarray,
        slots_min: np.ndarray,
        slots_max: np.ndarray,
        mutation_rate: float = 0.1,
        crazy_rate: float = 0.05
):
    """
    Mutates chromosome by attempting to reassign a group to a more preferred slot.

    Falls back to a random feasible slot if no improvement is possible.
    """
    assigned_slots, slot_occupancy = chromosome
    num_groups = len(assigned_slots)
    num_slots = len(slot_occupancy)

    if np.random.rand() >= mutation_rate:
        return

    idx = np.random.randint(num_groups)
    current_slot = assigned_slots[idx]
    size = group_sizes[idx]

    preferred_slots = choice_matrix[idx]
    current_rank = len(preferred_slots)

    for i in range(len(preferred_slots)):
        if preferred_slots[i] == current_slot:
            current_rank = i
            break

    possible_slots = preferred_slots[:current_rank].copy()
    np.random.shuffle(possible_slots)

    for new_slot in possible_slots:
        if (slot_occupancy[new_slot - 1] + size <= slots_max[new_slot - 1] and
                slot_occupancy[current_slot - 1] - size >= slots_min[current_slot - 1]):
            assigned_slots[idx] = new_slot
            slot_occupancy[current_slot - 1] -= size
            slot_occupancy[new_slot - 1] += size
            return

    if np.random.rand() >= crazy_rate:
        return

    alt_slots = np.delete(np.arange(1, num_slots + 1), current_slot - 1)
    np.random.shuffle(alt_slots)

    for new_slot in alt_slots:
        if (slot_occupancy[new_slot - 1] + size <= slots_max[new_slot - 1] and
                slot_occupancy[current_slot - 1] - size >= slots_min[current_slot - 1]):
            assigned_slots[idx] = new_slot
            slot_occupancy[current_slot - 1] -= size
            slot_occupancy[new_slot - 1] += size
            break


@njit
def run_ga_numba(
        problem_data: tuple,
        initialize_chromosome_numba: Callable,
        evaluate_numba: Callable,
        pop_size: int,
        num_generations: int,
        tournament_size: int,
        crossover_proba: float,
        allow_single_swap: bool,
        random_order: bool,
        mutation_rate: float,
        crazy_rate: float,
        elitism_ratio: float,
        mutation_func_id: int
) -> tuple[ChromosomeType, np.ndarray]:
    """
    Runs the core genetic algorithm loop using Numba for performance.

    Args:
        problem_data (tuple): Tuple of problem arrays and matrices.
        initialize_chromosome_numba (Callable): Function to create initial chromosomes.
        evaluate_numba (Callable): Function to evaluate chromosomes.
        pop_size (int): Size of the population.
        num_generations (int): Number of generations to evolve.
        tournament_size (int): Size of the tournament for selection.
        crossover_proba (float): Probability of crossover per gene.
        allow_single_swap (bool): Whether to allow one-way slot swaps.
        random_order (bool): Whether to shuffle gene order before crossover.
        mutation_rate (float): Probability of mutation per gene.
        crazy_rate (float): Probability of a "crazy" mutation.
        elitism_ratio (float): Fraction of population retained unmodified.
        mutation_func_id (int): Mutation strategy, either 0 for "random" or 1 for "prefer_better".

    Returns:
        tuple: Best chromosome and array of best cost per generation.
    """
    choice_dict_items, choice_matrix, choice_rank, group_sizes, penalties_array, slots_min, slots_max = problem_data
    num_groups = len(group_sizes)
    num_slots = len(slots_min)

    population = np.empty((pop_size, num_groups + num_slots), dtype=np.int32)
    costs = np.empty(pop_size, dtype=np.float32)
    best_gen_costs = np.empty(num_generations, dtype=np.float32)

    for i in range(pop_size):
        chrom = initialize_chromosome_numba(num_groups, num_slots, group_sizes, slots_min, slots_max, choice_dict_items,
                                            penalties_array)
        population[i, :num_groups] = chrom[0]
        population[i, num_groups:] = chrom[1]
        costs[i] = evaluate_numba(num_slots, chrom, slots_min, slots_max, group_sizes, choice_rank, penalties_array)

    sorted_indices = np.argsort(costs)
    population = population[sorted_indices]
    costs = costs[sorted_indices]

    best_chromosome = population[0].copy()
    best_cost = costs[0]

    elite_size = max(1, int(elitism_ratio * pop_size))

    for gen_i in range(num_generations):
        new_population = np.empty_like(population)
        new_population[:elite_size] = population[:elite_size]

        selected_indices = tournament_selection_numba(population, costs, pop_size - elite_size, tournament_size)

        for i in range(0, pop_size - elite_size, 2):
            p1 = population[selected_indices[i]]
            p2 = population[selected_indices[i + 1]]

            parent1 = (p1[:num_groups], p1[num_groups:])
            parent2 = (p2[:num_groups], p2[num_groups:])

            child1, child2 = crossover_numba(
                parent1[0], parent1[1], parent2[0], parent2[1],
                group_sizes, slots_min, slots_max,
                crossover_proba, allow_single_swap, random_order
            )

            if mutation_func_id == 0:
                mutation_random_swap_numba(child1, group_sizes, slots_min, slots_max, mutation_rate, crazy_rate)
                mutation_random_swap_numba(child2, group_sizes, slots_min, slots_max, mutation_rate, crazy_rate)
            elif mutation_func_id == 1:
                mutation_prefer_better_slot_numba(child1, group_sizes, choice_matrix, slots_min, slots_max,
                                                  mutation_rate, crazy_rate)
                mutation_prefer_better_slot_numba(child2, group_sizes, choice_matrix, slots_min, slots_max,
                                                  mutation_rate, crazy_rate)

            new_population[elite_size + i, :num_groups] = child1[0]
            new_population[elite_size + i, num_groups:] = child1[1]
            new_population[elite_size + i + 1, :num_groups] = child2[0]
            new_population[elite_size + i + 1, num_groups:] = child2[1]

        population = new_population

        for i in range(pop_size):
            chrom = (population[i, :num_groups], population[i, num_groups:])
            costs[i] = evaluate_numba(num_slots, chrom, slots_min, slots_max, group_sizes, choice_rank, penalties_array)

        sorted_indices = np.argsort(costs)
        population = population[sorted_indices]
        costs = costs[sorted_indices]

        if costs[0] < best_cost:
            best_chromosome = population[0].copy()
            best_cost = costs[0]

        best_gen_costs[gen_i] = best_cost

    best = (best_chromosome[:num_groups], best_chromosome[num_groups:])
    return best, best_gen_costs
