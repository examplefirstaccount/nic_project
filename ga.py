import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd


'''
Load data
'''

data = pd.read_csv('data/family_data.csv')
submission = pd.read_csv('data/sample_submission.csv')

matrix = data[['choice_0', 'choice_1', 'choice_2', 'choice_3', 'choice_4',
               'choice_5', 'choice_6', 'choice_7', 'choice_8', 'choice_9']].to_numpy()

'''
Cost function inspired by https://www.kaggle.com/xhlulu/santa-s-2019-4x-faster-cost-function
'''

family_size_dict = data[['n_people']].to_dict()['n_people']
cols = [f'choice_{i}' for i in range(10)]
choice_dict = data[cols].T.to_dict()

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

days = list(range(N_DAYS,0,-1))

family_size_ls = list(family_size_dict.values())
choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]

penalties_dict = {
    n: [
        0,
        50,
        50 + 9 * n,
        100 + 9 * n,
        200 + 9 * n,
        200 + 18 * n,
        300 + 18 * n,
        300 + 36 * n,
        400 + 36 * n,
        500 + 36 * n + 199 * n,
        500 + 36 * n + 398 * n
    ]
    for n in range(max(family_size_dict.values())+1)
}
# print(penalties_dict)

def cost_function(prediction: np.ndarray) -> float:
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}

    # Looping over each family; d is the day, n is size of that family,
    # and choice is their top choices
    for n, d, choice in zip(family_size_ls, prediction, choice_dict_num):
        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d not in choice:
            penalty += penalties_dict[n][-1]
        else:
            penalty += penalties_dict[n][choice[d]]

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for v in daily_occupancy.values():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000
            # print(v)

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]] ** 0.5
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty


def is_attendance_valid(attendance: int) -> bool:
    return MIN_OCCUPANCY <= attendance <= MAX_OCCUPANCY


'''
Class for chromosome representation
'''
class Chromosome:
    def __init__(self, num_days=N_DAYS):
        self.assigned_days = np.zeros(len(family_size_dict), dtype=int)
        self.daily_attendance = np.zeros(num_days, dtype=int)

    def is_swap_valid(self, family_idx, new_day) -> bool:
        current_day = self.assigned_days[family_idx]
        family_size = family_size_dict[family_idx]

        reduced = self.daily_attendance[current_day - 1] - family_size
        increased = self.daily_attendance[new_day - 1] + family_size

        return is_attendance_valid(reduced) and is_attendance_valid(increased)

    def update_attendance(self, family_idx, old_day, new_day):
        self.daily_attendance[old_day - 1] -= family_size_dict[family_idx]
        self.daily_attendance[new_day - 1] += family_size_dict[family_idx]

    def __repr__(self):
        return (
            f"Chromosome(\n"
            f"  assigned_days={list(self.assigned_days)},\n"
            f"  daily_attendance={list(self.daily_attendance)},\n"
            f"  valid_days={sum(is_attendance_valid(x) for x in self.daily_attendance)}/{len(self.daily_attendance)}\n"
            f")"
        )

'''
Initialize a chromosome randomly.
Valid Solution Rate: 100.00%
Mean Cost: 2 493 257.96
Median Cost: 2 250 130.03
'''
def initialize_chromosome() -> Chromosome:
    # chromosome = np.random.randint(1, N_DAYS + 1, size=len(family_size_dict))
    people_in_day = {day:0 for day in range(1, N_DAYS + 1)}
    chromosome = np.zeros(len(family_size_dict), dtype=int)
    for i in range(len(chromosome)):
        items = list(choice_dict_num[i].keys()).copy()
        while True:
            new_number = random.randint(1, N_DAYS)
            if new_number not in items:
                items += [new_number]
                break
        # print(items)
        weights = penalties_dict[family_size_dict[i]]
        # Обратные веса (чем больше исходный вес, тем меньше вероятность)
        inverse_weights = []
        for j in range(len(weights)):
            c = 50
            if people_in_day[items[j]] in range(125, 301):
                c += (people_in_day[items[j]] - 125) * 1000
            elif people_in_day[items[j]] > 300:
                c = 1000000000
            inverse_weights.append(1 / (weights[j] + c))

        # valid_items = [day for day in items if people_in_day[day] + family_size_ls[i] <= 300]
        # if not valid_items:  # Если все дни переполнены, ослабляем ограничение
            # valid_items = [day for day in range(1, N_DAYS + 1) if people_in_day[day] + family_size_ls[i] <= 300]
        new_items = []
        new_inverse_weights = []
        for j in range(len(items)):
            if people_in_day[items[j]] + family_size_ls[i] <= 300:
                new_items.append(items[j])
                new_inverse_weights.append(inverse_weights[j])


        # print(items, inverse_weights)
        chromosome[i] = random.choices(new_items, new_inverse_weights, k=1)[0]

        # if chromosome[i] not in people_in_day:
        #     people_in_day[chromosome[i]] = family_size_ls[i]
        # else:
        people_in_day[chromosome[i]] += family_size_ls[i]

    chromosome_cls = Chromosome()
    chromosome_cls.assigned_days = chromosome

    for key, value in people_in_day.items():
        chromosome_cls.daily_attendance[key - 1] = value

    return chromosome_cls


'''
Initialize a chromosome randomly.
Valid Solution Rate: 100.00%
Mean Cost: 74 184 215.24
Median Cost: 41 830 969.73
'''
def initialize_chromosome_na() -> Chromosome:
    chromosome = Chromosome()
    families = list(enumerate(matrix))
    np.random.shuffle(families)

    # Fill days below MIN_OCCUPANCY
    for family_id, choices in families:
        family_size = family_size_dict[family_id]
        days_to_fill = [d for d in days if chromosome.daily_attendance[d - 1] < MIN_OCCUPANCY]
        preferred_days_to_fill = list(set(choices).intersection(days_to_fill))

        if not days_to_fill:
            break

        if preferred_days_to_fill:
            assigned_day = np.random.choice(preferred_days_to_fill)
        else:
            assigned_day = np.random.choice(days_to_fill)

        chromosome.assigned_days[family_id] = assigned_day
        chromosome.daily_attendance[assigned_day - 1] += family_size

    # Assign remaining families without exceeding MAX_OCCUPANCY
    for family_id, choices in families:
        if chromosome.assigned_days[family_id] != 0:
            continue  # Skip already assigned families

        family_size = family_size_dict[family_id]
        days_to_fill = [d for d in days if chromosome.daily_attendance[d - 1] + family_size <= MAX_OCCUPANCY]
        preferred_days_to_fill = list(set(choices).intersection(days_to_fill))

        if not days_to_fill:
            assigned_day = np.random.choice(days)
        else:
            if preferred_days_to_fill:
                assigned_day = np.random.choice(preferred_days_to_fill)
            else:
                assigned_day = np.random.choice(days_to_fill)

        chromosome.assigned_days[family_id] = assigned_day
        chromosome.daily_attendance[assigned_day - 1] += family_size

    return chromosome


def test_initialization(n_runs: int = 100):
    valid_count = 0
    costs = []

    for _ in range(n_runs):
        chromosome = initialize_chromosome()
        cost = cost_function(chromosome.assigned_days)
        costs.append(cost)

        # Check validity
        if all(MIN_OCCUPANCY <= count <= MAX_OCCUPANCY for count in chromosome.daily_attendance):
            valid_count += 1

    valid_rate = valid_count / n_runs * 100
    mean_cost = np.mean(costs)
    median_cost = np.median(costs)

    print(f"Valid Solution Rate: {valid_rate:.2f}%")
    print(f"Mean Cost: {mean_cost:.2f}")
    print(f"Median Cost: {median_cost:.2f}")


'''
Perform tournament selection to choose parents.
'''
def selection(population: list[Chromosome], selection_size: int, tournament_size: int) -> list[Chromosome]:
    selected = []
    for _ in range(selection_size):
        # Randomly sample tournament_size individuals from the population
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament = [population[i] for i in tournament_indices]

        # Select the one with the lowest cost
        winner = min(tournament, key=lambda x: cost_function(x.assigned_days))
        selected.append(winner)
        # is_winner_in_selected = any(np.array_equal(winner, ind) for ind in selected)
        #
        # if is_winner_in_selected:
        #     selected.append(winner)
        # else:
        #     selected.append(tournament[random.randint(0, tournament_size-1)])
    # selected = sorted(population, key=cost_function)[:selection_size]
    # print(selection_size)
    # print(selected)
    return selected


'''
Perform uniform crossover between two parents.
'''
def crossover_bf(parent1, parent2):
    child = np.zeros_like(parent1)
    people_in_day = {day:0 for day in range(1, N_DAYS + 1)}
    for i in range(len(parent1)):
        day1, day2 = parent1[i], parent2[i]
        threshold = 0
        if people_in_day[day1] < 125 and people_in_day[day2] < 125:
            threshold = 0.5
        elif people_in_day[day1] >= 125:
            threshold = (300 - people_in_day[day1]) / (2 * (300 - 125))
        elif people_in_day[day2] >= 125:
            threshold = 1 - (300 - people_in_day[day2]) / (2 * (300 - 125))
        child[i] = day1 if np.random.rand() < threshold else day2
        people_in_day[child[i]] += family_size_ls[i]
    return child


def perform_swap(child1: Chromosome, child2: Chromosome, family_idx: int, day1: int, day2: int):
    """Helper function to execute the swap"""
    child1.update_attendance(family_idx, day1, day2)
    child2.update_attendance(family_idx, day2, day1)
    child1.assigned_days[family_idx] = day2
    child2.assigned_days[family_idx] = day1


def crossover(
        parent1: Chromosome,
        parent2: Chromosome,
        p: float = 1.0,
        allow_single_swap: bool = False,
        random_order: bool = False
) -> tuple[Chromosome, Chromosome, dict]:

    """
    Enhanced crossover with:
    - Option for single valid swaps
    - Random visitation order
    - Detailed statistics
    """
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    stats = {
        'successful_swaps': 0,
        'single_swaps': 0,
        'failed_swaps': 0,
        'swap_attempts': 0,
        'family_swap_counts': defaultdict(int)
    }

    family_indices = np.arange(len(parent1.assigned_days))
    if random_order:
        np.random.shuffle(family_indices)

    for family_idx in family_indices:
        day1 = parent1.assigned_days[family_idx]
        day2 = parent2.assigned_days[family_idx]
        stats['swap_attempts'] += 1

        # Check both-way swap validity
        both_valid = (child1.is_swap_valid(family_idx, day2) and
                      child2.is_swap_valid(family_idx, day1))

        # Check single-way swap validity
        child1_valid = child1.is_swap_valid(family_idx, day2)
        child2_valid = child2.is_swap_valid(family_idx, day1)

        if both_valid:
            if np.random.rand() < p:
                perform_swap(child1, child2, family_idx, day1, day2)
                stats['successful_swaps'] += 1
                stats['family_swap_counts'][family_idx] += 1
        elif allow_single_swap:
            if child1_valid and np.random.rand() < p:
                new_day = parent2.assigned_days[family_idx]
                child1.update_attendance(family_idx, day1, new_day)
                child1.assigned_days[family_idx] = new_day
                stats['single_swaps'] += 1
            elif child2_valid and np.random.rand() < p:
                new_day = parent1.assigned_days[family_idx]
                child2.update_attendance(family_idx, day2, new_day)
                child2.assigned_days[family_idx] = new_day
                stats['single_swaps'] += 1
        else:
            stats['failed_swaps'] += 1

    stats['swap_rate'] = stats['successful_swaps'] / stats['swap_attempts'] if stats['swap_attempts'] > 0 else 0
    return child1, child2, stats


def test_crossover(n_tests=100, p=1.0, allow_single_swap=False, random_order=False):
    """
    Comprehensive crossover testing
    Returns:
        - Average swap rates
        - Cost improvements
        - Family swap frequency distribution
    """
    results = {
        'swap_rates': [],
        'cost_changes': [],
        'single_swap_rates': [],
        'family_swap_freq': defaultdict(int)
    }

    for _ in range(n_tests):
        parent1 = initialize_chromosome()
        parent2 = initialize_chromosome()

        initial_cost = (cost_function(parent1.assigned_days) +
                        cost_function(parent2.assigned_days)) / 2

        child1, child2, stats = crossover(
            parent1, parent2,
            p=p,
            allow_single_swap=allow_single_swap,
            random_order=random_order
        )

        final_cost = (cost_function(child1.assigned_days) +
                      cost_function(child2.assigned_days)) / 2

        results['swap_rates'].append(stats['swap_rate'])
        results['cost_changes'].append(initial_cost - final_cost)
        results['single_swap_rates'].append(
            stats['single_swaps'] / stats['swap_attempts'] if stats['swap_attempts'] > 0 else 0)

        for fam_idx, count in stats['family_swap_counts'].items():
            results['family_swap_freq'][fam_idx] += count

    # Calculate statistics
    stats_summary = {
        'avg_swap_rate': np.mean(results['swap_rates']),
        'median_swap_rate': np.median(results['swap_rates']),
        'avg_cost_improvement': np.mean(results['cost_changes']),
        'successful_families': sorted(
            [(k, v / n_tests) for k, v in results['family_swap_freq'].items()],
            key=lambda x: -x[1]
        )[:10],  # Top 10 most swapped families
        'single_swap_impact': np.mean(results['single_swap_rates']) if allow_single_swap else 0
    }

    print(stats_summary)


'''
Mutate a chromosome by reassigning families to their preferred days.
If applying to all 5k families with p=0.01, then 40 mutations will be applied.
Average possible valid mutations per family (gene) is 90 (out of 100).
'''
def mutation1(chromosome: Chromosome, mutation_rate: float = 0.1):
    family_idx = np.random.randint(len(chromosome.assigned_days))
    current_day = chromosome.assigned_days[family_idx]

    valid_days = [
        day for day in range(1, N_DAYS + 1)
        if chromosome.is_swap_valid(family_idx, day) and day != current_day
    ]

    if valid_days and np.random.rand() < mutation_rate:
        new_day = np.random.choice(valid_days)
        chromosome.update_attendance(family_idx, current_day, new_day)
        chromosome.assigned_days[family_idx] = new_day


def mutation2(chromosome: Chromosome, mutation_rate: float = 0.01):
    num_mutations = 0

    for family_idx in range(len(chromosome.assigned_days)):
        if np.random.rand() < mutation_rate:
            current_day = chromosome.assigned_days[family_idx]

            valid_days = [
                day for day in range(1, N_DAYS + 1)
                if chromosome.is_swap_valid(family_idx, day) and day != current_day
            ]

            if valid_days:
                new_day = np.random.choice(valid_days)
                chromosome.update_attendance(family_idx, current_day, new_day)
                chromosome.assigned_days[family_idx] = new_day
                num_mutations += 1

    print(num_mutations)


'''
Create the next generation through crossover and mutation.
'''
def reproduction(
        mutation_func,
        parents: list[Chromosome],
        crossover_proba: float = 1.0,
        allow_single_swap: bool = False,
        random_order: bool = False,
        mutation_rate: float = 0.01
) -> list[Chromosome]:

    next_generation = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        child1, child2, stats = crossover(parent1, parent2, p=crossover_proba, allow_single_swap=allow_single_swap, random_order=random_order)
        mutation_func(child1, mutation_rate)
        mutation_func(child2, mutation_rate)
        next_generation.extend([child1, child2])
    return next_generation


'''
Find the best chromosome in the population.
'''
def epoch_optimal(population: list[Chromosome]):
    best_chromosome = min(population, key=lambda x: cost_function(x.assigned_days))
    best_cost = cost_function(best_chromosome.assigned_days)
    return best_chromosome, best_cost


# population = [initialize_chromosome() for _ in range(100)]
# print(population)
# print(epoch_optimal(population))

# for p in range(50):
#     days_ = {}
#     for i in range(len(population[p])):
#         day = population[p][i]
#         if day not in days_:
#             days_[day] = family_size_ls[i]
#         else:
#             days_[day] += family_size_ls[i]
#     for k in days_:
#         if days_[k] < 125 or days_[k] > 300:
#             print(True)
#             print(days_)
#             print()
#             break
# print(days_)


'''
The Genetic Algorithm function.
'''
def genetic_algorithm(
        mutation_func,
        pop_size: int = 100,
        num_generations: int = 200,
        tournament_size: int = 5,
        crossover_proba: float = 1.0,
        allow_single_swap: bool = False,
        random_order: bool = False,
        mutation_rate: float = 0.1,
        elitism_ratio: float = 0.1
) -> tuple[Chromosome, float]:

    # Initialize population
    population = [initialize_chromosome() for _ in range(pop_size)]
    population.sort(key=lambda x: cost_function(x.assigned_days))

    # for ch in population:
    #     print(cost_function(ch))
    # print(population)

    # Track the best solution
    best_chromosome, best_cost = population[0], cost_function(population[0].assigned_days)
    # print(best_chromosome, best_cost)

    for generation in range(num_generations):
        # Elitism: preserve top individuals
        elite_size = max(1, int(elitism_ratio * pop_size))
        elites = deepcopy(population[:elite_size])

        # Selection
        parents = selection(population, pop_size - elite_size, tournament_size)

        # Reproduction
        offspring = reproduction(mutation_func, parents, crossover_proba, allow_single_swap, random_order, mutation_rate)

        # Combine elites and offspring
        population = elites + offspring

        # Find the best in the current generation
        current_best, current_cost = epoch_optimal(population)

        # Update the best solution
        if current_cost < best_cost:
            best_chromosome, best_cost = deepcopy(current_best), current_cost

        print(f"Generation {generation + 1}: Best Cost = {best_cost}")
    return best_chromosome, best_cost


'''
Run the GA.
'''
best_ch, best_c = genetic_algorithm(
    pop_size=100,
    num_generations=2000,
    tournament_size=5,
    crossover_proba=0.5,
    allow_single_swap=True,
    random_order=True,
    mutation_func=mutation1,
    mutation_rate=0.3,
    elitism_ratio=0.1
)

print("Best Chromosome:", best_ch)
print("Best Cost:", best_c)

submission['assigned_day'] = best_ch.assigned_days
submission.to_csv('data/submission.csv', index=False)
