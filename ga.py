import numpy as np
import pandas as pd

np.random.seed(52)

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

def cost_function(prediction):
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


'''
Initialize a chromosome randomly.
'''
def initialize_chromosome():
    chromosome = np.random.randint(1, N_DAYS + 1, size=len(family_size_dict))
    return chromosome


'''
Perform tournament selection to choose parents.
'''
def selection(population, selection_size, tournament_size):
    selected = []
    for _ in range(selection_size):
        # Randomly sample tournament_size individuals from the population
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament = [population[i] for i in tournament_indices]

        # Select the one with the lowest cost
        winner = min(tournament, key=lambda x: cost_function(x))
        selected.append(winner)
    return selected


'''
Perform uniform crossover between two parents.
'''
def crossover(parent1, parent2):
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        child[i] = parent1[i] if np.random.rand() < 0.5 else parent2[i]
    return child


'''
Mutate a chromosome by reassigning families to their preferred days.
'''
def mutation(chromosome, mutation_rate=0.01):
    for family_idx in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            # Assign a random day (1 to 100)
            chromosome[family_idx] = np.random.randint(1, N_DAYS + 1)
    return chromosome


'''
Create the next generation through crossover and mutation.
'''
def reproduction(parents, mutation_rate=0.01):
    next_generation = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i + 1]
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)
        next_generation.append(mutation(child1, mutation_rate))
        next_generation.append(mutation(child2, mutation_rate))
    return next_generation


'''
Find the best chromosome in the population.
'''
def epoch_optimal(population):
    best_chromosome = min(population, key=lambda x: cost_function(x))
    best_cost = cost_function(best_chromosome)
    return best_chromosome, best_cost


'''
The Genetic Algorithm function.
'''
def genetic_algorithm(pop_size=100, num_generations=200, tournament_size=5, mutation_rate=0.2):
    # Initialize population
    population = [initialize_chromosome() for _ in range(pop_size)]

    # Track the best solution
    best_chromosome, best_cost = epoch_optimal(population)

    for generation in range(num_generations):
        # Selection
        parents = selection(population, pop_size, tournament_size)

        # Reproduction
        population = reproduction(parents, mutation_rate)

        # Find the best in the current generation
        current_best, current_cost = epoch_optimal(population)

        # Update the best solution
        if current_cost < best_cost:
            best_chromosome, best_cost = current_best, current_cost

        print(f"Generation {generation + 1}: Best Cost = {best_cost}")

    return best_chromosome, best_cost


'''
Run the GA.
'''
best_ch, best_c = genetic_algorithm(
    pop_size=50,
    num_generations=100,
    tournament_size=5,
    mutation_rate=0.1
)

print("Best Chromosome:", best_ch)
print("Best Cost:", best_c)

submission['assigned_day'] = best_ch
submission.to_csv('data/submission.csv', index=False)
