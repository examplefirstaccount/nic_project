import pandas as pd

from core.ga import GeneticAlgorithm
from santa import SantaSchedulingProblem


df = pd.read_csv('data/family_data.csv')
df.rename(columns={"family_id": "group_id", "n_people": "group_size"}, inplace=True)
problem = SantaSchedulingProblem(df, 100, 125, 300)

ga = GeneticAlgorithm(problem)
best_solution, best_cost = ga.run(
    pop_size=100,
    num_generations=40,
    tournament_size=5,
    crossover_proba=0.5,
    allow_single_swap=True,
    random_order=True,
    mutation_type="prefer_better",
    mutation_rate=0.3,
    elitism_ratio=0.1,
    verbose=True
)

submission = pd.read_csv('data/sample_submission.csv')
submission['assigned_day'] = best_solution.assigned_slots
submission.to_csv('data/submission.csv', index=False)
