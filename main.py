import time

import pandas as pd
import matplotlib.pyplot as plt

from core.ga import GeneticAlgorithm
from core.problem import initialize_chromosome_numba
from examples.santa import SantaSchedulingProblem, evaluate_numba


def main():
    """
    Main function to solve the Santa scheduling problem using a Genetic Algorithm (GA).

    - Loads the family data.
    - Initializes the SantaSchedulingProblem.
    - Configures and runs the Genetic Algorithm.
    - Saves the best assignment to a CSV file.
    - Plots and saves the convergence of the GA (best cost per generation).
    """

    # Load input data
    df = pd.read_csv('data/family_data.csv')
    df.rename(columns={"family_id": "group_id", "n_people": "group_size"}, inplace=True)

    # Instantiate the problem with occupancy limits (125 <= occupancy <= 300) for 100 slots
    problem = SantaSchedulingProblem(df, 100, 125, 300)

    # Initialize the GA solver
    ga = GeneticAlgorithm(problem)

    # Run the GA and track time
    start_time = time.time()
    assigned_slots, best_cost = ga.run(
        initialize_chromosome_func=initialize_chromosome_numba,
        evaluate_func=evaluate_numba,
        pop_size=2000,
        num_generations=1000,
        tournament_size=5,
        crossover_proba=0.5,
        allow_single_swap=True,
        random_order=True,
        mutation_type="prefer_better",
        mutation_rate=0.3,
        crazy_rate=0.1,
        elitism_ratio=0.1,
        verbose=True
    )
    elapsed_time = time.time() - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

    # Save the assignment results to submission file
    submission = pd.read_csv('data/sample_submission.csv')
    submission['assigned_day'] = assigned_slots
    submission.to_csv('data/submission.csv', index=False)
    print("Submission saved to data/submission.csv")

    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(ga.best_costs_history, label='Best Cost', color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.title("GA Convergence Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/convergence_plot.png")
    print("Convergence plot saved to data/convergence_plot.png")


if __name__ == "__main__":
    main()
