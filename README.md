# Santa Workshop Optimization using Genetic Algorithms

This project provides a flexible Genetic Algorithm (GA) framework designed for solving group-to-slot scheduling problems. It was initially developed for the "Santa Workshop Optimization" challenge, where families (groups) need to be assigned to specific days (slots) while respecting daily attendance constraints and family preferences. The framework is built with extensibility and performance in mind, leveraging Numba for computationally intensive parts.

The core idea is to find a schedule that minimizes a total penalty score. This score is typically a combination of penalties for:
1.  Assigning groups to less-preferred slots.
2.  Violating slot occupancy limits (e.g., too few or too many people on a given day).
3.  Other problem-specific constraints (e.g., accounting costs based on daily attendance fluctuations in the Santa example).

## Features

*   **Generic GA Core:** A reusable `GeneticAlgorithm` class in `core/ga.py`.
*   **Abstract Problem Definition:** `core/problem.py` defines an `SchedulingProblem` abstract base class, allowing you to define custom scheduling problems.
*   **Numba Optimized:** Critical components like chromosome initialization, evaluation, crossover, and mutation are JIT-compiled with Numba for high performance.
*   **Constraint-Aware Operators:** GA operators (initialization, crossover, mutation) are designed to respect slot capacity constraints, promoting valid solutions.
*   **Configurable GA Parameters:** Population size, number of generations, selection strategy (tournament size), crossover probability, mutation rates, and elitism are easily configurable.
*   **Customizable Problem Logic:**
    *   **Evaluation Function:** Define your own problem-specific cost/fitness function.
    *   **Penalty Structure:** Define how penalties are calculated based on preferences and other factors.
    *   **Initialization:** While a robust default initialization is provided, you can supply your own Numba-compatible initialization function.
*   **Example Implementation:** The `examples/santa.py` module demonstrates how to use the framework for the Santa Workshop Optimization problem.
*   **Jupyter Notebook:** `ga.ipynb` provides a pure Python (non-Numba) version of the GA flow for easier understanding of the core algorithm.

## Prerequisites and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/examplefirstaccount/nic_project.git
    cd nic_project
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install `numpy`, `pandas`, `matplotlib`, and `numba`.

## Running the Santa Example

The `main.py` script runs the Genetic Algorithm for the Santa Workshop Optimization problem defined in `examples/santa.py`.

1.  **Ensure you have `data/family_data.csv` and `data/sample_submission.csv`.**
2.  **Run the script:**
    ```bash
    python main.py
    ```
    This will:
    *   Load family data.
    *   Initialize `SantaSchedulingProblem`.
    *   Run the `GeneticAlgorithm`.
    *   Print progress and the best cost found.
    *   Save the best assignment to `data/submission.csv`.
    *   Save a convergence plot to `data/convergence_plot.png`.

## Using the Framework for a Custom Scheduling Problem

To adapt this framework for your own scheduling problem, you'll primarily need to:

1.  **Prepare Your Input Data:**
    Your data should be in a Pandas DataFrame with at least these columns:
    *   `group_id`: A unique integer identifier for each group (0 to `num_groups - 1`).
    *   `group_size`: The size of each group.
    *   `choice_X`: Columns representing group preferences (e.g., `choice_0`, `choice_1`, ...). Values should be slot IDs (1-based or 0-based, be consistent). The `SchedulingProblem` base class expects 1-based slot IDs in choice columns.

2.  **Create a Custom Problem Class:**
    Inherit from `core.problem.SchedulingProblem` and implement the required methods.

    ```python
    # my_custom_problem.py
    import numpy as np
    from numba import njit
    from core.problem import SchedulingProblem
    from core.chromosome import ChromosomeType

    class MyCustomProblem(SchedulingProblem):
        def build_penalty_array(self) -> np.ndarray:
            # Define how penalties are calculated for assigning a group of a certain size
            # to a choice of a certain rank, or to an unlisted choice.
            # This array is used by the default initializer and can be used by your evaluator.
            # Shape: [max_group_size + 1, num_choices + 1]
            # The last column (num_choices) is for unlisted/non-preferred slots.
            num_choices = len(self.choice_cols)
            max_group_size = self.data["group_size"].max() # Or however you determine it
            penalty_array = np.zeros((max_group_size + 1, num_choices + 1), dtype=np.float64)

            for size in range(max_group_size + 1):
                for rank in range(num_choices): # 0 to num_choices-1
                    penalty_array[size][rank] = (rank + 1) * size # Example: penalty increases with rank and size
                # Penalty for unlisted choice (rank = num_choices)
                penalty_array[size][num_choices] = 100 * size # Example: high penalty for unlisted
            return penalty_array

        def evaluate(self, chromosome: ChromosomeType) -> float:
            # This method can call a Numba-jitted function for performance.
            # It's passed to the GA's run method.
            # You need to ensure the Numba function signature matches what `run_ga_numba` expects
            # if you use the default GA runner.
            return evaluate_my_custom_problem_numba(
                self.num_slots,
                chromosome,
                self.slots_min,
                self.slots_max,
                self.group_sizes, # or self.group_size_ls for a list
                self.choice_rank, # Precomputed in SchedulingProblem
                self.penalties_array # From build_penalty_array
                # Add any other data your Numba evaluator needs
            )

    @njit
    def evaluate_my_custom_problem_numba(
            num_slots: int,
            chromosome: ChromosomeType,
            slots_min: np.ndarray,
            slots_max: np.ndarray,
            group_sizes: np.ndarray,
            choice_rank: np.ndarray, # Shape: [num_groups, num_slots], value is rank or -1
            penalties_array: np.ndarray
            # ... other params ...
    ) -> float:
        assigned_slots, slot_occupancy_from_chromosome = chromosome # slot_occupancy might be pre-calculated or calculated here
        
        # Recalculate or verify slot_occupancy if needed
        # The chromosome passed to evaluate usually has slot_occupancy pre-calculated by
        # initialization/crossover/mutation. If your evaluation logic needs to be absolutely sure
        # or if those operators might not perfectly maintain it, recalculate it.
        # However, the provided GA operators *do* update slot_occupancy.
        
        current_slot_occupancy = np.zeros(num_slots, dtype=np.int32)
        total_penalty = 0.0

        # 1. Preference Penalties
        for group_idx in range(len(assigned_slots)):
            slot_assigned = assigned_slots[group_idx] # 1-based from problem
            slot_idx_0based = slot_assigned - 1
            group_s = group_sizes[group_idx]
            
            current_slot_occupancy[slot_idx_0based] += group_s

            rank = choice_rank[group_idx, slot_idx_0based]
            if rank == -1: # Unlisted choice
                total_penalty += penalties_array[group_s, penalties_array.shape[1] - 1] # Last col for unlisted
            else:
                total_penalty += penalties_array[group_s, rank]

        # 2. Slot Occupancy Violation Penalties
        for slot_idx_0based in range(num_slots):
            if current_slot_occupancy[slot_idx_0based] < slots_min[slot_idx_0based] or \
               current_slot_occupancy[slot_idx_0based] > slots_max[slot_idx_0based]:
                total_penalty += 1_000_000  # Large penalty

        # 3. Other Custom Penalties (e.g., fairness, distribution, etc.)
        # ... add your logic here ...

        return total_penalty
    ```

3.  **Update `main.py` (or create your own script):**
    *   Instantiate your custom problem class.
    *   Pass your custom Numba-fied evaluation function and (optionally) a custom Numba-fied initialization function to `ga.run()`.

    ```python
    # your_main_script.py
    import pandas as pd
    from core.ga import GeneticAlgorithm
    from core.problem import initialize_chromosome_numba # Default Numba initializer
    from my_custom_problem import MyCustomProblem, evaluate_my_custom_problem_numba

    df = pd.read_csv('path/to/your_custom_data.csv')
    # Potentially rename columns if they don't match 'group_id', 'group_size'
    # df.rename(columns={"id": "group_id", "size": "group_size"}, inplace=True)

    problem = MyCustomProblem(
        data=df,
        num_slots=50,       # Your number of slots
        min_occupancy=10,   # Your min occupancy per slot
        max_occupancy=30    # Your max occupancy per slot
    )

    ga = GeneticAlgorithm(problem)
    assigned_slots, best_cost = ga.run(
        initialize_chromosome_func=initialize_chromosome_numba, # Use default or your custom one
        evaluate_func=evaluate_my_custom_problem_numba,     # Your custom Numba evaluator
        # ... other GA parameters ...
    )
    print(f"Best assignment: {assigned_slots}, Cost: {best_cost}")
    ```
