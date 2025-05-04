import numpy as np
from numba import njit

from core.chromosome import ChromosomeType
from core.problem import SchedulingProblem


class SantaSchedulingProblem(SchedulingProblem):
    """
    Specialized scheduling problem for Santa Workshop Optimization problem.
    This subclass defines how penalties are calculated based on group sizes and slot preferences.
    """

    def build_penalty_array(self) -> np.ndarray:
        """
        Constructs a penalty lookup table based on group sizes and ranking of choices.

        Returns:
            np.ndarray: A 2D array where each row corresponds to a group size,
                        and each column corresponds to a choice rank (0–10).
                        The cell [n, r] contains the penalty for assigning a group
                        of size `n` to their `r`-th choice.
        """
        num_choices = len(self.choice_cols)
        max_group_size = max(self.group_size_ls)
        penalty_array = np.zeros((max_group_size + 1, num_choices + 1), dtype=np.float64)

        for n in range(max_group_size + 1):
            # Penalty formula defined per problem specifications
            penalty_array[n] = [
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

        return penalty_array

    def evaluate(self, chromosome: ChromosomeType) -> float:
        """
        Evaluates a chromosome (i.e., a possible solution) using the numba-optimized function.

        Args:
            chromosome (ChromosomeType): A tuple of assigned slots and slot occupancies.

        Returns:
            float: The computed penalty (lower is better).
        """
        return evaluate_numba(
            self.num_slots, chromosome,
            self.slots_min, self.slots_max,
            self.group_size_ls, self.choice_rank, self.penalties_array
        )


@njit
def evaluate_numba(
        num_slots: int,
        chromosome: ChromosomeType,
        slots_min: np.ndarray,
        slots_max: np.ndarray,
        group_size_ls: np.ndarray,
        choice_rank: np.ndarray,
        penalties_array: np.ndarray
) -> float:
    """
    Numba-accelerated evaluation function for computing the total cost (penalty) of a schedule.

    Args:
        num_slots (int): Total number of available time slots.
        chromosome (ChromosomeType): Tuple of assigned slots and slot occupancies.
        slots_min (np.ndarray): Minimum allowed occupancy per slot.
        slots_max (np.ndarray): Maximum allowed occupancy per slot.
        group_size_ls (np.ndarray): Array of group sizes.
        choice_rank (np.ndarray): Matrix indicating the rank of each slot for each group (-1 if not chosen).
        penalties_array (np.ndarray): Penalty lookup table for group size and rank.

    Returns:
        float: Total penalty for the given chromosome.
    """
    assigned_slots, _ = chromosome
    daily_occupancy = np.zeros(num_slots, dtype=np.int32)
    penalty = 0.0

    # Compute preference penalties and build daily occupancy array
    for i in range(assigned_slots.shape[0]):
        d = assigned_slots[i] - 1  # Adjust to 0-based index
        n = group_size_ls[i]
        daily_occupancy[d] += n

        rank = choice_rank[i, d]
        if rank == -1:
            # Penalty for assigning to a slot outside the top 10 choices
            penalty += penalties_array[n, 10]
        else:
            # Penalty based on preference rank
            penalty += penalties_array[n, rank]

    # Penalize if slot occupancy is outside allowed limits
    for i in range(num_slots):
        if daily_occupancy[i] < slots_min[i] or daily_occupancy[i] > slots_max[i]:
            penalty += 1e8  # Large penalty for violating soft constraints

    # Accounting cost based on variation in daily occupancy
    acc = max(0, ((daily_occupancy[num_slots - 1] - 125.0) / 400.0) * (daily_occupancy[num_slots - 1] ** 0.5))
    yesterday = daily_occupancy[num_slots - 1]

    for i in range(num_slots - 2, -1, -1):
        today = daily_occupancy[i]
        diff = abs(today - yesterday)
        acc += max(0, ((today - 125.0) / 400.0) * (today ** (0.5 + diff / 50.0)))
        yesterday = today

    penalty += acc  # Add accounting cost to total penalty
    return penalty
