import numpy as np
from numba import njit

from core.problem import SchedulingProblem


class SantaSchedulingProblem(SchedulingProblem):

    def build_penalty_array(self) -> np.ndarray:
        num_choices = len(self.choice_cols)
        max_group_size = max(self.group_size_ls)
        penalty_array = np.zeros((max_group_size + 1, num_choices + 1), dtype=np.float64)

        for n in range(max_group_size + 1):
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

    def evaluate(self, assigned_slots: np.ndarray) -> float:
        return cost_function_numba(self.num_slots, assigned_slots, self.slots_min, self.slots_max, self.group_size_ls, self.choice_rank, self.penalties_array)


@njit
def cost_function_numba(
        num_slots: int,
        assigned_slots: np.ndarray,
        slots_min: np.ndarray,
        slots_max: np.ndarray,
        group_size_ls: np.ndarray,
        choice_rank: np.ndarray,
        penalties_array: np.ndarray
) -> float:

    daily_occupancy = np.zeros(num_slots, dtype=np.int32)
    penalty = 0.0

    # Preference penalties and daily occupancy
    for i in range(assigned_slots.shape[0]):
        d = assigned_slots[i] - 1  # adjust for 0-based index
        n = group_size_ls[i]
        daily_occupancy[d] += n

        rank = choice_rank[i, d]
        if rank == -1:
            penalty += penalties_array[n, 10]
        else:
            penalty += penalties_array[n, rank]

    # Soft constraints
    for i in range(num_slots):
        if daily_occupancy[i] < slots_min[i] or daily_occupancy[i] > slots_max[i]:
            penalty += 1e8

    # Accounting cost
    acc = max(0, ((daily_occupancy[num_slots - 1] - 125.0) / 400.0) * (daily_occupancy[num_slots - 1] ** 0.5))
    yesterday = daily_occupancy[num_slots - 1]

    for i in range(num_slots - 2, -1, -1):
        today = daily_occupancy[i]
        diff = abs(today - yesterday)
        acc += max(0, ((today - 125.0) / 400.0) * (today ** (0.5 + diff / 50.0)))
        yesterday = today

    penalty += acc  # Add accounting cost to total penalty
    return penalty
