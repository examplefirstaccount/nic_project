import numpy as np

from core.chromosome import Chromosome
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

    def evaluate(self, chromosome: Chromosome) -> float:
        penalty = 0

        days = list(range(self.num_slots, 0, -1))
        daily_occupancy = {k: 0 for k in days}

        for n, d, choice in zip(self.group_size_ls, chromosome.assigned_slots, self.choice_dict_num):
            daily_occupancy[d] += n

            if d not in choice:
                penalty += self.penalties_array[n][-1]
            else:
                penalty += self.penalties_array[n][choice[d]]

        for occupancy, min_occ, max_occ in zip(chromosome.slot_occupancy, self.slots_min, self.slots_max):
            if occupancy < min_occ or occupancy > max_occ:
                penalty += 100_000_000

        accounting_cost = (daily_occupancy[days[0]] - 125.0) / 400.0 * daily_occupancy[days[0]] ** 0.5
        accounting_cost = max(0, accounting_cost)

        yesterday_count = daily_occupancy[days[0]]
        for day in days[1:]:
            today_count = daily_occupancy[day]
            diff = abs(today_count - yesterday_count)
            accounting_cost += max(0,
                                   (daily_occupancy[day] - 125.0) / 400.0 * daily_occupancy[day] ** (0.5 + diff / 50.0))
            yesterday_count = today_count

        penalty += accounting_cost

        return penalty
