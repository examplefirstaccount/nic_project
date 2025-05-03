import numpy as np
from numba import njit


class Chromosome:
    def __init__(self, num_slots: int, num_groups: int) -> None:
        self.num_slots = num_slots
        self.num_groups = num_groups
        self.assigned_slots = np.zeros(num_groups, dtype=np.int32)
        self.slot_occupancy = np.zeros(num_slots, dtype=np.int32)

    def copy(self) -> 'Chromosome':
        new_chromo = Chromosome(self.num_slots, self.num_groups)
        new_chromo.assigned_slots = self.assigned_slots.copy()
        new_chromo.slot_occupancy = self.slot_occupancy.copy()
        return new_chromo

    def is_swap_valid(self, group_idx: int, new_slot: int, group_sizes: list[int], slots_min: list, slots_max: list) -> bool:
        current_slot = self.assigned_slots[group_idx]
        group_size = group_sizes[group_idx]

        reduced = self.slot_occupancy[current_slot - 1] - group_size
        increased = self.slot_occupancy[new_slot - 1] + group_size

        reduce_valid = slots_min[current_slot - 1] <= reduced <= slots_max[current_slot - 1]
        increase_valid = slots_min[new_slot - 1] <= increased <= slots_max[new_slot - 1]

        return reduce_valid and increase_valid

    def update_occupancy(self, group_idx: int, old_slot: int, new_slot: int, group_sizes: list[int]) -> None:
        self.slot_occupancy[old_slot - 1] -= group_sizes[group_idx]
        self.slot_occupancy[new_slot - 1] += group_sizes[group_idx]


@njit
def is_swap_valid(
        group_idx: int,
        new_slot: int,
        assigned_slots: np.ndarray,
        slot_occupancy: np.ndarray,
        group_sizes: np.ndarray,
        slots_min: np.ndarray,
        slots_max: np.ndarray
) -> bool:
    current_slot = assigned_slots[group_idx]
    group_size = group_sizes[group_idx]

    reduced = slot_occupancy[current_slot - 1] - group_size
    increased = slot_occupancy[new_slot - 1] + group_size

    reduce_valid = slots_min[current_slot - 1] <= reduced <= slots_max[current_slot - 1]
    increase_valid = slots_min[new_slot - 1] <= increased <= slots_max[new_slot - 1]

    return reduce_valid and increase_valid


@njit
def update_occupancy(
        group_idx: int,
        old_slot: int,
        new_slot: int,
        slot_occupancy: np.ndarray,
        group_sizes: np.ndarray
) -> None:
    slot_occupancy[old_slot - 1] -= group_sizes[group_idx]
    slot_occupancy[new_slot - 1] += group_sizes[group_idx]
