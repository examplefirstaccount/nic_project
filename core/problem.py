from abc import ABC, abstractmethod
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd
from numba import njit

from core.chromosome import ChromosomeType

Numeric = Union[Number, np.number]


class SchedulingProblem(ABC):
    """
    Abstract base class for scheduling problems.

    Defines input structure, constraints, and initialization for group-slot assignment problems.
    """

    REQUIRED_COLUMNS = {"group_id", "group_size"}

    def __init__(
            self,
            data: pd.DataFrame,
            num_slots: int,
            min_occupancy: Union[Numeric, np.ndarray],
            max_occupancy: Union[Numeric, np.ndarray]
    ):
        """
        Initializes problem with group-slot preferences and capacity constraints.

        Args:
            data: DataFrame with columns: 'group_id', 'group_size', and 'choice_X' for preferred slots.
            num_slots: Total number of available time slots.
            min_occupancy: Minimum required capacity per slot (scalar or array of shape [num_slots]).
            max_occupancy: Maximum allowed capacity per slot (scalar or array of shape [num_slots]).
        """
        self.data = data.copy()
        self.num_slots = num_slots
        self.num_groups = self.data.shape[0]

        self._validate_input_data()

        self.slots_min = self._init_slot_bounds(min_occupancy, name="min_occupancy")
        self.slots_max = self._init_slot_bounds(max_occupancy, name="max_occupancy")

        self.group_sizes = self.data["group_size"].to_numpy()
        self.group_size_dict = self.data.set_index("group_id")["group_size"].to_dict()
        self.group_size_ls = [self.group_size_dict[i] for i in range(self.num_groups)]

        self.choice_cols = sorted([col for col in self.data.columns if col.startswith("choice_")])
        self.choice_matrix = self.data[self.choice_cols].to_numpy()

        # Each group's preferences mapped: {slot -> rank}
        self.choice_dict_num = [
            {day: rank for rank, day in enumerate(row)}
            for row in self.choice_matrix
        ]

        # For Numba-optimized access:
        self.choice_dict_items = np.array([np.array(list(d.keys()), dtype=np.int32) for d in self.choice_dict_num])
        self.choice_rank = -np.ones((self.num_groups, self.num_slots), dtype=np.int8)
        for i in range(self.choice_matrix.shape[0]):
            for rank, day in enumerate(self.choice_matrix[i]):
                self.choice_rank[i, day - 1] = rank

        # Penalties for cost function and custom initialization
        self.penalties_array = np.array(self.build_penalty_array())

    def _validate_input_data(self) -> None:
        """Ensures required columns and at least one choice column are present."""
        missing = self.REQUIRED_COLUMNS - set(self.data.columns)
        choice_cols = [col for col in self.data.columns if col.startswith("choice_")]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")
        if not choice_cols:
            raise ValueError("No 'choice_X' columns found in input data.")

    def _init_slot_bounds(self, bounds: Union[Numeric, np.ndarray], name: str) -> np.ndarray:
        """
        Initializes slot capacity bounds from scalar or array input.

        Args:
            bounds: Either a single number or an array for per-slot limits.
            name: Name used in error messages.

        Returns:
            bounds as an int ndarray of shape [num_slots].
        """
        if isinstance(bounds, (int, float, np.number)):
            return np.full(self.num_slots, bounds)
        elif isinstance(bounds, np.ndarray):
            if bounds.shape[0] != self.num_slots:
                raise ValueError(f"{name} must have shape ({self.num_slots},), got {bounds.shape}")
            return bounds.astype(int)
        else:
            raise TypeError(f"{name} must be a numeric value or NumPy array.")

    def build_penalty_array(self) -> np.ndarray:
        """
        Builds a penalty matrix: [group_size, rank] -> penalty value.

        Default logic penalizes:
        - Low-ranked preferences (linearly),
        - Unlisted (non-preferred) slots (heavily).

        Returns:
            penalty_array of shape [max_group_size + 1, num_choices + 1]
        """
        num_choices = len(self.choice_cols)
        max_group_size = max(self.group_size_ls)
        penalty_array = np.zeros((max_group_size + 1, num_choices + 1), dtype=np.int32)

        for size in range(max_group_size + 1):
            # Skip the first rank and keep its penalty as 0
            for rank in range(1, num_choices):
                penalty_array[size][rank] = 2 * rank + size

            # Give higher penalty for the not preferred slots
            penalty_array[size][num_choices] = 10 * num_choices + size

        return penalty_array

    def initialize_chromosome(self) -> ChromosomeType:
        """
        Initializes a valid chromosome using weighted random choice based on penalties.

        Returns:
            Tuple of (assigned_slots, slot_occupancy)
        """
        return initialize_chromosome_numba(self.num_groups, self.num_slots, self.group_sizes, self.slots_min, self.slots_max, self.choice_dict_items, self.penalties_array)

    @abstractmethod
    def evaluate(self, chromosome: ChromosomeType) -> float:
        """
        Abstract method: evaluates cost of a chromosome.

        Must be implemented in a concrete subclass.
        """
        pass


@njit
def initialize_chromosome_numba(
        num_groups: int,
        num_slots: int,
        group_sizes: np.ndarray,
        slots_min: np.ndarray,
        slots_max: np.ndarray,
        choice_dict_items: np.ndarray,
        penalties_array: np.ndarray
) -> ChromosomeType:
    """
    Fast chromosome initializer using Numba.

    For each group:
    - Combines their preferred slots with one random non-preferred slot
    - Computes inverse-penalty-based weights
    - Chooses a feasible slot based on those weights

    Returns:
        Tuple: (assigned_slots, slot_occupancy)
    """
    assigned_slots = np.zeros(num_groups, dtype=np.int32)
    slot_occupancy = np.zeros(num_slots, dtype=np.int32)

    for i in range(num_groups):
        # Add one random non-preferred slot
        preferred_slots = choice_dict_items[i]
        all_slots = np.empty(len(preferred_slots) + 1, dtype=np.int32)
        all_slots[:-1] = preferred_slots

        # Find a random non-preferred slot
        while True:
            new_slot = np.random.randint(1, num_slots + 1)
            if new_slot not in preferred_slots:
                all_slots[-1] = new_slot
                break

        # Compute inverse penalty weights for all options
        inverse_weights = np.zeros(len(all_slots), dtype=np.float64)
        for j in range(len(all_slots)):
            slot_idx = all_slots[j] - 1
            penalty = 50

            current_occ = slot_occupancy[slot_idx]
            if slots_min[slot_idx] <= current_occ <= slots_max[slot_idx]:
                penalty += (current_occ - slots_min[slot_idx]) * 1000
            elif current_occ > slots_max[slot_idx]:
                penalty = 1_000_000_000

            inverse_weights[j] = 1.0 / (penalties_array[group_sizes[i], j] + penalty)

        # Mask infeasible slots (would exceed max capacity)
        feasible_mask = np.array([
            (slot_occupancy[slot - 1] + group_sizes[i] <= slots_max[slot - 1])
            for slot in all_slots
        ])

        if np.any(feasible_mask):
            feasible_slots = all_slots[feasible_mask]
            feasible_weights = inverse_weights[feasible_mask]
            weights_norm = feasible_weights / np.sum(feasible_weights)

            # Weighted random selection
            chosen_slot = feasible_slots[
                np.argmax(np.cumsum(weights_norm) >= np.random.random())
            ]
        else:
            chosen_slot = 1  # Fallback if nothing is feasible

        assigned_slots[i] = chosen_slot
        slot_occupancy[chosen_slot - 1] += group_sizes[i]

    return assigned_slots, slot_occupancy
