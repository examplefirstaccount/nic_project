from abc import ABC, abstractmethod
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd

from core.chromosome import Chromosome


Numeric = Union[Number, np.number]


class SchedulingProblem(ABC):
    REQUIRED_COLUMNS = {"group_id", "group_size"}

    def __init__(
            self,
            data: pd.DataFrame,
            num_slots: int,
            min_occupancy: Union[Numeric, np.ndarray],
            max_occupancy: Union[Numeric, np.ndarray]
    ):
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
        self.choice_dict_num = [
            {day: rank for rank, day in enumerate(row)}
            for row in self.choice_matrix
        ]

        self.penalties_array = self.build_penalty_array()

    def _validate_input_data(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self.data.columns)
        choice_cols = [col for col in self.data.columns if col.startswith("choice_")]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")
        if not choice_cols:
            raise ValueError("No 'choice_X' columns found in input data.")

    def _init_slot_bounds(self, bounds: Union[Numeric, np.ndarray], name: str) -> np.ndarray:
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
        Dummy implementation of penalty array builder.
        Override in a subclass if your cost function needs a more specific version.
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

    def initialize_chromosome(self) -> Chromosome:
        """
        Creates an initial Chromosome with valid group-to-day assignments.

        Each group is assigned a day based on their preferences and current slot occupancy.
        One extra random (non-preferred) day is added as an option.
        Days are selected using weighted random choice, where lower penalty means higher chance.
        Assignments respect the max slot capacity.

        Override in a subclass if you will need a more specific version.

        Returns:
            Chromosome: The initial assignment of groups to days with slot occupancies.
        """
        amount_in_slot = {slot: 0 for slot in range(1, self.num_slots + 1)}
        chromosome = np.zeros(self.num_groups, dtype=np.int32)

        for i in range(self.num_groups):
            items = list(self.choice_dict_num[i].keys()).copy()
            items_set = set(items)

            # Add one random non-preferred slot
            while True:
                new_slot = np.random.randint(1, self.num_slots + 1)
                if new_slot not in items_set:
                    items.append(new_slot)
                    items_set.add(new_slot)
                    break

            weights = self.penalties_array[self.group_size_dict[i]]
            inverse_weights = []
            for j in range(len(weights)):
                penalty = 50
                current_occupancy = amount_in_slot[items[j]]
                slot_min = self.slots_min[items[j] - 1]
                slot_max = self.slots_max[items[j] - 1]

                # Additional penalty based on current slot occupancy to decrease choosing the most occupied days
                if slot_min <= current_occupancy <= slot_max:
                    penalty += (current_occupancy - slot_min) * 1000
                elif current_occupancy > slot_max:
                    penalty = 1_000_000_000

                # Final weight = inverse of (base penalty + occupancy penalty) < 1
                inverse_weights.append(1 / (weights[j] + penalty))

            # Filter to only feasible slot options (wouldn't violate restrictions)
            new_items = []
            new_inverse_weights = []
            for j in range(len(items)):
                current_occupancy = amount_in_slot[items[j]]
                slot_max = self.slots_max[items[j] - 1]

                if current_occupancy + self.group_size_ls[i] <= slot_max:
                    new_items.append(items[j])
                    new_inverse_weights.append(inverse_weights[j])

            # Select a slot probabilistically based on weights
            probabilities = np.array(new_inverse_weights) / np.sum(new_inverse_weights)
            chosen_slot = np.random.choice(new_items, size=1, p=probabilities)[0]
            chromosome[i] = chosen_slot
            amount_in_slot[chosen_slot] += self.group_size_ls[i]

        # Construct Chromosome object
        chromosome_obj = Chromosome(self.num_slots, self.num_groups)
        chromosome_obj.assigned_slots = chromosome
        for slot, amount in amount_in_slot.items():
            chromosome_obj.slot_occupancy[slot - 1] = amount

        return chromosome_obj

    @abstractmethod
    def evaluate(self, chromosome: Chromosome) -> float:
        """
        Abstract method to compute the cost of a chromosome's assigned slots.
        Must be implemented by subclasses.
        """
        pass
