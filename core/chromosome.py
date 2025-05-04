from typing import Tuple

import numpy as np

# A chromosome is represented as a tuple:
# - assigned_slots: ndarray of slot assignment per group (shape: [num_groups])
# - slot_occupancy: ndarray of total size per slot (shape: [num_slots])
ChromosomeType = Tuple[np.ndarray, np.ndarray]
