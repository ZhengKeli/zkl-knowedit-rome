from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike


def stack_with_padding(
    arrays: Iterable[np.ndarray],
    pad: ArrayLike,
) -> np.ndarray:
    arrays = tuple(arrays)
    max_len = max(len(array) for array in arrays)

    arrays_padded = []
    for array in arrays:
        array_pad = np.full([max_len - len(array), *array.shape[1:]], pad, dtype=array.dtype)
        array_padded = np.concatenate([array, array_pad])
        arrays_padded.append(array_padded)

    return np.stack(arrays_padded)
