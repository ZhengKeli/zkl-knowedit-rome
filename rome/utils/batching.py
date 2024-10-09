from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike


def stack_with_padding(
    arrays: Iterable[np.ndarray],
    pad: ArrayLike, *,
    return_mask: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    arrays = tuple(arrays)
    max_len = max(len(array) for array in arrays)

    arrays_padded = []
    for array in arrays:
        array_pad = np.full([max_len - len(array), *array.shape[1:]], pad, dtype=array.dtype)
        array_padded = np.concatenate([array, array_pad])
        arrays_padded.append(array_padded)

    if not return_mask:
        return np.stack(arrays_padded)

    arrays_mask = []
    for array in arrays:
        array_ones = np.ones_like(array, dtype=bool)
        array_zeros = np.zeros([max_len - len(array), *array.shape[1:]], dtype=bool)
        array_mask = np.concatenate([array_ones, array_zeros])
        arrays_mask.append(array_mask)
    return np.stack(arrays_padded), np.stack(arrays_mask)
