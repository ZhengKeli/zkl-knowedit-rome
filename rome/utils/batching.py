from typing import Iterable, Iterator, Literal

import numpy as np
from numpy.typing import ArrayLike


def stack_with_aligning(
    arrays: Iterable[np.ndarray], *,
    size: int | Literal['max', 'min'] | None = None,
    pad: ArrayLike | None = None,
    return_mask: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    arrays = tuple(arrays)

    if size is None:
        if pad is None:
            size = 'min'
        else:
            size = 'max'

    if size == 'max':
        size = max(len(array) for array in arrays)
    if size == 'min':
        size = min(len(array) for array in arrays)

    arrays_aligned = []
    for array in arrays:
        if len(array) < size:
            array_pad = np.broadcast_to(np.asarray(pad, dtype=array.dtype), [size - len(array), *array.shape[1:]])
            array_aligned = np.concatenate([array, array_pad])
        elif len(array) > size:
            array_aligned = array[:size]
        else:
            array_aligned = array
        arrays_aligned.append(array_aligned)

    if not return_mask:
        return np.stack(arrays_aligned)

    arrays_mask = []
    for array in arrays:
        array_ones = np.ones_like(array, dtype=bool)
        array_zeros = np.zeros([size - len(array), *array.shape[1:]], dtype=bool)
        array_mask = np.concatenate([array_ones, array_zeros])
        arrays_mask.append(array_mask)
    return np.stack(arrays_aligned), np.stack(arrays_mask)


def iter_by_batch(
    arrays: Iterable[np.ndarray], *,
    pad: ArrayLike,
    batch_len: int,
    batch_size: int,
    return_mask: bool = False
) -> Iterator[np.ndarray] | Iterator[tuple[np.ndarray, np.ndarray]]:
    batch = []
    for array in arrays:
        batch.append(array)
        if len(batch) == batch_size:
            yield stack_with_aligning(batch, size=batch_len, pad=pad, return_mask=return_mask)
            batch.clear()
    if batch:
        yield stack_with_aligning(batch, size=batch_len, pad=pad, return_mask=return_mask)
