import  numpy as np
from typing import Sized, Generator, Collection, Sequence, Union, Optional


def _get_all_dimensions(batch, level: int = 0, res = None):
    """Return all presented element sizes of each dimension.

    Args:
        batch: Data array.
        level: Recursion level.
        res: List containing element sizes of each dimension.

    Return:
        List, i-th element of which is list containing all presented sized of batch's i-th dimension.

    Examples:
        >>> x = [[[1], [2, 3]], [[4], [5, 6, 7], [8, 9]]]
        >>> _get_all_dimensions(x)
        [[2], [2, 3], [1, 2, 1, 3, 2]]

    """
    if not level:
        res = [[len(batch)]]
    if len(batch) and isinstance(batch[0], Sized) and not isinstance(batch[0], str):
        level += 1
        if len(res) <= level:
            res.append([])
        for item in batch:
            res[level].append(len(item))
            _get_all_dimensions(item, level, res)
    return res

def chunk_generator(items_list: list, chunk_size: int) -> Generator[list, None, None]:
    """Yields consecutive slices of list.

    Args:
        items_list: List to slice.
        chunk_size: Length of slice.

    Yields:
        list: ``items_list`` consecutive slices.

    """
    for i in range(0, len(items_list), chunk_size):
        yield items_list[i:i + chunk_size]


def get_dimensions(batch):
    """Return maximal size of each batch dimension."""
    return list(map(max, _get_all_dimensions(batch)))


def pad(batch,zp_batch= None, dtype = np.float32, padding = 0):
    """Fills the end of each array item to make its length maximal along each dimension.

    Args:
        batch: Initial array.
        zp_batch: Padded array.
        dtype = Type of padded array.
        padding = Number to will initial array with.

    Returns:
        Padded array.

    Examples:
        >>> x = np.array([[1, 2, 3], [4], [5, 6]])
        >>> zero_pad(x)
        array([[1., 2., 3.],
               [4., 0., 0.],
               [5., 6., 0.]], dtype=float32)

    """
    if zp_batch is None:
        dims = get_dimensions(batch)
        zp_batch = np.ones(dims, dtype=dtype) * padding
    if zp_batch.ndim == 1:
        zp_batch[:len(batch)] = batch
    else:
        for b, zp in zip(batch, zp_batch):
            pad(b, zp)
    return zp_batch


def pad_truncate(batch: Sequence[Sequence[Union[int, float, np.integer, np.floating,
                                                     Sequence[Union[int, float, np.integer, np.floating]]]]],
                      max_len: int, pad: str = 'post', trunc: str = 'post',
                      dtype: Optional[Union[type, str]] = None) -> np.ndarray:
    """

    Args:
        batch: assumes a batch of lists of word indexes or their vector representations
        max_len: resulting length of every batch item
        pad: how to pad shorter batch items: can be ``'post'`` or ``'pre'``
        trunc: how to truncate a batch item: can be ``'post'`` or ``'pre'``
        dtype: overrides dtype for the resulting ``ndarray`` if specified,
         otherwise ``np.int32`` is used for 2-d arrays and ``np.float32`` — for 3-d arrays

    Returns:
        a 2-d array of size ``(len(batch), max_len)`` or a 3-d array of size ``(len(batch), max_len, len(batch[0][0]))``
    """
    if isinstance(batch[0][0], Collection):  # ndarray behaves like a Sequence without actually being one
        size = (len(batch), max_len, len(batch[0][0]))
        dtype = dtype or np.float32
    else:
        size = (len(batch), max_len)
        dtype = dtype or np.int32

    padded_batch = np.zeros(size, dtype=dtype)
    for i, batch_item in enumerate(batch):
        if len(batch_item) > max_len:  # trunc
            padded_batch[i] = batch_item[slice(max_len) if trunc == 'post' else slice(-max_len, None)]
        else:  # pad
            padded_batch[i, slice(len(batch_item)) if pad == 'post' else slice(-len(batch_item), None)] = batch_item

    return np.asarray(padded_batch)