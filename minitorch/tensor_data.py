from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index (Index): A tuple representing the indices of the tensor.
        strides (Strides): A tuple of strides for each dimension of the tensor.

    Returns:
    -------
        int: The single-dimensional position in storage.

    """
    # ASSIGN2.1
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position
    # END ASSIGN2.1


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal (int): The position to convert to an index.
        shape (Shape): A tuple representing the shape of the tensor.
        out_index (OutIndex): The output index corresponding to the ordinal position.

    Returns:
    -------
        None

    """
    # ASSIGN2.1
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh
    # END ASSIGN2.1


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index (Index): Multidimensional index of the larger tensor.
        big_shape (Shape): The shape of the larger tensor.
        shape (Shape): The shape of the smaller tensor.
        out_index (OutIndex): The resulting index in the smaller tensor.

    Returns:
    -------
        None

    """
    # ASSIGN2.2
    for t, s in enumerate(shape):
        if s > 1:
            out_index[t] = big_index[t + (len(big_shape) - len(shape))]
        else:
            out_index[t] = 0
    return None
    # END ASSIGN2.2


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 (UserShape): The shape of the first tensor.
        shape2 (UserShape): The shape of the second tensor.

    Returns:
    -------
        UserShape: The broadcasted shape.

    Raises:
    ------
        IndexingError: If the two shapes cannot be broadcasted according to broadcasting rules.

    """
    smaller_shape = None
    larger_shape = None

    # Determine smaller and larger shapes based on their length
    if len(shape1) > len(shape2):
        smaller_shape = shape2
        larger_shape = shape1
    else:
        smaller_shape = shape1
        larger_shape = shape2

    result_shape = ()  # Initialize an empty tuple for the broadcasted shape

    # Iterate over each dimension of the larger shape
    for i in range(len(larger_shape)):
        # Get the dimension size from the larger and smaller shapes
        larger_dim_size = larger_shape[i]
        if i < len(larger_shape) - len(smaller_shape):
            smaller_dim_size = larger_shape[i]
        else:
            smaller_dim_size = smaller_shape[
                i - (len(larger_shape) - len(smaller_shape))
            ]

        # Ensure larger_dim_size is the larger dimension, and smaller_dim_size is the smaller
        if larger_dim_size >= smaller_dim_size:
            larger_dim = larger_dim_size
            smaller_dim = smaller_dim_size
        else:
            larger_dim = smaller_dim_size
            smaller_dim = larger_dim_size

        # Check if the larger dimension is divisible by the smaller dimension
        if larger_dim % smaller_dim != 0:
            raise IndexingError(
                f"Cannot broadcast dimension {i}: {larger_dim} % {smaller_dim} != 0."
            )
        else:
            # Append the larger dimension to the result shape
            result_shape += (larger_dim,)

    return result_shape
    # TODO: Implement for Task 2.2.
    # raise NotImplementedError("Need to implement for Task 2.2")


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Calculate the strides required for a given shape in a contiguous layout.

    Args:
    ----
        shape (UserShape): The shape of the tensor.

    Returns:
    -------
        UserStrides: A tuple of strides for each dimension.

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """A class to represent the data structure of a Tensor with
    multidimensional indexing and storage.

    Attributes
    ----------
        _storage (Storage): The underlying storage of the tensor data.
        _strides (Strides): The strides for each dimension.
        _shape (Shape): The shape of the tensor.
        strides (UserStrides): User-accessible strides.
        shape (UserShape): User-accessible shape.
        dims (int): Number of dimensions in the tensor.

    Methods
    -------
        to_cuda_(): Convert the storage to a CUDA device.
        is_contiguous(): Check if the tensor is stored contiguously.
        shape_broadcast(): Static method to compute the broadcasted shape of two shapes.
        index(): Compute the flattened index for a given multidimensional index.
        indices(): Generate indices for the elements in the tensor.
        sample(): Get a random valid index within the tensor.
        get(): Retrieve the value associated with a given key.
        set(): Set a value for a specific key.
        tuple(): Return core tensor data as a tuple.
        permute(): Permute the dimensions of the tensor.
        to_string(): Convert tensor data to a string representation.

    """

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initialize the TensorData object with storage, shape, and optional strides.

        Args:
        ----
            storage (Union[Sequence[float], Storage]): The storage of tensor data.
            shape (UserShape): The shape of the tensor.
            strides (Optional[UserStrides], optional): The strides for each dimension.
                If not provided, they are calculated from the shape. Defaults to None.

        Raises:
        ------
            IndexingError: If the length of strides does not match the shape.

        """
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert the tensor storage to CUDA."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check if the layout is contiguous, i.e., outer dimensions have bigger
        strides than inner dimensions.

        Returns
        -------
            bool: True if the tensor is stored contiguously.

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Computes the broadcasted shape of two shapes for element-wise operations.

        Args:
        ----
            shape_a (UserShape): The shape of the first tensor.
            shape_b (UserShape): The shape of the second tensor.

        Returns:
        -------
            UserShape: The resulting shape after applying broadcasting rules.

        Notes:
        -----
            Broadcasting allows tensors of different shapes to be used in
            arithmetic operations. This method follows the standard broadcasting
            rules used in numerical computing libraries.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Computes the flattened index for the given input index in the tensor.

        Args:
        ----
            index (Union[int, UserIndex]): The index (or tuple of indices) to convert to a flattened index.

        Returns:
        -------
            int: The flattened index corresponding to the given input index.

        Raises:
        ------
            IndexingError: If the provided index is not the correct size or is out of bounds.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generates the indices for the elements in the tensor based on its shape.

        Yields
        ------
            Iterable[UserIndex]: A tuple representing the indices of each element in the tensor.

        Notes
        -----
            The indices are generated in a flattened order based on the tensor's shape.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index within the tensor.

        Returns
        -------
            UserIndex: A random index in the tensor's shape.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieves the value associated with the given key.

        Args:
        ----
            key (UserIndex): The key (or index) for which to retrieve the value.

        Returns:
        -------
            float: The value associated with the specified key.

        Raises:
        ------
            KeyError: If the key is not found in the storage.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Sets the value for the specified key.

        Args:
        ----
            key (UserIndex): The key (or index) for which to set the value.
            val (float): The value to associate with the specified key.

        Raises:
        ------
            KeyError: If the key is not found in the storage.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple.

        Returns
        -------
            Tuple[Storage, Shape, Strides]: A tuple containing storage, shape, and strides of the tensor.

        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            order (list): A permutation of the dimensions.

        Returns:
        -------
            TensorData: A new `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"
        # ASSIGN2.1
        return TensorData(
            self._storage,
            tuple([self.shape[o] for o in order]),
            tuple([self._strides[o] for o in order]),
        )

    def to_string(self) -> str:
        """Convert the tensor data to a string representation.

        Returns
        -------
            str: A string representation of the tensor data.

        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
