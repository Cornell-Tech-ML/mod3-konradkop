from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Protocol for a variable in a computational graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable.

        Args:
        ----
            x (Any): The derivative value to be accumulated.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Get the unique identifier for this variable.

        Returns
        -------
            int: A unique identifier for the variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf in the computational graph.

        Returns
        -------
            bool: True if the variable is a leaf, otherwise False.

        """
        ...

    def is_constant(self) -> bool:
        """Check if this variable is constant.

        Returns
        -------
            bool: True if the variable is constant, otherwise False.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of this variable.

        Returns
        -------
            Iterable[Variable]: An iterable of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute derivatives with respect to parent variables.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing
            parent variables and their corresponding gradients.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    result = []

    def depthFirstSearch(var: Variable) -> None:
        if var.unique_id in visited:
            return
        if var.is_constant():
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            depthFirstSearch(parent)
        result.append(var)

    depthFirstSearch(variable)
    # reversing cause its used backwards
    result.reverse()
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute
    derivatives for the leaf nodes.

    Args:
    ----
    variable : Variable
        The right-most variable in the computation graph from which
        backpropagation will start. This variable represents the output
        of the graph.
    deriv : Any
        The derivative value that we want to propagate backward to the
        leaf nodes. This typically represents the gradient of the loss
        with respect to the output variable.

    Returns:
    -------
    None
        This function does not return a value. Instead, it writes the
        computed derivatives to the derivative values of each leaf
        variable through the `accumulate_derivative` method.

    """
    # Perform a topological sort on the computation graph starting from the given variable.
    topoSort = topological_sort(variable)

    # Initialize a dictionary to hold derivatives, starting with the given variable's unique ID and its derivative.
    dict = {variable.unique_id: deriv}

    # Iterate through each scalar in the topologically sorted order.
    for scal in topoSort:
        # If the scalar's unique ID is not in the dictionary, skip to the next scalar.
        if scal.unique_id not in dict:
            continue

        # If the scalar is a leaf node, skip further processing for this scalar.
        if scal.is_leaf():
            continue

        # Retrieve the derivative for the current scalar from the dictionary.
        # This is the derivative that will be propagated backward.
        if scal.unique_id in dict:
            deriv = dict[scal.unique_id]

        # Get the current derivative for the scalar being processed.
        curr_deriv = dict[scal.unique_id]

        # Compute the new derivative for the current scalar using the chain rule.
        deriv = scal.chain_rule(curr_deriv)

        # For each child variable of the current scalar and its corresponding derivative.
        for child, d in deriv:
            # If the child is a leaf node, accumulate its derivative.
            if child.is_leaf():
                child.accumulate_derivative(d)
            else:
                # If the child's unique ID is already in the dictionary, add the derivative.
                if child.unique_id in dict:
                    dict[child.unique_id] += d
                else:
                    # Otherwise, initialize the child's derivative in the dictionary.
                    dict[child.unique_id] = d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values"""
        return self.saved_values


# konradChangedThisFile
