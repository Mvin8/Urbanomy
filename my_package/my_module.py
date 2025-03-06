"""
This module provides an example class and a function for basic arithmetic operations.
"""

class MyClass:
    """
    A class that represents a simple example with an integer argument.

    Parameters
    ----------
    my_arg : int
        An integer argument that will be stored as an instance attribute.

    Attributes
    ----------
    my_arg : int
        The integer argument provided during initialization.
    """

    def __init__(self, my_arg: int):
        self.my_arg = my_arg

def my_function(a: int, b: int) -> int:
    """
    Add two integers together.

    Parameters
    ----------
    a : int
        The first integer to be added.
    b : int
        The second integer to be added.

    Returns
    -------
    int
        The sum of the two input integers.
    """
    return a + b