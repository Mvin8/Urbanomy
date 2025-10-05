import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from methods.my_module import my_function

@pytest.fixture
def a():
    return 2

@pytest.fixture
def b():
    return 2

@pytest.fixture
def c():
    return 4

def test_my_function(a, b, c):
    assert my_function(a, b) == c

