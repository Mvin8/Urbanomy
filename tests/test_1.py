import pytest
from my_package.my_module import my_function

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

