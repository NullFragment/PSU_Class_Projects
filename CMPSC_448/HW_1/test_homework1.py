###################################
### CMPSC 448 - Spring 2018
### Homework 1
###################################

import pytest
import pickle
from homework1 import *

f = open('data.pck', 'rb')
tests = pickle.load(f)
f.close()


@pytest.mark.parametrize('inp, expected', [i for i in tests['q1']])
def test_q1(inp, expected):
    blah = q1(inp)
    print(blah)
    assert blah == pytest.approx(expected, 0.005)


# @pytest.mark.parametrize('inp, expected', [i for i in tests['q2']])
# def test_q2(inp, expected):
#     assert q2(inp) == pytest.approx(expected, 0.0000005)
#
#
# @pytest.mark.parametrize('inp, expected', [i for i in tests['q3']])
# def test_q3(inp, expected):
#     assert q3(inp) == pytest.approx(expected, 0.0005)
