import numpy as np
from polytope import polytope

from .. import gp_reachability


def test__is_ellipsoid_inside_polytope__inside__returns_true():
    poly = polytope.box2poly(([[0, 10], [0, 10]]))
    A = poly.A
    b = np.expand_dims(poly.b, 1)
    p = np.array([[5, 5]]).T
    q = np.array([[2, 1], [1, 2]])
    assert gp_reachability.is_ellipsoid_inside_polytope(p, q, A, b) is True


def test__is_ellipsoid_inside_polytope__partially_out__returns_false():
    poly = polytope.box2poly(([[0, 10], [0, 10]]))
    A = poly.A
    b = np.expand_dims(poly.b, 1)
    p = np.array([[0, 0]]).T
    q = np.array([[2, 1], [1, 2]])
    assert gp_reachability.is_ellipsoid_inside_polytope(p, q, A, b) is False


def test__is_ellipsoid_inside_polytope__outside__returns_false():
    poly = polytope.box2poly(([[0, 10], [0, 10]]))
    A = poly.A
    b = np.expand_dims(poly.b, 1)
    p = np.array([[20, 20]]).T
    q = np.array([[2, 1], [1, 2]])
    assert gp_reachability.is_ellipsoid_inside_polytope(p, q, A, b) is False
