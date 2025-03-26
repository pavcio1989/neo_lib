import math
import neo_lib
import pytest


@pytest.mark.vec_norm
def test_vec_norm():
    vec_norm_res1 = neo_lib.general.vec.vec_norm(vector=[1.0, 0.0])
    assert vec_norm_res1 == 1.0

    vec_norm_res2 = neo_lib.general.vec.vec_norm(vector=[1.0, 1.0])
    assert vec_norm_res2 == math.sqrt(2)

    vec_norm_res3 = neo_lib.general.vec.vec_norm(vector=[5.0, 4.0])
    assert vec_norm_res3 == math.sqrt(41)

    vec_norm_res4 = neo_lib.general.vec.vec_norm(vector=[1.0, 1.0, 1.0])
    assert vec_norm_res4 == math.sqrt(3)

    vec_norm_res5 = neo_lib.general.vec.vec_norm(vector=[2.0, 4.0, -5.0, 6.0], norm='p2')
    assert vec_norm_res5 == 9.0


@pytest.mark.vec_dotprod
def test_vec_dotprod():
    dot_res1 = neo_lib.general.vec.vec_dotprod(vector1=[1.0, 2.0, 3.0],
                                               vector2=[-2.0, 5.0, 8.0])
    assert dot_res1 == 32.0


@pytest.mark.vec_angle
def test_vec_angle():
    angle_res1 = neo_lib.general.vec.vec_angle(vector1=[1.0, 0.0],
                                               vector2=[0.0, 1.0])
    assert angle_res1 == math.pi / 2.0
