import numpy as np
import pytest
from ravest.param import Parameter, Parameterisation


# First, test the underlying conversion functions

@pytest.mark.parametrize("e, w", [
    (e, w) for e in np.arange(0, 1, 0.1) for w in np.arange(-np.pi/2, np.pi/2 + np.pi/4, np.pi/4)
])
def test_convert_e_w_to_secosw_sesinw(e, w):
    para = Parameterisation("per k e w tp")
    expected_secosw = np.sqrt(e) * np.cos(w)
    expected_sesinw = np.sqrt(e) * np.sin(w)
    secosw, sesinw = para.convert_e_w_to_secosw_sesinw(e, w)
    assert np.isclose(secosw, expected_secosw)
    assert np.isclose(sesinw, expected_sesinw)

@pytest.mark.parametrize("e, w", [
    (e, w) for e in np.arange(0, 1, 0.1) for w in np.arange(-np.pi/2, np.pi/2 + np.pi/4, np.pi/4)
])
def test_convert_e_w_to_ecosw_esinw(e, w):
    para = Parameterisation("per k e w tp")
    expected_ecosw = e * np.cos(w)
    expected_esinw = e * np.sin(w)
    ecosw, esinw = para.convert_e_w_to_ecosw_esinw(e, w)
    assert np.isclose(ecosw, expected_ecosw)
    assert np.isclose(esinw, expected_esinw)

@pytest.mark.parametrize("ecosw, esinw", [
    (ecosw, esinw) for ecosw in np.arange(0, 1, 0.1) for esinw in np.arange(0, 1, 0.1)
])
def test_convert_ecosw_esinw_to_e_w(ecosw, esinw):
    para = Parameterisation("per k ecosw esinw tp")
    expected_e = np.sqrt(ecosw**2 + esinw**2)
    expected_w = np.arctan2(esinw, ecosw)
    e, w = para.convert_ecosw_esinw_to_e_w(ecosw, esinw) 
    assert np.isclose(e, expected_e)
    assert np.isclose(w, expected_w)

@pytest.mark.parametrize("secosw, sesinw", [
    (secosw, sesinw) for secosw in np.arange(0, 1, 0.1) for sesinw in np.arange(0, 1, 0.1)
])
def test_convert_secosw_sesinw_to_e_w(secosw, sesinw):
    para = Parameterisation("per k secosw sesinw tp")
    expected_e = secosw**2 + sesinw**2
    expected_w = np.arctan2(sesinw, secosw)
    e, w = para.convert_secosw_sesinw_to_e_w(secosw, sesinw) 
    assert np.isclose(e, expected_e)
    assert np.isclose(w, expected_w)

@pytest.mark.parametrize("tp", [
    tp for tp in np.arange(0, 10, 0.1)
])
def test_convert_tp_to_tc_circular(tp): # if e = 0 & w = pi/2, then tp = tc
    para = Parameterisation("per k e w tp")
    per = 10
    e = 0
    w = np.pi/2
    expected_tc = tp
    tc = para.convert_tp_to_tc(tp, per, e, w)
    assert expected_tc == tc

@pytest.mark.parametrize("tp, e, w, tc", [
    (0,    0.3, 3*np.pi/8, 0.32487717871429983),
    (3.33, 0.51, -np.pi/5, 5.200496945307864),
    (5,    0.69,        0, 5.493187444825672),
    (8.2, 0.8,    np.pi/7, 8.34625216953673)
])
def test_convert_tp_to_tc_eccentric(tp, e, w, tc):
    para = Parameterisation("per k e w tc")
    per = 10
    assert np.isclose(tc, para.convert_tp_to_tc(tp, per, e, w))

@pytest.mark.parametrize("tc", [
    tc for tc in np.arange(0, 10, 0.1)
])
def test_convert_tc_to_tp_circular(tc): # if e = 0 & w = pi/2, then tc = tp
    para = Parameterisation("per k e w tc")
    expected_tp = tc
    per = 10
    e = 0
    w = np.pi/2
    tp = para.convert_tc_to_tp(tc, per, e, w)
    assert expected_tp == tp

@pytest.mark.parametrize("tc, e, w, tp", [
    (0,    0.3, 3*np.pi/8, -0.32487717871429983),
    (3.33, 0.51, -np.pi/5, 1.459503054692136),
    (5,    0.69,        0, 4.506812555174328),
    (8.2, 0.8,    np.pi/7, 8.05374783046327)
])
def test_convert_tc_to_tp_eccentric(tc, e, w, tp):
    para = Parameterisation("per k e w tc")
    per = 10
    assert np.isclose(tp, para.convert_tc_to_tp(tc, per, e, w))


# Second, test the automatic conversion function, for each of the 
# parameterisations

ALLOWED_PARAMETERISATIONS = ["per k e w tp",
                             "per k e w tc",
                             "per k ecosw esinw tp",
                             "per k ecosw esinw tc",
                             "per k secosw sesinw tp",
                             "per k secosw sesinw tc"]

def test_invalid_parameterisation():
    with pytest.raises(Exception):
        Parameterisation("not a valid parameterisation")

