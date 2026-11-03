import math
import pytest
from ambifix_hitl.contracts import LABELS,ValidationError,valid_probabilities,validate_text,valid_label,uncertainty
from ambifix_hitl.predictors import MockPredictor


@pytest.mark.parametrize("value",["", " \n\t", None, 17, True, "x"*20001, "a\x00b", "a\ud800b"])
def test_invalid_requirement(value):
    with pytest.raises(ValidationError):validate_text(value)


def test_exact_text_and_boundary():
    text="  Mixed CASE é.\n\tEnd  "
    assert validate_text(text)==text
    assert len(validate_text("x"*20000))==20000


@pytest.mark.parametrize("value",[0,7,-1,"2",2.0,True,None])
def test_invalid_class(value):
    with pytest.raises(ValidationError):valid_label(value)


@pytest.mark.parametrize("values",[[0.2]*5,[0.2]*6,[1,0,0,0,0,float('nan')],[1,0,0,0,0,float('inf')],[-1,1,1,0,0,0],[True,0,0,0,0,0]])
def test_invalid_probabilities(values):
    with pytest.raises(ValidationError):valid_probabilities(values)


def test_entropy_bounds_and_no_calibration():
    assert uncertainty([1,0,0,0,0,0],0.75)["value"]==0
    u=uncertainty([1/6]*6,0.75)
    assert math.isclose(u["value"],1) and u["prioritized"] and not u["calibrated"]
    for threshold in [-.1,1.1,float('nan'),True]:
        with pytest.raises(ValidationError):uncertainty([1/6]*6,threshold)


def test_mock_is_explicit_deterministic_six_class():
    a=MockPredictor().predict("A brand-new mock rehearsal requirement.")
    b=MockPredictor().predict("A brand-new mock rehearsal requirement.")
    assert a==b and a.is_mock and a.model_id.startswith("mock-")
    assert len(a.probabilities)==6 and abs(sum(a.probabilities)-1)<1e-14
    assert LABELS[2]!=LABELS[4]
