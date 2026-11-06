"""Exercise the real, time-bounded Math-Verify worker on Windows and POSIX."""
import pytest
from KGLMF.src.evaluation.grader import grade
from KGLMF.src.data.protocol import digest


@pytest.mark.parametrize('prediction,expected', [(r'FINAL: \boxed{\frac{1}{2}}',1),
                                                (r'FINAL: \boxed{3}',0), ('No answer.',0)])
def test_math_worker_correct_wrong_and_missing(prediction,expected):
    pytest.importorskip('math_verify')
    request=dict(id='unicode-数学',metric='external_math',prediction=prediction,
                 prediction_sha256=digest(prediction),answers=[r'\frac{1}{2}'])
    result=grade(request,None)
    assert result['score']==expected and result['id']==request['id']


def test_unparseable_gold_is_infrastructure_error():
    pytest.importorskip('math_verify')
    import subprocess
    prediction=r'\boxed{1}'
    with pytest.raises(subprocess.CalledProcessError):
        grade(dict(id='bad-gold',metric='external_math',prediction=prediction,
                   prediction_sha256=digest(prediction),answers=['']),None)


def test_escaped_currency_symbol_is_not_a_math_delimiter():
    pytest.importorskip('math_verify')
    prediction=r'FINAL: \boxed{18.90}'
    assert grade(dict(id='currency',metric='external_math',prediction=prediction,
                      prediction_sha256=digest(prediction),answers=[r'\$18.90']),None)['score']==1
