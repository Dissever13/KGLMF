"""JSON-in/JSON-out task scorer. Only an explicit call evaluates predictions.

Math uses installed Math-Verify. Source-provided code tests run exclusively in
an isolated Docker container, with no host mounts/network/GPU access.
"""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import uuid
from importlib.metadata import version
from .protocol import digest


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```",text,flags=re.S)
    return blocks[-1].strip() if blocks else re.sub(r"^FINAL:\s*","",text.strip(),flags=re.I)


# The child is disposable; it never executes inside the host Python process.
WORKER = r'''
import contextlib,io,json,os,sys,tempfile,traceback,types,unittest
payload=json.load(sys.stdin)
solution,tests=payload['solution'],payload['test']
if not isinstance(tests,str) or not tests.strip():
    raise ValueError('Source test must be a nonempty Python string')
module=types.ModuleType('candidate')
module.__dict__['__file__']='/tmp/candidate.py'
sys.modules['candidate']=module
captured=io.StringIO()
with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
    try:
        exec(compile(solution,'candidate.py','exec'),module.__dict__)
    except ModuleNotFoundError:
        raise  # Missing image dependencies are infrastructure errors, not failures.
    except BaseException:
        print(json.dumps({'score':0}),file=sys.__stdout__)
        sys.exit(0)
    try:
        exec(compile(tests,'source_tests.py','exec'),module.__dict__)
    except ModuleNotFoundError:
        raise
    except BaseException:
        print(json.dumps({'score':0}),file=sys.__stdout__)
        sys.exit(0)
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    if suite.countTestCases():
        result=unittest.TextTestRunner(stream=captured,verbosity=0).run(suite)
        score=int(result.wasSuccessful())
    else:
        import pytest
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'test_candidate.py')
            with open(path,'w') as f:
                f.write(solution+'\n'+tests+'\n')
            code=pytest.main(['-q','-p','no:cacheprovider',path])
            if code in (2,3,4,5):
                raise RuntimeError('Invalid/uncollected source tests: pytest exit '+str(code))
            score=int(code==0)
print(json.dumps({'score':score}))
'''


def grade(request, image, timeout=30, docker="docker"):
    if digest(request["prediction"]) != request["prediction_sha256"]:
        raise ValueError("Prediction hash mismatch")
    if request["metric"] == "external_math":
        # Bound the entire parser/verifier in a disposable subprocess. Math-Verify's
        # nested multiprocessing timeout is not spawn-compatible on Windows.
        proc = subprocess.run([sys.executable, "-m", __package__ + ".grader", "--math-worker"],
                              input=json.dumps(request), capture_output=True, text=True,
                              encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8",
                                                         PYTHONPATH=os.pathsep.join(sys.path)),
                              timeout=timeout, check=True)
        return json.loads(proc.stdout)
    if request["metric"] != "external_code":
        raise ValueError("Only code/math use this external scorer")
    tests = request["raw"].get("test")
    if not isinstance(tests,str) or not tests.strip():
        raise ValueError("Recover source tests as executable Python; do not infer tests from answers")
    image_id = subprocess.run([docker,"image","inspect",image,"--format","{{.Id}}"],capture_output=True,text=True,check=True).stdout.strip()
    if not image_id.startswith("sha256:"):
        raise ValueError("Cannot resolve immutable evaluator image ID")
    name = "kglmf-test-"+uuid.uuid4().hex
    command = [docker,"run","--rm","-i","--name",name,"--network","none","--read-only",
               "--cap-drop","ALL","--security-opt","no-new-privileges","--pids-limit","64",
               "--memory","1g","--cpus","1","--tmpfs","/tmp:rw,nosuid,size=128m",
               "--user","65534:65534",image_id,"python","-c",WORKER]
    payload = dict(solution=extract_code(request["prediction"]),test=tests)
    try:
        proc = subprocess.run(command,input=json.dumps(payload),capture_output=True,text=True,timeout=timeout,check=True)
        score = json.loads(proc.stdout)["score"]
    except subprocess.TimeoutExpired:
        score = 0
    finally:
        # Only the uniquely named child from this call is targeted.
        subprocess.run([docker,"rm","-f",name],capture_output=True,check=False)
    return dict(id=request["id"],prediction_sha256=request["prediction_sha256"],score=score,
                harness="KG-LMF source-test container ("+request["dataset"]+")",revision=image_id+":"+digest(WORKER))


def math_worker(request):
    """Called only by grade() inside its externally time-bounded subprocess."""
    from math_verify import parse, verify
    if digest(request["prediction"]) != request["prediction_sha256"]:
        raise ValueError("Prediction hash mismatch")
    gold = request["answers"][0]
    delimited_gold = gold if re.search(r"(?<!\\)\$.*(?<!\\)\$", gold, re.S) else "$" + gold + "$"
    gold_expr = parse(delimited_gold, parsing_timeout=None,
                      raise_on_error=True)
    if not gold_expr:
        raise ValueError("Unparseable source math answer")
    with contextlib.redirect_stdout(io.StringIO()):
        pred = parse(request["prediction"], parsing_timeout=None)
        score = int(bool(pred) and verify(gold_expr, pred, timeout_seconds=None))
    return dict(id=request["id"], prediction_sha256=request["prediction_sha256"], score=score,
                harness="huggingface/Math-Verify", revision=version("math-verify"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",default="kglmf-evaluator:1")
    parser.add_argument("--timeout",type=int,default=30)
    parser.add_argument("--docker",default="docker")
    parser.add_argument("--math-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    print(json.dumps(math_worker(request) if args.math_worker else grade(request,args.image,args.timeout,args.docker)))


if __name__ == "__main__":
    main()
