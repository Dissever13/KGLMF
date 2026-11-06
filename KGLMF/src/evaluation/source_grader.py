"""Source-test evaluator: bind the source's solution module inside its container."""
import sys,json
from KGLMF.src.evaluation import grader

# Bound diagnostic output without changing candidate code, tests, or time limits.
# Otherwise nonterminating print loops exhaust the container before timeout.
grader.WORKER=grader.WORKER.replace('captured=io.StringIO()', '''
class BoundedCapture(io.StringIO):
    def write(self, text):
        remaining=max(0,1048576-self.tell())
        if remaining:super().write(text[:remaining])
        return len(text)
captured=BoundedCapture()
''')
grader.WORKER=grader.WORKER.replace("payload=json.load(sys.stdin)", "os.chdir('/tmp')\npayload=json.load(sys.stdin)")
grader.WORKER=grader.WORKER.replace("sys.modules['candidate']=module", "sys.modules['candidate']=module\nsys.modules['solution']=module")
grader.WORKER=grader.WORKER.replace("module=types.ModuleType('candidate')", r'''
import ast,subprocess
try:
    cases=ast.literal_eval(tests)
except (ValueError,SyntaxError):
    cases=None
if isinstance(cases,dict):
    if set(cases)!=set(['stdin','stdout']) or not cases['stdin'] or len(cases['stdin'])!=len(cases['stdout']):
        raise ValueError('Unsupported source I/O test structure')
    score=1
    for inp,expected in zip(cases['stdin'],cases['stdout']):
        try:
            p=subprocess.run([sys.executable,'-c',solution],input=str(inp),text=True,capture_output=True,timeout=3)
            if p.returncode or p.stdout.split()!=str(expected).split():score=0;break
        except subprocess.TimeoutExpired:score=0;break
    print(json.dumps({'score':score}));sys.exit(0)
module=types.ModuleType('candidate')
''')
if __name__ == '__main__':
    grader.main()
