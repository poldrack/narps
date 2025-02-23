## Code metrics generated using: radon raw *.py -O raw_metrics.json -j

import json
import pandas as pd
import narps
from narps.narps import Narps
import sys
import os
sys.path.append('src/narps')
import inspect
from radon.raw import analyze


class_members = inspect.getmembers(Narps)


with open('src/narps/raw_metrics.json') as f:
    raw_metrics = json.load(f)

df = pd.DataFrame.from_dict(raw_metrics).T

print('total lines of code:', df['loc'].sum())

print(inspect_hierarchy(Narps))

rawcode = inspect.getsource(Narps).split('\n')

# split into chunks by def statements

chunks = {}
methodname = None
for i, line in enumerate(rawcode):
    if len(line) == 0:
        continue
    if line.lstrip().startswith('def '):
        methodname = line.split('(')[0].split('def ')[1]
        chunks[methodname] = [line]
        continue
    # skip lines before first method
    if methodname is not None:
        chunks[methodname].append(line)
    
if not os.path.exists('src/narps/code_chunks'):
    os.mkdir('src/narps/code_chunks')

metrics = {}

for k, v in chunks.items():
    fname = f'src/narps/code_chunks/{k}.py'
    with open(fname, 'w') as f:
        f.write('\n'.join(v))
    
    cmd = f'radon raw {fname} -O {fname.replace(".py", "_metrics.json")} -j'
    run_shell_cmd(cmd)
    with open(fname.replace('.py', '_metrics.json')) as f:
        metrics.update(json.load(f))

metrics_df = pd.DataFrame(metrics).T
print('total lines of code:', metrics_df['loc'].sum())
print(f'range of lines of code: {metrics_df["loc"].min()} to {metrics_df["loc"].max()}')