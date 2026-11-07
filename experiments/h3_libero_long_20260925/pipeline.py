"""Fixed training/evaluation sequence with independent progress for each child."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import atomic_json, digest, progress, shared_root, source_hashes


def group(root, out, stage, specs):
    children=[]
    try:
        for name,command,cursor_path,result_path,expected in specs:
            log=(out/f'{name}.log').open('x')
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','TOKENIZERS_PARALLELISM':'false'})
            children.append(dict(name=name,child=child,log=log,cursor_path=cursor_path,
                result_path=result_path,expected=expected,last=None,advanced=time.time(),done=False))
        atomic_json(out/f'{stage}_identity.json',dict(started_at=time.time(),
            jobs=[dict(name=x['name'],pid=x['child'].pid,command=x['child'].args) for x in children]))
        while True:
            now=time.time(); observations=[]
            for item in children:
                current=json.loads(item['cursor_path'].read_text()) if item['cursor_path'].exists() else {}
                cursor=(current.get('phase'),current.get('cursor'))
                if cursor!=item['last'] and cursor!=(None,None):
                    item['last'],item['advanced']=cursor,now
                code=item['child'].poll()
                if code is not None and not item['done']:
                    if code!=0: raise RuntimeError(f"{item['name']} exited {code}; see its preserved log")
                    result=json.loads(item['result_path'].read_text())
                    if result['state']!=item['expected']: raise ValueError('invalid child completion state')
                    item['done']=True
                if not item['done'] and now-item['advanced']>900:
                    raise RuntimeError(f"{item['name']} primary cursor stale")
                observations.append(dict(name=item['name'],pid=item['child'].pid,done=item['done'],
                    progress=current,last_advance=item['advanced'],exit_code=code))
            atomic_json(out/f'{stage}_status.json',dict(observed_at=now,jobs=observations))
            # Cursor changes only when a real child changes, never for a heartbeat alone.
            progress(out,stage,[dict(name=x['name'],cursor=x['last'],done=x['done']) for x in children])
            if all(x['done'] for x in children): break
            time.sleep(15)
    finally:
        for item in children:
            if item['child'].poll() is None:
                item['child'].terminate()
                item['child'].wait(timeout=30)
            item['log'].close()


def aggregate(root, names, architecture, milestone, states):
    rows=[]; contracts=[]
    for name in names:
        path=root/'evaluations'/name
        result=json.loads((path/'result.json').read_text())
        if result['records_sha256']!=digest(path/'episodes.jsonl'): raise ValueError('episode integrity mismatch')
        contracts.append(json.loads((path/'contract.json').read_text()))
        rows.extend(json.loads(s) for s in (path/'episodes.jsonl').read_text().splitlines())
    expected={(t,s) for t in range(10) for s in range(states)}
    if len(rows)!=len(expected) or {(r['task'],r['initial_state']) for r in rows}!=expected:
        raise ValueError('paired evaluation coverage mismatch')
    if len({c['checkpoint_sha256'] for c in contracts})!=1: raise ValueError('mixed checkpoint')
    return dict(architecture=architecture,update=milestone,count=len(rows),
        successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
        task_successes={str(t):sum(r['success'] for r in rows if r['task']==t) for t in range(10)},
        policy_seconds=sum(r['policy_seconds'] for r in rows),encoding_seconds=sum(r['encoding_seconds'] for r in rows),
        checkpoint_sha256=contracts[0]['checkpoint_sha256'],evaluation_shards=names,
        latency_note='four simultaneous evaluation workers; throughput timings are not standalone latency')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True); a=p.parse_args(); root=shared_root(a.root)
    if Path(a.name).name!=a.name: raise ValueError('invalid pipeline identity')
    out=root/a.name; out.mkdir(exist_ok=False)
    source=Path(__file__).parent
    gate=json.loads((root/'preflight_v1/result.json').read_text())
    if gate['state']!='gpu_training_gate_passed': raise ValueError('GPU gate required')
    if gate['source_sha256']!=source_hashes(source): raise ValueError('source differs from tested gate')
    for arch in ('flat','hierarchical'):
        smoke=json.loads((root/'evaluations'/f'smoke_{arch}_v1'/'result.json').read_text())
        if smoke['state']!='smoke_complete' or smoke['count']!=1: raise ValueError('simulator gate required')
    atomic_json(out/'contract.json',dict(seed=42,updates=[5000,15000,30000],development_states=5,
        final_states=50,max_steps=520,source_sha256=source_hashes(source),preflight_sha256=digest(root/'preflight_v1/result.json')))
    previous={}; results=[]
    for milestone in (5000,15000,30000):
        for architecture in ('hierarchical','flat'):
            name=f'{a.name}_{architecture}_{milestone}'
            run=root/'runs'/name
            command=[sys.executable,str(source/'train.py'),'--root',str(root),
                '--architecture',architecture,'--name',name,'--until',str(milestone)]
            if architecture in previous: command+=['--resume',str(previous[architecture])]
            group(root,out,f'train_{architecture}_{milestone}',[(name,command,run/'progress.json',run/'result.json',
                'training_complete' if milestone==30000 else 'training_segment_complete')])
            previous[architecture]=run/'latest.pt'
            checkpoint=run/f'ema_{milestone:06d}.pt'
            states=50 if milestone==30000 else 5
            specs=[]; names=[]
            for shard in range(4):
                ev=f'{name}_eval{shard}'; names.append(ev)
                command=[sys.executable,str(source/'evaluate.py'),'--root',str(root),'--checkpoint',str(checkpoint),
                    '--output',ev,'--states',str(states),'--task-indices',*[str(t) for t in range(shard,10,4)]]
                path=root/'evaluations'/ev
                specs.append((ev,command,path/'progress.json',path/'result.json','evaluation_complete'))
            group(root,out,f'evaluate_{architecture}_{milestone}',specs)
            results.append(aggregate(root,names,architecture,milestone,states))
            atomic_json(out/'results.json',dict(state='partial',results=results))
    atomic_json(out/'result.json',dict(state='execution_complete',results=results,
        scientific_claim='success rates require interpretation; not an automatic superiority verdict'))
    progress(out,'complete',len(results))


if __name__=='__main__': main()
