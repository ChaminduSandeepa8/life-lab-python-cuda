"""Additional measured evidence for the report's block-size and boundary sections."""
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from life.core import Config
from life.benchmark import suite, save_report
from life.gpu import cuda, cuda_status

root=Path(__file__).resolve().parents[1]
if not cuda_status()['available']:
    raise SystemExit('Real GPU access required. Run python app.py --check --require-cuda in your environment.')
dev=cuda.get_current_device()
facts={
    'cuda':cuda_status(),
    'multiprocessors':int(dev.MULTIPROCESSOR_COUNT),
    'warp_size':int(dev.WARP_SIZE),
    'max_threads_per_block':int(dev.MAX_THREADS_PER_BLOCK),
    'os_visible_ram_gib':(os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/1024**3
                          if hasattr(os, 'sysconf') else None),
}
dest=root/'results'/'report-study'
dest.mkdir(parents=True,exist_ok=True)
(dest/'hardware.json').write_text(json.dumps(facts,indent=2)+'\n')
blocks=[(8,8),(16,16),(32,8),(32,16)]
report=suite([Config(size=512,block_x=x,block_y=y) for x,y in blocks],mode='both',
             progress=lambda i,t,n:print(f'Block study {i+1}/{t}: {blocks[i]}',flush=True))
save_report(report,dest/'blocks')
dead=suite([Config(size=n,boundary='dead') for n in (128,256,512,1024)],mode='both',
           progress=lambda i,t,n:print(f'Dead-boundary study {i+1}/{t}: {n}',flush=True))
save_report(dead,dest/'dead')
assert all(r['verified'] for r in report['results']+dead['results'])
print('All supplementary comparisons matched.',flush=True)
