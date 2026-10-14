"""Generate the assignment report from measured benchmark JSON; never invent timings."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor('#102437')
TEAL = colors.HexColor('#167963')
MINT = colors.HexColor('#60caa7')
CORAL = colors.HexColor('#eb8d76')
MUTED = colors.HexColor('#536579')
LIGHT = colors.HexColor('#edf4f5')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSON produced by python -m life.benchmark (legacy optimized CPU baseline)')
    parser.add_argument('--student', default='Name and student ID: to be completed')
    parser.add_argument('--partner', default='Individual / second member: to be completed')
    args=parser.parse_args()
    font_dir=Path(reportlab.__file__).parent/'fonts'
    pdfmetrics.registerFont(TTFont('Lab',str(font_dir/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('Lab-Bold',str(font_dir/'VeraBd.ttf')))
    data=json.loads(Path(args.input).read_text())
    rows=data['results'];env=data['environment'];first=rows[0]['config']
    complete=all(r['cpu'] and r['cuda'] and r['verified'] is True for r in rows)
    all_cpu=all(r['cpu'] for r in rows)
    all_gpu=all(r['cuda'] for r in rows)
    status='Measured CPU/CUDA comparison' if complete else 'Partial experiment - hardware comparison incomplete'
    out=ROOT/'output'/'pdf';out.mkdir(parents=True,exist_ok=True)
    docs=ROOT/'docs';docs.mkdir(exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyLab',fontName='Lab',fontSize=10,leading=15,textColor=NAVY,spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallLab',fontName='Lab',fontSize=8,leading=11,textColor=MUTED,spaceAfter=6))
    styles.add(ParagraphStyle(name='TitleLab',fontName='Lab-Bold',fontSize=35,leading=40,textColor=NAVY,spaceAfter=16))
    styles.add(ParagraphStyle(name='HeadingLab',fontName='Lab-Bold',fontSize=21,leading=27,textColor=NAVY,spaceAfter=15))
    styles.add(ParagraphStyle(name='SubLab',fontName='Lab-Bold',fontSize=12,leading=17,textColor=TEAL,spaceBefore=10,spaceAfter=8))
    styles.add(ParagraphStyle(name='KickerLab',fontName='Lab-Bold',fontSize=9,leading=14,textColor=TEAL,spaceAfter=14))
    story=[];markdown=['# Life Lab: Conway\'s Game of Life - HPC report',args.student,args.partner,'**'+status+'**']
    def p(text,style='BodyLab'):
        return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
    def para(text,style='BodyLab'):
        story.append(p(text,style));markdown.append(text)
    def heading(n,title):
        story.append(p(f'{n:02d} / HIGH PERFORMANCE COMPUTING','KickerLab'))
        story.append(p(title,'HeadingLab'));markdown.append('## '+title)
    def sub(title):
        story.append(p(title,'SubLab'));markdown.append('### '+title)
    def table(headers,values,widths):
        wrapped=[[p(x,'SmallLab') for x in headers]]+[[p(x,'SmallLab') for x in row] for row in values]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
          ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),6),
          ('LEFTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),1,MINT),
          ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5e0e6'))]))
        story.append(t);story.append(Spacer(1,12))
        markdown.append('| '+' | '.join(headers)+' |')
        markdown.append('| '+' | '.join('---' for _ in headers)+' |')
        markdown.extend('| '+' | '.join(str(x) for x in row)+' |' for row in values)
    def newpage():story.append(PageBreak())

    para('LIFE LAB / EXPERIMENT REPORT','KickerLab')
    story.append(Spacer(1,22))
    para("Small rules.\nParallel worlds.",'TitleLab')
    para("Conway's Game of Life using 2D CPU arrays and NVIDIA CUDA",'HeadingLab')
    para(args.student);para(args.partner)
    para('Evidence recorded: '+data['created_at'][:10]+' | '+status,'SmallLab')
    story.append(Spacer(1,15))
    sub('Abstract')
    para('This project implements the same cellular automaton with a vectorized NumPy CPU engine and a CUDA kernel. A local dashboard supports pattern exploration, while a separate measurement path excludes rendering. The experiment varies grid size, uses reproducible input, and records compute and end-to-end times for repeated runs. Full final grids are compared whenever both engines execute.')
    if complete:
        para('The supplied evidence contains real CPU and CUDA measurements with matching final grids for every configuration. The following pages report these measurements and their limits.')
    else:
        para('The supplied evidence does not contain a complete, verified CPU/CUDA hardware comparison. GPU access was unavailable to the measurement process; this can also occur when a sandbox hides an installed GPU. CUDA simulator tests check kernel logic only. Hardware timing and equality checks are required before claiming a GPU speedup from this evidence.')
    sub('Problem and scope')
    para('Life applies the B3/S23 rule to eight neighbours: a cell is alive next generation if it has three neighbours, or if it was already alive and has two. Updates are simultaneous [1]. This implementation uses finite grids with explicitly selected wrap-around or dead-outside boundaries.')
    table(['Deliverable','Implementation'],[
      ['Interactive experiment','Original navy, mint and coral dashboard; Python simulation'],
      ['Two computing methods','NumPy 2D arrays and a Numba-CUDA device kernel'],
      ['Evidence and presentation','JSON/CSV measurements, this report, and a 2:45 recording script']], [120,387])
    para('Final submission still requires student details and the student\'s own presentation recording.' + ('' if complete else ' A verified real-GPU experiment is also required.'),'SmallLab')

    newpage();heading(2,'Implementation and optimization')
    sub('CPU: reusable two-dimensional arrays')
    para('Cells are stored in C-contiguous uint8 arrays. Each generation copies the current state into the interior of a reusable halo array. Its border is filled from opposite edges for wrap-around, or remains zero for dead-outside. Eight neighbour slices are added into a preallocated uint8 count array. Boolean masks implement birth and survival, writing a separate next array before swapping references.')
    para('Vectorized NumPy operations avoid per-cell Python interpretation. Work buffers and masks are allocated once per run, reducing repeated allocation. Neighbour counts cannot exceed eight, so uint8 is sufficient. A separate scalar nested-loop implementation is retained as a correctness oracle rather than an artificially slow benchmark baseline.')
    sub('CUDA: independent cell updates')
    para('The kernel maps x to columns and y to rows using cuda.grid(2). Each in-range thread counts eight neighbours, reads its current cell and writes one result. A 32x8 block has 256 threads; the launch grid rounds up both dimensions. Bounds checks protect partial blocks. Adjacent x threads access consecutive columns in C-order storage [2].')
    para('Two device buffers persist across all generations. Host-to-device transfer occurs before the loop and device-to-host transfer after it. Ordered launches on the default stream preserve generation dependencies without a host synchronization after every step. A final synchronization is required before accepting timing/output. Current and next buffers prevent read/write races.')
    table(['Property','CPU 2D arrays','CUDA'],[
      ['Execution','Vectorized array operations on CPU','One cell per GPU thread'],
      ['Time complexity','O(H x W) work per generation','O(H x W) total work; parallel execution'],
      ['Memory','O(H x W), halo/count/masks plus two grids','O(H x W), two device grids plus host data'],
      ['Data movement','Host memory only','One input upload and one output download'],
      ['Dependency','Python + NumPy','NVIDIA GPU, driver and CUDA Python libraries'],
      ['Tradeoff','Low launch overhead; useful for small grids','Parallel work can amortize launch/transfer costs']], [87,210,210])
    sub('Limits of optimization')
    para('Shared-memory tiling, bit packing, temporal blocking and sparse algorithms are not implemented. Repeated global neighbour reads and modulo operations remain possible bottlenecks. The chosen block is a starting point, not a proven optimum; compare supported block shapes on the target GPU.')

    newpage();heading(3,'Experimental method')
    table(['Parameter','Recorded configuration'],[
      ['Grid sizes',', '.join(f'{r["config"]["size"]} x {r["config"]["size"]}' for r in rows)],
      ['Generations / repetitions',f'{first["iterations"]} / {first["repeats"]} per size'],
      ['Input',f'{first["pattern"]}; density {first["density"]}; seed {first["seed"]}'],
      ['Boundary / CUDA block',f'{first["boundary"]} / {first["block_x"]} x {first["block_y"]}'],
      ['CPU',env['cpu']],['OS / Python',env['os']+' / '+env['python']],
      ['Libraries',', '.join(f'{k} {v or "not installed"}' for k,v in env['packages'].items())],
      ['NVIDIA device / driver',env['nvidia_smi']],
    ],[125,382])
    sub('Reproducible procedure')
    for text in [
      '1. Generate the initial grid once for each configuration. Each repetition starts from an identical copy; initial and final SHA-256 hashes are saved.',
      '2. Warm up each engine outside measurement. This removes initial JIT compilation and context setup from the CUDA interval.',
      '3. Repeat the requested generations without drawing or logging. Alternate CPU/GPU measurement order on successive repetitions.',
      '4. Compare complete output arrays after every repetition in both mode. If any pair differs, suppress speedups and flag the experiment.',
      '5. Report median, minimum and maximum values plus all raw samples. The observed range describes run-to-run variability, not a statistical confidence interval.']:
        para(text)
    sub('Timing definitions')
    para('CPU compute uses perf_counter around the generation loop after allocation. CUDA compute uses events around the launch sequence; it includes device idle gaps if host launch overhead leaves the GPU waiting. It is not a sum of isolated kernel active durations. Event synchronization ensures queued work has completed [3].')
    para('CPU total includes input validation/copy, buffer allocation, simulation and final output copy. CUDA total includes validation, allocation, H2D, simulation, synchronization and D2H. Event creation and cold startup/JIT are excluded. Both totals include Python control overhead. Speedup = median CPU time / median CUDA time for the same timing definition.')

    newpage();heading(4,'Measured performance')
    para(status,'KickerLab')
    f=lambda x:'Not measured' if x is None else f'{x:.3f}'
    values=[]
    for r in rows:
        values.append([str(r['config']['size'])+' x '+str(r['config']['size']),
          f(r['cpu']['compute_ms'] if r['cpu'] else None),f(r['cpu']['total_ms'] if r['cpu'] else None),
          f(r['cuda']['compute_ms'] if r['cuda'] else None),f(r['cuda']['total_ms'] if r['cuda'] else None),
          'Pending' if r['speedup_total'] is None else f'{r["speedup_total"]:.2f}x'])
    table(['Grid','CPU compute\nms','CPU total\nms','CUDA compute\nms','CUDA total\nms','Total\nspeedup'],values,[85,83,83,88,88,80])
    chart=Drawing(507,220)
    bars=HorizontalBarChart();bars.x=70;bars.y=32;bars.width=408;bars.height=155
    series=[]
    if all_cpu:series.append([r['cpu']['compute_ms'] for r in rows])
    if all_gpu:series.append([r['cuda']['compute_ms'] for r in rows])
    if series:
        bars.data=series;bars.categoryAxis.categoryNames=[str(r['config']['size'])+' x '+str(r['config']['size']) for r in rows]
        bars.categoryAxis.labels.fontSize=8;bars.valueAxis.labels.fontSize=8;bars.valueAxis.valueMin=0
        bars.categoryAxis.labels.fontName='Lab';bars.valueAxis.labels.fontName='Lab'
        bars.valueAxis.labelTextFormat='%.1f'
        bars.valueAxis.valueMax=max(max(s) for s in series)*1.15 or 1
        bars.valueAxis.valueStep=bars.valueAxis.valueMax/4
        bars.bars[0].fillColor=MINT if all_cpu else CORAL
        if len(series)>1:bars.bars[1].fillColor=CORAL
        bars.bars.strokeColor=None;bars.barSpacing=3;bars.groupSpacing=12
        chart.add(bars)
        chart.add(String(70,204,'Median compute time (ms)',fontName='Lab-Bold',fontSize=10,fillColor=NAVY))
        chart.add(String(70,9,'Mint: CPU'+('   |   Coral: CUDA' if all_gpu else '   |   CUDA: not measured'),fontName='Lab',fontSize=8,fillColor=MUTED))
        story.append(chart);renderSVG.drawToFile(chart,str(docs/'performance.svg'))
        markdown.append('![Measured compute times](performance.svg)')
    sub('Observed variability')
    ranges=[]
    for r in rows:
        for backend in ('cpu','cuda'):
            if r[backend]:
                s=r[backend];ranges.append(f'{r["config"]["size"]} {backend.upper()}: {s["compute_ms_min"]:.3f}-{s["compute_ms_max"]:.3f} ms')
    para('; '.join(ranges)+'.','SmallLab')
    sub('Interpretation')
    if complete:
        para('End-to-end CPU/CUDA ratios: '+ '; '.join(f'{r["config"]["size"]} x {r["config"]["size"]}: {r["speedup_total"]:.2f}x' for r in rows)+'. A ratio above one means CUDA is faster; below one means CPU is faster.')
        para('The small grid exposes fixed launch and transfer overhead, while larger grids provide more independent cells to distribute across GPU threads. Compute and total columns separate these effects. These observations apply to this implementation and recorded machine; they are not universal device rankings.')
    else:
        if all_cpu and len(rows)>1:
            ratio=rows[-1]['cpu']['compute_ms']/rows[0]['cpu']['compute_ms']
            para(f'Across the first and last recorded sizes, median CPU compute time changes by {ratio:.2f}x. Grid growth increases the number of cell updates; cache behaviour and fixed overhead prevent assuming an exact proportional timing relationship.')
        para('No GPU crossover point, acceleration factor or CPU/GPU ranking can be inferred from the current evidence. The technical reasons for possible GPU acceleration are hypotheses until measured on real hardware. Empty CUDA values are deliberate, not zero-duration executions.')

    newpage();heading(5,'Validation, limitations and conclusion')
    sub('Correctness evidence')
    para('The test suite checks birth/survival for all neighbour counts, stable blocks, period-two blinkers, four-generation glider translation, input immutability, invalid inputs and deterministic seeds. CPU results are checked against scalar loops on narrow and rectangular grids with both boundaries. Export structure and mismatch handling are tested separately.')
    para('The CUDA correctness tests cover wrap/dead boundaries, rectangular and partial blocks, supported block shapes and buffer swaps. A simulator executes kernel logic on the CPU and cannot validate device code generation, real concurrency or GPU performance. Record the actual test outcome on the machine being studied.')
    if complete:
        para('The supplied benchmark records full CPU/CUDA agreement for every reported grid size and repetition. Hardware and driver details are taken from that run. Benchmark equality does not establish that the separate regression suite was executed.','SmallLab')
    sub('Threats to validity')
    para('Wall times depend on CPU/GPU model, driver, package versions, clock/thermal state and competing workloads. This experiment measures one dense vectorized CPU implementation against one basic CUDA implementation. It is not a comparison against an optimized multicore stencil, shared-memory CUDA kernel, sparse engine or Hashlife. Finite boundaries also differ from the infinite mathematical game.')
    sub('Conclusion and next step')
    para('The project demonstrates synchronous 2D-array evolution and a CUDA parallel decomposition with device-resident double buffering. It separates visualization from measurement and records reproducible evidence. '+('The recorded hardware comparison is verified; interpret performance only for these configurations and hardware.' if complete else 'The CPU experiment is complete locally. A real-GPU run is still required to finish the assignment\'s empirical comparison.'))
    sub('Reproduce and complete')
    para('Run python -m life.benchmark --sizes 128 256 512 1024 --iterations 100 --repeats 5 --mode both --output results/gpu on the NVIDIA machine. Confirm verified=true for each size, then regenerate this report with tools/build_report.py and your student details. Record the provided presentation script within three minutes. See README.md for installation and exact commands.','SmallLab')
    sub('References and provenance')
    for text in [
      '[1] Conway\'s Game of Life. Wikipedia. https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life',
      '[2] NVIDIA. Writing CUDA Kernels. https://nvidia.github.io/numba-cuda/user/kernels.html',
      '[3] NVIDIA. CUDA Host API (events and synchronization). https://nvidia.github.io/numba-cuda/reference/host.html',
      '[4] NVIDIA. Installation. https://nvidia.github.io/numba-cuda/user/installation.html',
      '[5] User-supplied feature reference: amantha2002, Conway-s-Game-of-Life-Python-and-cuda-, GitHub. Source and UI in Life Lab were independently written; no source files were copied.',
      'Sources consulted 14 October 2026. Numerical evidence: '+str(Path(args.input))+'. Raw repetition values and environment details are retained in this JSON and the accompanying CSV.'
    ]:para(text,'SmallLab')

    def decorate(canvas,doc):
        canvas.saveState();w,h=A4
        canvas.setFillColor(TEAL);canvas.rect(44,h-26,42,3,fill=1,stroke=0)
        canvas.setStrokeColor(colors.HexColor('#d7e2e6'));canvas.line(44,42,w-44,42)
        canvas.setFillColor(MUTED);canvas.setFont('Lab',8)
        canvas.drawString(44,27,'LIFE LAB  /  HPC ASSIGNMENT')
        canvas.drawRightString(w-44,27,f'{doc.page:02d}')
        canvas.restoreState()
    pdf=out/'life-lab-report.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=48,bottomMargin=55,
                         title='Life Lab - CPU and CUDA HPC Report',author=args.student)
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    (docs/'report.md').write_text('\n\n'.join(markdown)+'\n')
    print(f'Created {pdf} and {docs / "report.md"}')

if __name__=='__main__':main()
