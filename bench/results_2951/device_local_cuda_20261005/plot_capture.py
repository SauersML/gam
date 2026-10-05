"""Plot measured MATS local-target capture timings, not full-training throughput."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
report = json.loads((root / 'capture/results/local_capture_bench.json').read_text())
assert report['parity_passed'] and report['mode'] == 'cuda'
cases = report['cases']
labels = ['Clean model', 'Attention output weights × 1.5', 'Attention output weights × −0.5']
cpu = np.array([case['cpu_capture_median_seconds'] for case in cases])
cuda = np.array([case['prepared_capture_median_seconds'] for case in cases])
errors = []
for case in cases:
    for repeat in case['repeated_parity']:
        assert repeat['parity']['passed']
        if repeat['prepared_device']:
            errors.extend(part['max_absolute_error'] for part in repeat['parity']['inputs'])
            errors.append(repeat['parity']['targets']['max_absolute_error'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(11.8, 6.1))
fig.subplots_adjust(left=.30, right=.96, bottom=.24, top=.77)
y = np.arange(len(cases))
colors = ['#526D82', '#087F8C']
for shift, values, label, color, key in [(-.17, cpu, 'CPU reference', colors[0], 'cpu_capture_seconds'),
                                       (.17, cuda, 'Prepared CUDA', colors[1], 'prepared_capture_seconds')]:
    ax.barh(y+shift, values, height=.27, color=color, label=label)
    for row, (case, value) in enumerate(zip(cases, values)):
        samples = case[key]
        ax.scatter(samples, np.repeat(row+shift, len(samples)), color='white', edgecolors=color,
                   s=17, linewidth=.6, zorder=3)
        ax.text(value+.015, row+shift, f'{value*1000:.1f} ms', va='center', fontsize=10)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 1.03)
ax.set_xlabel('Capture time, seconds · median of five repetitions after warmup')
ax.xaxis.grid(True, alpha=.14)
ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2, frameon=False)
for row, ratio in enumerate(cpu/cuda):
    ax.text(.985, row, f'{ratio:.1f}×', ha='right', va='center', fontsize=15,
            color=colors[1], fontweight='bold')
fig.text(.04, .94, 'Native local targets: 21× faster on CUDA', fontsize=21, fontweight='bold')
fig.text(.04, .877, 'MATS · NVIDIA L40 · FP64 · 512 tokens · one MLP site in a four-layer language model', fontsize=11)
fig.text(.04, .12, f'All elementwise checks passed; largest absolute discrepancy {max(errors):.2g}.\n'
                  'Final input/target transfers included. Model import, compilation and warmup excluded.', fontsize=10)
fig.text(.04, .038, 'Measures supervision capture only—not full training, the new resident mixture, or mechanism discovery.  Slurm job 16419.',
         fontsize=9, color='#444444')
fig.savefig(root / 'cuda_local_capture.png', dpi=190)
fig.savefig(root / 'cuda_local_capture.pdf')
summary = {'job': 16419, 'source': 'edfb9aa49b5d4630445a1d54b79d24351d33c487',
           'speedup_by_case': dict(zip(labels, (cpu/cuda).tolist())),
           'max_absolute_error': max(errors), 'parity_passed': True,
           'scope': report['scope'], 'setup_excluded': True,
           'repetitions_after_warmup': report['repetitions_after_one_warmup']}
(root / 'FIGURE.json').write_text(json.dumps(summary, indent=2)+'\n')
