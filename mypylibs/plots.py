"""Report figures: trajectory + dense cloud for each run, and sensitivity bars."""
import os
from os import path as opath

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _cloud(ax, xyz, rgb, i, j, n=40000):
    if len(xyz) == 0:
        return
    sel = np.random.default_rng(0).permutation(len(xyz))[:n]
    ax.scatter(xyz[sel, i], xyz[sel, j], s=0.15, c=rgb[sel] / 255.0, linewidths=0)
    ax.set_aspect('equal')
    ax.set_xticks([]), ax.set_yticks([])


def plot_overview(results, rows, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    # Figure 1: main runs (f x1.0, undistorted): orbit top view, dense side/top.
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for r, v in enumerate(['a', 'b']):
        xyz, rgb, C, sparse = results[f'{v}_10u']
        ax = axes[r, 0]
        sp = sparse[np.linalg.norm(sparse, axis=1) < 2.5]
        ax.scatter(sp[:, 0], sp[:, 2], s=0.1, c='0.6', linewidths=0)
        ax.plot(C[:, 0], C[:, 2], 'r.-', ms=2, lw=0.8)
        ax.plot(C[0, 0], C[0, 2], 'go', ms=4)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f'video {v}')
        _cloud(axes[r, 1], xyz, rgb, 0, 1)
        _cloud(axes[r, 2], xyz, rgb, 0, 2)
    axes[0, 0].set_title('cameras + sparse (top)', fontsize=8)
    axes[0, 1].set_title('dense cloud (side)', fontsize=8)
    axes[0, 2].set_title('dense cloud (top)', fontsize=8)
    fig.tight_layout()
    fig.savefig(opath.join(fig_dir, 'overview.png'), dpi=200)
    plt.close(fig)

    # Figure 2: top views of the camera orbit for the six runs of each video.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    styles = {0.8: ':', 1.0: '-', 1.2: '--'}
    for ax, v in zip(axes, ['a', 'b']):
        for und, col in [(True, 'tab:blue'), (False, 'tab:orange')]:
            for fac in [0.8, 1.0, 1.2]:
                name = f'{v}_{round(fac * 10):02d}{"u" if und else "d"}'
                C = results[name][2]
                ax.plot(C[:, 0], C[:, 2], styles[fac], color=col, lw=0.9,
                        label=f'f x{fac} {"u" if und else "d"}')
        ax.set_aspect('equal'); ax.set_title(f'video {v}: camera orbit (top view)', fontsize=8)
        ax.tick_params(labelsize=6)
    axes[1].legend(fontsize=5, loc='best')
    fig.tight_layout()
    fig.savefig(opath.join(fig_dir, 'orbits.png'), dpi=200)
    plt.close(fig)


def write_table(rows, fig_dir):
    """LaTeX table of the twelve runs, input by report.tex."""
    EOL = 2 * chr(92)  # LaTeX row terminator

    def f(r, k, fmt):
        v = r.get(k)
        return '--' if v is None or v != v else fmt.format(v)
    lines = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\hline',
             'run & kf & 2-view & BA & swept & gap & sphere & dense ' + EOL,
             r' & reg. & px & px & deg & $/R$ & \% & pts ' + EOL, r'\hline']
    for i, r in enumerate(rows):
        if i == 6:
            lines.append(r'\hline')
        name = r['run'].replace('_', r'\_')
        lines.append(' & '.join([
            name, f"{r['registered']}/{r['keyframes']}",
            f(r, 'pair_reproj', '{:.2f}'),
            f(r, 'reproj_before_global_ba', '{:.2f}') + r'$\to$' + f(r, 'reproj_final', '{:.2f}'),
            f(r, 'swept_deg', '{:.0f}'), f(r, 'loop_gap', '{:.2f}'),
            f({'s': r.get('sphere_rms_rel', float('nan')) * 100}, 's', '{:.1f}'),
            f(r, 'dense_points', '{:d}')]) + ' ' + EOL)
    lines += [r'\hline', r'\end{tabular}']
    with open(opath.join(fig_dir, 'table.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
