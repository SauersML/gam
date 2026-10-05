"""Plot measured CO2, a real gamfit fit, and its uncertainty at two scales.

Fit the frozen NOAA/Scripps monthly measurements from 1980–2024; display
2015–2024. The lower panel subtracts the fitted mean, so the 95% credible
band and 95% observation interval are readable without inflating them.
All interval boundaries come directly from Model.predict; no decorative bands.

Data: https://gml.noaa.gov/ccgg/trends/data.html
Credit: Xin Lan, NOAA/GML, and Ralph Keeling, Scripps Institution of Oceanography.
Run: python -m scripts.gen_readme_co2
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gamfit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/data/co2_mm_mlo.txt'
OUTPUT = ROOT / 'docs/images'
FORMULA = 'co2 ~ s(time, k=100) + s(phase, periodic=true, period=1, k=12)'
THEMES = {
    'light': dict(background='#ffffff', ink='#465468', grid='#e4eaf1',
                  data='#24374e', fit='#2361db', band='#548bdc'),
    'dark': dict(background='#0d1117', ink='#b6c3d4', grid='#263142',
                 data='#e1eaf4', fit='#73b5ff', band='#548bdc'),
}


def fit():
    data = pd.read_csv(SOURCE, sep=r'\s+', comment='#', header=None,
                       names=['year', 'month', 'time', 'co2', 'noaa_trend', 'days', 'sd', 'se'])
    data = data[(data.year >= 1980) & (data.year <= 2024) & (data.co2 > 0)].copy()
    data['phase'] = (data.month - 0.5) / 12
    model = gamfit.fit(data, FORMULA)
    displayed = data[data.year >= 2015].copy()
    time = np.linspace(displayed.time.min(), displayed.time.max(), 1800)
    grid = {'time': time, 'phase': time % 1}
    bands = model.predict(grid, interval=0.95, observation_interval=True, return_type='dict')
    at_data = model.predict(displayed, interval=0.95, observation_interval=True, return_type='dict')
    assert len(data) == 540 and len(displayed) == 120
    for predictions in (bands, at_data):
        for field in ('posterior_mean', 'posterior_mean_lower', 'posterior_mean_upper',
                      'observation_lower', 'observation_upper'):
            assert np.isfinite(predictions[field]).all()
        assert (predictions['observation_lower'] <= predictions['posterior_mean_lower']).all()
        assert (predictions['observation_upper'] >= predictions['posterior_mean_upper']).all()
        assert (predictions['posterior_mean_lower'] <= predictions['posterior_mean']).all()
        assert (predictions['posterior_mean_upper'] >= predictions['posterior_mean']).all()
    residual = displayed.co2.to_numpy() - at_data['posterior_mean']
    print(f'540 training measurements, 120 displayed; real 95% mean and observation intervals; '
          f'displayed residual RMS {np.sqrt(np.mean(residual ** 2)):.3f} ppm')
    return displayed, time, bands, residual


def uncertainty(ax, time, bands, theme, center=0):
    ax.fill_between(time, bands['observation_lower'] - center,
                    bands['observation_upper'] - center,
                    color=theme['band'], alpha=0.18, linewidth=0, zorder=1)
    ax.fill_between(time, bands['posterior_mean_lower'] - center,
                    bands['posterior_mean_upper'] - center,
                    color=theme['band'], alpha=0.48, linewidth=0, zorder=2)


def draw(data, time, bands, residual, theme, path):
    plt.rcParams.update({'font.family': ['Helvetica Neue', 'DejaVu Sans'],
                         'font.size': 23, 'savefig.bbox': None})
    fig = plt.figure(figsize=(16, 10.5), dpi=240, facecolor=theme['background'])
    main = fig.add_axes((0.11, 0.405, 0.865, 0.565), facecolor=theme['background'])
    errors = fig.add_axes((0.11, 0.115, 0.865, 0.225), facecolor=theme['background'])
    mean = bands['posterior_mean']
    uncertainty(main, time, bands, theme)
    main.plot(time, mean, color=theme['fit'], linewidth=5,
              solid_capstyle='round', zorder=3)
    main.scatter(data.time, data.co2, s=105, color=theme['data'],
                 edgecolors=theme['background'], linewidth=0.8, zorder=4)
    main.set(ylim=(395, 430), ylabel='CO₂ (ppm)')
    main.set_yticks([400, 410, 420, 430])
    main.tick_params(labelbottom=False)

    # Exactly the same intervals, translated by the fitted mean, in ppm.
    uncertainty(errors, time, bands, theme, center=mean)
    for field in ('observation_lower', 'observation_upper'):
        errors.plot(time, bands[field] - mean, color=theme['band'],
                    linewidth=2, alpha=0.8, zorder=2)
    errors.axhline(0, color=theme['fit'], linewidth=3, zorder=3)
    errors.scatter(data.time, residual, s=90, color=theme['data'],
                   edgecolors=theme['background'], linewidth=0.8, zorder=4)
    limit = max(0.8, float(np.max(np.abs(residual))) * 1.12,
                float(np.max(bands['observation_upper'] - mean)) * 1.12)
    errors.set(ylim=(-limit, limit), xlabel='Year', ylabel='Residual (ppm)')
    errors.set_yticks([-0.5, 0, 0.5])
    for ax in (main, errors):
        ax.set_xlim(2014.9, 2025.05)
        ax.set_xticks([2015, 2017, 2019, 2021, 2023, 2025])
        ax.grid(axis='y', color=theme['grid'], linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=14, labelsize=22, colors=theme['ink'])
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_color(theme['ink'])
            label.set_fontsize(25)
        ax.xaxis.labelpad = 18
        ax.yaxis.labelpad = 20
        for spine in ax.spines.values():
            spine.set_visible(False)
        assert not ax.get_title() and not ax.texts and ax.get_legend() is None
    assert not fig.texts
    fig.savefig(path, dpi=240, facecolor=theme['background'])
    plt.close(fig)


def main():
    data, time, bands, residual = fit()
    for name, theme in THEMES.items():
        suffix = '_dark' if name == 'dark' else ''
        draw(data, time, bands, residual, theme, OUTPUT / f'readme_co2{suffix}.png')


if __name__ == '__main__':
    main()
