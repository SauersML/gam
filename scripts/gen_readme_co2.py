"""Draw a real additive decomposition of NOAA/Scripps monthly CO2.

The checked-in NOAA source is frozen for reproduction; only measured monthly
means from 1980–2024 are used (not NOAA's seasonally adjusted column). A single
gamfit model fits a time smooth plus a periodic year-fraction smooth. Their
sum reconstructs its predictions. No decorative contours or synthetic data.

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
    'light': dict(background='#ffffff', ink='#536173', grid='#e7ebf0',
                  data='#6d7b8e', fit='#007f88'),
    'dark': dict(background='#0d1117', ink='#a8b4c2', grid='#26303e',
                 data='#8797ac', fit='#70d5cf'),
}


def fit():
    data = pd.read_csv(SOURCE, sep=r'\s+', comment='#', header=None,
                       names=['year', 'month', 'time', 'co2', 'noaa_trend', 'days', 'sd', 'se'])
    data = data[(data.year >= 1980) & (data.year <= 2024) & (data.co2 > 0)].copy()
    data['phase'] = (data.month - 0.5) / 12
    model = gamfit.fit(data, FORMULA)

    def predict(time, phase):
        time, phase = np.broadcast_arrays(time, phase)
        return np.asarray(model.predict({'time': time.ravel(), 'phase': phase.ravel()},
                                        return_type='dict')['posterior_mean'])

    reference = 2000.0
    # Center the periodic term over a complete year, including the intercept
    # in the time component. Use the same offset for every prediction.
    offset = predict(reference, np.arange(1024) / 1024).mean()
    seasonal_zero = predict(reference, [0])[0] - offset
    seasonal = lambda phase: predict(reference, phase) - offset
    trend = lambda time: predict(time, 0) - seasonal_zero
    np.testing.assert_allclose(trend(data.time) + seasonal(data.phase),
                               predict(data.time, data.phase), atol=1e-8, rtol=0)
    np.testing.assert_allclose(seasonal([0]), seasonal([1]), atol=1e-8, rtol=0)
    assert len(data) == 540
    residual = data.co2.to_numpy() - predict(data.time, data.phase)
    assert np.isfinite(residual).all()
    print(f'{len(data)} real monthly measurements; reconstruction verified; '
          f'fit residual RMS: {np.sqrt(np.mean(residual ** 2)):.3f} ppm')
    return data, trend, seasonal


def draw(data, trend, seasonal, theme, path):
    plt.rcParams.update({'font.family': ['Helvetica Neue', 'DejaVu Sans'],
                         'font.size': 13, 'savefig.bbox': None})
    fig = plt.figure(figsize=(16, 8.5), dpi=240, facecolor=theme['background'])
    left = fig.add_axes((0.065, 0.13, 0.535, 0.81), facecolor=theme['background'])
    right = fig.add_axes((0.710, 0.13, 0.26, 0.81), facecolor=theme['background'])
    time = np.linspace(data.time.min(), data.time.max(), 1600)
    phase = np.linspace(0, 1, 500)
    # Height carries measured CO2, and lines carry the two additive terms.
    left.scatter(data.time, data.co2, s=9, color=theme['data'],
                 alpha=0.7, linewidth=0, zorder=2)
    left.plot(time, trend(time), color=theme['fit'], linewidth=2.5, zorder=3)
    left.set(xlim=(1979.5, 2025), ylim=(335, 430), xlabel='Year', ylabel='CO₂ (ppm)')
    left.set_xticks([1980, 1990, 2000, 2010, 2020])
    left.set_yticks([340, 360, 380, 400, 420])

    right.scatter(data.phase, data.co2.to_numpy() - trend(data.time), s=11,
                  color=theme['data'], alpha=0.45, linewidth=0, zorder=2)
    right.plot(phase, seasonal(phase), color=theme['fit'], linewidth=2.5, zorder=3)
    right.axhline(0, color=theme['grid'], linewidth=1, zorder=1)
    right.set(xlim=(0, 1), ylim=(-5, 5), xlabel='Year fraction', ylabel='Seasonal CO₂ (ppm)')
    right.set_xticks([0, 0.25, 0.5, 0.75, 1])
    right.set_yticks([-4, -2, 0, 2, 4])
    for ax in (left, right):
        ax.grid(axis='y', color=theme['grid'], linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=12, labelsize=12, colors=theme['ink'])
        ax.xaxis.label.set_color(theme['ink'])
        ax.yaxis.label.set_color(theme['ink'])
        ax.xaxis.labelpad = 18
        ax.yaxis.labelpad = 18
        for spine in ax.spines.values():
            spine.set_visible(False)
        assert not ax.get_title() and not ax.texts and ax.get_legend() is None
    assert not fig.texts
    fig.savefig(path, dpi=240, facecolor=theme['background'])
    plt.close(fig)


def main():
    data, trend, seasonal = fit()
    for name, theme in THEMES.items():
        suffix = '_dark' if name == 'dark' else ''
        draw(data, trend, seasonal, theme, OUTPUT / f'readme_co2{suffix}.png')


if __name__ == '__main__':
    main()
