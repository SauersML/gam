"""One circular plot of a real cyclic temperature fit, with a 95% mean band.

NOAA GHCN-Daily station USW00094728 (New York Central Park), 2024.
Monthly points average paired, quality-approved daily (TMAX + TMIN) / 2.
Fit the 12 monthly means with gamfit's periodic smooth; radius is temperature
in Celsius and angle is month. The fitted mean and credible bounds join at
the year boundary. The radial origin is -10 C, explicitly reflected by the
radial limits; neither temperatures nor uncertainty widths are rescaled.

Source: https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00094728.dly
Format: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
Run from the repo root: python -m scripts.gen_readme_temperature
"""
from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gamfit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/data/central_park_2024.dly'
FORMULA = 'temperature ~ s(phase, periodic=true, period=1, k=8)'
THEMES = {
    'light': dict(background='#ffffff', ink='#45566c', grid='#dce5ef',
                  mean='#1767c4', band='#489ad4', points='#182e47'),
    'dark': dict(background='#0d1117', ink='#b7c8dc', grid='#2a394d',
                 mean='#78c8ff', band='#489ad4', points='#f0f6fc'),
}


def monthly_data():
    daily = {}
    for line in SOURCE.read_text().splitlines():
        assert line[:11] == 'USW00094728' and line[11:15] == '2024'
        month = int(line[15:17])
        element = line[17:21]
        values = []
        for day in range(calendar.monthrange(2024, month)[1]):
            record = line[21 + day * 8:29 + day * 8]
            value, quality = int(record[:5]), record[6]
            values.append(value / 10 if value != -9999 and quality == ' ' else np.nan)
        daily[month, element] = np.asarray(values)
    means = []
    for month in range(1, 13):
        pairs = (daily[month, 'TMAX'] + daily[month, 'TMIN']) / 2
        assert np.isfinite(pairs).sum() >= 25
        means.append(np.nanmean(pairs))
    return {'phase': (np.arange(12) + 0.5) / 12, 'temperature': np.asarray(means)}


def fit():
    data = monthly_data()
    model = gamfit.fit(data, FORMULA)
    phase = np.linspace(0, 1, 1441)
    bands = model.predict({'phase': phase}, interval=0.95, return_type='dict')
    for field in ('posterior_mean', 'posterior_mean_lower', 'posterior_mean_upper'):
        assert np.isfinite(bands[field]).all()
        np.testing.assert_allclose(bands[field][0], bands[field][-1], atol=1e-10)
    # Verify the derivative also wraps, rather than just connecting endpoints.
    epsilon = 1e-5
    seam = model.predict({'phase': [-epsilon, 0, epsilon, 1-epsilon, 1, 1+epsilon]},
                         return_type='dict')['posterior_mean']
    np.testing.assert_allclose(seam[:3], seam[3:], atol=1e-9, rtol=0)
    print('12 measured monthly means; fitted mean, 95% credible bounds and seam derivatives verified.')
    return data, phase * 2 * np.pi, bands


def draw(data, theta, bands, theme, path):
    plt.rcParams.update({'font.family': ['Helvetica Neue', 'DejaVu Sans'],
                         'font.size': 24, 'savefig.bbox': None})
    fig = plt.figure(figsize=(12, 12), dpi=240, facecolor=theme['background'])
    ax = fig.add_axes((0.14, 0.12, 0.75, 0.75), projection='polar',
                     facecolor=theme['background'])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(-10, 32)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.set_yticklabels(['−10', '0', '10', '20', '30'], color=theme['ink'], fontsize=22)
    ax.set_rlabel_position(0)
    months = data['phase'] * 2 * np.pi
    ax.set_xticks(months)
    ax.set_xticklabels([str(month) for month in range(1, 13)],
                       fontsize=27, color=theme['ink'])
    ax.tick_params(axis='x', pad=20)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=theme['grid'], linewidth=1.2)
    ax.spines['polar'].set_visible(False)
    ax.fill_between(theta, bands['posterior_mean_lower'], bands['posterior_mean_upper'],
                    color=theme['band'], alpha=0.35, linewidth=0, zorder=2)
    ax.plot(theta, bands['posterior_mean'], color=theme['mean'], linewidth=6,
            solid_capstyle='round', zorder=3)
    ax.scatter(months, data['temperature'], s=220, color=theme['points'],
               edgecolors=theme['background'], linewidth=1.2, zorder=4)
    ax.set_xlabel('Month', fontsize=27, labelpad=32, color=theme['ink'])
    ax.set_ylabel('Temperature (°C)', fontsize=27, labelpad=55, color=theme['ink'])
    assert len(fig.axes) == 1 and not ax.get_title() and not ax.texts and not fig.texts
    assert ax.get_legend() is None
    # Check physical marker separation after the polar transform, including
    # the December–January pair. All twelve measurements remain visible.
    fig.canvas.draw()
    positions = ax.transData.transform(np.column_stack([months, data['temperature']]))
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    diameter = (np.sqrt(220) + 1.2) * fig.dpi / 72
    assert distances.min() > diameter * 1.5, (distances.min(), diameter)
    print(f'{path.name}: minimum point separation {distances.min():.0f}px; '
          f'marker diameter {diameter:.0f}px')
    fig.savefig(path, dpi=240, facecolor=theme['background'])
    plt.close(fig)


def main():
    data, theta, bands = fit()
    for name, theme in THEMES.items():
        suffix = '_dark' if name == 'dark' else ''
        draw(data, theta, bands, theme, ROOT / f'docs/images/readme_temperature{suffix}.png')


if __name__ == '__main__':
    main()
