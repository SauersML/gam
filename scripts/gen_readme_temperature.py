"""One 3-D loop of all 366 daily temperatures and a periodic fit.

NOAA GHCN-Daily station USW00094728 (New York Central Park), 2024.
Each observation is a quality-approved daily (TMAX + TMIN) / 2 in Celsius.
No averaging into months or downsampling: all 366 daily observations appear.
Angle encodes day of year, while both height and colour encode temperature.
The continuous fitted loop has a vertical 95% credible ribbon. Its mean and
bounds wrap at the year boundary; calendar months mark the circular base axis.

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
FORMULA = 'temperature ~ s(phase, periodic=true, period=1, k=60)'
THEMES = {
    'light': dict(background='#ffffff', ink='#45566c', grid='#dce5ef',
                  mean='#1767c4', band='#489ad4', points='#182e47'),
    'dark': dict(background='#0d1117', ink='#b7c8dc', grid='#2a394d',
                 mean='#78c8ff', band='#489ad4', points='#f0f6fc'),
}


def daily_data():
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
    temperature = np.concatenate([
        (daily[month, 'TMAX'] + daily[month, 'TMIN']) / 2
        for month in range(1, 13)
    ])
    assert len(temperature) == 366 and np.isfinite(temperature).all()
    return {'phase': (np.arange(366) + 0.5) / 366, 'temperature': temperature}


def fit():
    data = daily_data()
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
    print('366 measured daily temperatures; fitted mean, 95% credible bounds and seam derivatives verified.')
    return data, phase * 2 * np.pi, bands


def draw(data, theta, bands, theme, path):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    plt.rcParams.update({'font.family': ['Helvetica Neue', 'DejaVu Sans'],
                         'font.size': 24, 'savefig.bbox': None})
    fig = plt.figure(figsize=(14, 11), dpi=240, facecolor=theme['background'])
    ax = fig.add_axes((0.025, 0.075, 0.91, 0.89), projection='3d',
                     facecolor=theme['background'], computed_zorder=False)
    normalize = Normalize(-10, 35)
    palette = LinearSegmentedColormap.from_list('temperature', [
        '#233b91', '#405cc4', '#7870c5', '#c36090', '#e97250', '#f0b353',
    ])
    mean = bands['posterior_mean']
    lower, upper = bands['posterior_mean_lower'], bands['posterior_mean_upper']
    x, y = np.cos(theta), np.sin(theta)
    positions = np.column_stack([x, y, mean])
    # The central loop's geometry and colour both encode predicted temperature.
    segments = np.stack([positions[:-1], positions[1:]], axis=1)
    colors = palette(normalize((mean[:-1] + mean[1:]) / 2))
    curve = Line3DCollection(segments, colors=colors, linewidths=8,
                             capstyle='round', zorder=5)
    # A vertical ribbon has exactly the model's 95% credible bounds as edges.
    quads = np.stack([
        np.column_stack([x[:-1], y[:-1], lower[:-1]]),
        np.column_stack([x[1:], y[1:], lower[1:]]),
        np.column_stack([x[1:], y[1:], upper[1:]]),
        np.column_stack([x[:-1], y[:-1], upper[:-1]]),
    ], axis=1)
    ax.add_collection3d(Poly3DCollection(quads, facecolors=colors,
                                        edgecolors='none', alpha=0.4, zorder=3))
    ax.add_collection3d(curve)

    days = data['phase'] * 2 * np.pi
    dx, dy = np.cos(days), np.sin(days)
    # Every daily observation is plotted, including short weather fluctuations.
    observations = ax.scatter(dx, dy, data['temperature'], s=95,
               c=palette(normalize(data['temperature'])),
               edgecolors=theme['background'], linewidth=0.5,
               depthshade=False, alpha=0.85, zorder=4)
    assert len(observations._offsets3d[0]) == 366
    daily_positions = np.column_stack([dx, dy, data['temperature']])
    raw_segments = np.stack([daily_positions[:-1], daily_positions[1:]], axis=1)
    raw_colors = palette(normalize((data['temperature'][:-1] + data['temperature'][1:])/2))
    ax.add_collection3d(Line3DCollection(raw_segments, colors=raw_colors,
                                        linewidths=1.2, alpha=0.3, zorder=2))
    month_lengths = np.asarray([calendar.monthrange(2024, month)[1] for month in range(1, 13)])
    month_centers = (np.cumsum(month_lengths) - month_lengths/2) / 366
    months = month_centers * 2 * np.pi

    # The floor circle is the month axis, not a second data series.
    ax.plot(1.3*x, 1.3*y, np.full_like(theta, -10), color=theme['grid'], linewidth=1.6, zorder=1)
    for month, angle in enumerate(months, 1):
        ax.plot([1.3*np.cos(angle), 1.38*np.cos(angle)],
                [1.3*np.sin(angle), 1.38*np.sin(angle)], [-10, -10],
                color=theme['ink'], linewidth=2, zorder=1)
        ax.text(1.53*np.cos(angle), 1.53*np.sin(angle), -11, str(month),
                color=theme['ink'], fontsize=24, ha='center', va='center', zorder=6)
    ax.text(0.4, -1.8, -12, 'Month', color=theme['ink'], fontsize=26,
            ha='center', va='center', zorder=6)

    ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), zlim=(-13, 35))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([-10, 0, 10, 20, 30])
    ax.set_zlabel('Temperature (°C)', fontsize=26, labelpad=22, color=theme['ink'])
    ax.tick_params(axis='z', labelsize=23, pad=8, colors=theme['ink'])
    ax.set_box_aspect((1, 1, 0.9), zoom=1.20)
    ax.set_proj_type('ortho')
    ax.view_init(elev=28, azim=-55)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color(theme['grid'])
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    assert len(fig.axes) == 1 and not ax.get_title() and not fig.texts
    assert ax.get_legend() is None
    fig.canvas.draw()
    fig.savefig(path, dpi=240, facecolor=theme['background'])
    plt.close(fig)


def main():
    data, theta, bands = fit()
    for name, theme in THEMES.items():
        suffix = '_dark' if name == 'dark' else ''
        draw(data, theta, bands, theme, ROOT / f'docs/images/readme_temperature{suffix}.png')


if __name__ == '__main__':
    main()
