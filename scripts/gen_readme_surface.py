"""Compare noisy data with a real fitted Matérn response surface.

The seeded synthetic benchmark is shared with scripts/docs_figures/render_all.py.
Only the observations and Model.predict's posterior mean are rendered. The two
panels share their limits, aspect ratio and camera; the mesh follows evaluated
predictor coordinates. No ground-truth surface is substituted for the fit.

Run from the repository root: python -m scripts.gen_readme_surface
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import gamfit
from scripts.docs_figures.render_all import make_regression, truth_2d

OUTPUT = Path(__file__).resolve().parents[1] / 'docs' / 'images'
THEMES = {
    'light': dict(background='#ffffff', ink='#526071', points='#226b91',
                  surface='#86b9ce', mesh='#285d75', pane='#f8fafc', axis='#c4ccd5'),
    'dark': dict(background='#0d1117', ink='#a8b4c2', points='#82c2df',
                 surface='#6ca9c2', mesh='#b7e0ed', pane='#131b25', axis='#465262'),
}


def fit():
    data = make_regression(n=800, seed=7)
    model = gamfit.fit(data, 'y ~ matern(x1, x2, centers=100)')
    coordinates = np.linspace(0, 1, 101)
    x1, x2 = np.meshgrid(coordinates, coordinates)
    prediction = model.predict({'x1': x1.ravel(), 'x2': x2.ravel()}, return_type='dict')
    mean = np.asarray(prediction['posterior_mean']).reshape(x1.shape)
    assert np.isfinite(mean).all()
    rmse = np.sqrt(np.mean((mean - truth_2d(x1, x2)) ** 2))
    print(f'800 seeded observations; fitted surface RMSE against generating function: {rmse:.4f}')
    return data, x1, x2, mean


def draw(data, x1, x2, mean, theme, path):
    plt.rcParams.update({'font.family': ['Helvetica Neue', 'DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 15,
                         'axes.labelcolor': theme['ink'],
                         'xtick.color': theme['ink'], 'ytick.color': theme['ink'],
                         'savefig.bbox': None})
    fig = plt.figure(figsize=(16, 9), dpi=240, facecolor=theme['background'])
    axes = [fig.add_axes(rect, projection='3d', facecolor=theme['background'])
            for rect in ((0.005, 0.06, 0.47, 0.89), (0.500, 0.06, 0.47, 0.89))]
    # No response colourmap: height alone encodes response in both panels.
    axes[0].scatter(data['x1'], data['x2'], data['y'], s=15,
                    color=theme['points'], linewidth=0, depthshade=False, alpha=0.8)
    axes[1].plot_surface(x1, x2, mean, color=theme['surface'],
                         rcount=101, ccount=101, linewidth=0,
                         antialiased=False, shade=True, alpha=1)
    axes[1].plot_wireframe(x1, x2, mean, rstride=5, cstride=5,
                           color=theme['mesh'], linewidth=0.5, alpha=0.65)
    lower = min(np.min(data['y']), mean.min()) - 0.08
    upper = max(np.max(data['y']), mean.max()) + 0.08
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(lower, upper),
               xlabel='x₁', ylabel='x₂', zlabel='y')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0.5, 1])
        ax.set_zticks([-1, 0, 1])
        ax.view_init(elev=28, azim=-62)
        ax.set_box_aspect((1, 1, 0.85))
        ax.set_proj_type('ortho')
        ax.grid(False)
        ax.tick_params(labelsize=11, pad=3, colors=theme['ink'])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(theme['pane'])
            axis.pane.set_edgecolor(theme['axis'])
            axis.pane.set_alpha(0.3)
            axis.line.set_color(theme['axis'])
            axis.label.set_color(theme['ink'])
        assert not ax.get_title() and not ax.texts and ax.get_legend() is None
    assert not fig.texts
    fig.savefig(path, facecolor=theme['background'], dpi=240)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, x1, x2, mean = fit()
    for name, theme in THEMES.items():
        suffix = '_dark' if name == 'dark' else ''
        draw(data, x1, x2, mean, theme, OUTPUT / f'readme_surface{suffix}.png')


if __name__ == '__main__':
    main()
