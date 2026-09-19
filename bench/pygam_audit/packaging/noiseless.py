import numpy as np, gamfit, warnings
warnings.simplefilter("ignore")
x=np.linspace(0,1,500)
for sd in [0.0, 1e-6, 1e-3, 0.1]:
    y=np.sin(6*x)+np.random.default_rng(1).normal(0,1,500)*sd
    try:
        m=gamfit.fit({'x':x,'y':y},'y ~ s(x)'); print(sd,'ok', m.summary() is not None)
    except Exception as e: print(sd,'FAIL',type(e).__name__, str(e)[:100])
import pygam
print('pygam noiseless', pygam.LinearGAM(pygam.s(0)).fit(x[:,None],np.sin(6*x)).statistics_['edof'])
