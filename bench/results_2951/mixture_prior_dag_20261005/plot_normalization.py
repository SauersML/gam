"""Plot the exact normalizer of the audited two-block mixture family, not model data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
a=np.linspace(0,0.995,700)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax,dim,color in zip(axes,[1,128],['#2764a5','#b44d32']):
    missing=np.logaddexp(np.log(0.75),np.log(0.25)-dim*np.log1p(-a*a))/np.log(2)
    ax.plot(a,missing,color=color,lw=2.6,label='Circular conditional product')
    ax.axhline(0,color='#4c8664',lw=2,ls='--',label='Directed conditional prior')
    ax.set(xlabel='Mutual coefficient a',ylabel='Missing normalization cost (bits)',title=f'{dim}-dimensional parameter blocks',xlim=(0,1))
    ax.grid(alpha=.16)
    ax.legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('The former sharing prior omitted a normalization cost',fontsize=16,fontweight='bold')
fig.supxlabel('Exact family: Z = 3/4 + (1/4) |1 − a²|⁻ᵈ.  At a = 1 the circular product is nonintegrable.\nAnalytic diagnostic of the prior; these are not language-model measurements.',fontsize=10)
fig.savefig(out/'normalization.png',dpi=180)
fig.savefig(out/'normalization.pdf')
