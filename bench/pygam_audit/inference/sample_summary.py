import json, numpy as np, sys
L=[json.loads(l) for l in open(sys.argv[1])]
g=[o['gamfit'] for o in L if 'error' not in o['gamfit']]; q=[o['pygam_sample'] for o in L if 'error' not in o['pygam_sample']]
both=[o for o in L if 'error' not in o['gamfit'] and 'error' not in o['pygam_sample']]
print('reps',len(L),'gamfit ok',len(g),'pygam ok',len(q),'both',len(both))
m=lambda xs,k: float(np.mean([x[k] for x in xs]))
print('gamfit sample cover %.3f width %.3f time %.2fs | predict cover %.3f width %.3f | %s %s'%(m(g,'cover'),m(g,'width'),m(g,'time'),m(g,'pred_cover'),m(g,'pred_width'),g[0]['method'],g[0]['cov_src']))
print('pygam  sample cover %.3f width %.3f time %.2fs'%(m(q,'cover'),m(q,'width'),m(q,'time')))
print('paired (both ok): gamfit sample %.3f  gamfit predict %.3f  pygam %.3f'%(np.mean([o['gamfit']['cover'] for o in both]),np.mean([o['gamfit']['pred_cover'] for o in both]),np.mean([o['pygam_sample']['cover'] for o in both])))
