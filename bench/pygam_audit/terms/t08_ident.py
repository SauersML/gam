from common import *
rng=np.random.default_rng(102); x=rng.uniform(0,1,200); y=1/(1+np.exp(-30*(x-.5)))+rng.normal(0,.3,200)
for form in ['y ~ s(x)','y ~ s(x, shape=monotone_increasing)','y ~ s(x, shape=convex)']:
    m=gamfit.fit(dict(x=x,y=y),form)
    st=m._coefficient_state(); p=len(st['beta']); C=np.array(st['covariance_flat']).reshape(p,p)
    print(form, "intercept beta=%.3f sd=%.3f"%(st['beta'][0], np.sqrt(C[0,0])), "corr(b0,b1)=%.4f"%(C[0,1]/np.sqrt(C[0,0]*C[1,1])), "max sd=%.3f"%np.sqrt(np.diag(C)).max())
    g=np.linspace(0,1,201); out=m.predict(dict(x=g), interval=0.95)
    if isinstance(out,dict) or hasattr(out,'keys'):
        ks=list(out.keys()); print("  keys",ks)
        lo=np.asarray(out.get('posterior_mean_lower', out.get('lower'))); hi=np.asarray(out.get('posterior_mean_upper', out.get('upper')))
        print("  band width mean=%.3f; lower worst decrease=%.3g upper worst decrease=%.3g"%(np.mean(hi-lo), check_shape(lo,g,'inc'), check_shape(hi,g,'inc')))
    else: print(" ",type(out))
    s=m.summary(); print("  ", str(s)[:600].replace("\n","\n   "))
