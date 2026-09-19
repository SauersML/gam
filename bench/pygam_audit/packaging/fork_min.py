import os, time, signal, numpy as np
import gamfit
rng=np.random.default_rng(0); x=rng.uniform(0,1,400); y=np.sin(6*x)+rng.normal(0,.3,400)
pid=os.fork()
if pid==0:
    import faulthandler, sys
    faulthandler.dump_traceback_later(15, exit=True, file=sys.stderr)
    m=gamfit.fit({"x":x,"y":y},"y ~ s(x)")
    print("child fit OK", flush=True); os._exit(0)
t0=time.time()
while time.time()-t0<25:
    r=os.waitpid(pid, os.WNOHANG)
    if r[0]: print("child exited", r, round(time.time()-t0,1)); break
    time.sleep(0.5)
else:
    for t in os.listdir(f"/proc/{pid}/task"):
        print("task", t, open(f"/proc/{pid}/task/{t}/wchan").read(), open(f"/proc/{pid}/task/{t}/stat").read().split()[2])
    os.kill(pid, signal.SIGKILL); print("child HUNG, killed")
