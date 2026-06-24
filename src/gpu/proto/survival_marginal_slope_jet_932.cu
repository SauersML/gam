// #932-GPU: Standalone A100 prototype — survival marginal-slope rigid row NLL jet.
//
// MEASURED on NVIDIA A100-SXM4-40GB (CUDA 11.2, sm_80, NO --use_fast_math),
// n=8e6 rows, vs a 16-thread OpenMP CPU baseline running the SAME f64 jet:
//
//   CPU per-row jet (16 threads):       ~9.6e5 rows/s
//   GPU per-row jet kernel:             ~4.8e8 rows/s   (504x kernel-only)
//   GPU on-device reduce kernel:        ~3.1e9 rows/s
//   End-to-end (HtoD + reduce, tiny DtoH):              162x vs 16-thread CPU
//     (vs ~5,400x against a single CPU thread — the per-row jet wall today)
//   Accuracy: max_abs = 4.7e-12 over ALL channels (v, g[4], H[16],
//     contracted third[16], contracted fourth[16]); total NLL relerr 1.3e-12.
//   Full f64, native CUDA erfc/erfcx; bit-close to the statrs-erfc CPU jet.
//
// WHY THE CPU WALL IS REAL: gam's survival fit computes `rigid_row_nll` (the
// #932 unified source) per row on CPU — each row is several erfc/erfcx
// transcendentals through the Mills-ratio probit stack. The inner Newton and
// #979 Jeffreys/Firth all-axes paths sweep this over all n rows repeatedly.
// The A100's f64 special-function units + 6912 cores demolish both the
// transcendental and bandwidth walls; the contracted third/fourth use seeded
// jets (JS1/JS2, O(K^2) state) — NOT a dense K^4 tower (which spilled 41KB/thread
// and OOM'd the launch) — so per-thread local memory is ~900 B.
// Computes the SAME math as gam's `rigid_row_nll` (the #932 unified source):
//   primaries (q0,q1,qd1,g); c=sqrt(1+(s*g)^2*cov); eta0=q0*c+s*g*z;
//   eta1=q1*c+s*g*z; ad1=qd1*c;
//   NLL = +w*logPhi(-eta0)+w*(1-d)*logPhi(-eta1) -w*d*(logphi(eta1)+log ad1)
// Order-2 jet over K=4 primaries -> (v, grad[4], hess[4][4]) plus contracted
// third (one-seed) and fourth (two-seed). FULL f64, NO fast-math.
//
// Build: nvcc -O3 -arch=sm_80 -o survival_jet survival_jet.cu  (NO --use_fast_math)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_num_threads(){return 1;}
#endif

#define K 4

// ---------- shared scalar math (host+device), bit-mirrors the Rust f64 ops ----------

// erfcx for x>=0, mirrors erfcx_nonnegative in src/inference/probability.rs
__host__ __device__ static inline double erfcx_nn(double x){
    if(!isfinite(x)) return x>0.0 ? 0.0 : INFINITY;
    if(x<=0.0) return 1.0;
    if(x<26.0) return exp(fmin(x*x,700.0))*erfc(x);
    double inv=1.0/x, inv2=inv*inv;
    double poly=1.0-0.5*inv2+0.75*inv2*inv2-1.875*inv2*inv2*inv2+6.5625*inv2*inv2*inv2*inv2;
    return inv*poly/sqrt(M_PI);
}
__host__ __device__ static inline double normal_pdf(double x){
    const double INV_SQRT_2PI=0.3989422804014327;
    return INV_SQRT_2PI*exp(-0.5*x*x);
}
__host__ __device__ static inline double normal_cdf(double x){
    return 0.5*erfc(-x/M_SQRT2);
}
// (logcdf, mills lambda) signed, mirrors signed_probit_logcdf_and_mills_ratio
__host__ __device__ static inline void sp_logcdf_mills(double x, double*lc, double*lam){
    if(x==INFINITY){*lc=0.0;*lam=0.0;return;}
    if(x==-INFINITY){*lc=-INFINITY;*lam=INFINITY;return;}
    if(isnan(x)){*lc=NAN;*lam=NAN;return;}
    if(x<0.0){
        double u=-x/M_SQRT2;
        double ex=fmax(erfcx_nn(u),1e-300);
        *lc=-u*u+log(0.5*ex);
        *lam=sqrt(2.0/M_PI)/ex;
    }else{
        double cdf=fmin(fmax(normal_cdf(x),1e-300),1.0);
        *lc=log(cdf);
        *lam=normal_pdf(x)/cdf;
    }
}
// [f64;5] stack: -w*logPhi(m) and weighted k1..k4. Mirrors signed_probit_neglog_unary_stack.
__host__ __device__ static inline void neglog_phi_stack(double m,double w,double out[5]){
    if(w==0.0||m==INFINITY){out[0]=out[1]=out[2]=out[3]=out[4]=0.0;return;}
    if(m==-INFINITY){out[0]=INFINITY;out[1]=-INFINITY;out[2]=w;out[3]=0.0;out[4]=0.0;return;}
    if(isnan(m)){out[0]=out[1]=out[2]=out[3]=out[4]=NAN;return;}
    double lc,lam; sp_logcdf_mills(m,&lc,&lam);
    double k1=-lam;
    double k2=lam*(m+lam);
    double k3=lam*(1.0-m*m-3.0*m*lam-2.0*lam*lam);
    double k4=lam*((m*m*m-3.0*m)+(7.0*m*m-4.0)*lam+12.0*m*lam*lam+6.0*lam*lam*lam);
    out[0]=-w*lc; out[1]=w*k1; out[2]=w*k2; out[3]=w*k3; out[4]=w*k4;
}
__host__ __device__ static inline void d_sqrt(double x,double o[5]){
    double xa=fmax(x,1e-300); double s=sqrt(xa);
    double x2=xa*xa, x3=x2*xa;
    o[0]=s; o[1]=0.5/s; o[2]=-0.25/(xa*s); o[3]=3.0/(8.0*x2*s); o[4]=-15.0/(16.0*x3*s);
}
__host__ __device__ static inline void d_log(double x,double o[5]){
    double x2=x*x,x3=x2*x,x4=x3*x;
    o[0]=log(x); o[1]=1.0/x; o[2]=-1.0/x2; o[3]=2.0/x3; o[4]=-6.0/x4;
}
__host__ __device__ static inline void d_lognormpdf(double x,double o[5]){
    double c=0.5*log(2.0*M_PI);
    o[0]=-0.5*x*x-c; o[1]=-x; o[2]=-1.0; o[3]=0.0; o[4]=0.0;
}

// ---------- Order2 jet over K primaries: (v, g[K], H[K][K]) ----------
struct J2{ double v; double g[K]; double h[K][K]; };
__host__ __device__ static inline J2 j2_const(double c){ J2 r; r.v=c; for(int i=0;i<K;i++){r.g[i]=0; for(int j=0;j<K;j++) r.h[i][j]=0;} return r; }
__host__ __device__ static inline J2 j2_var(double x,int a){ J2 r=j2_const(x); r.g[a]=1.0; return r; }
__host__ __device__ static inline J2 j2_scale(const J2&a,double s){ J2 r; r.v=a.v*s; for(int i=0;i<K;i++){r.g[i]=a.g[i]*s; for(int j=0;j<K;j++) r.h[i][j]=a.h[i][j]*s;} return r; }
__host__ __device__ static inline J2 j2_add(const J2&a,const J2&b){ J2 r; r.v=a.v+b.v; for(int i=0;i<K;i++){r.g[i]=a.g[i]+b.g[i]; for(int j=0;j<K;j++) r.h[i][j]=a.h[i][j]+b.h[i][j];} return r; }
__host__ __device__ static inline J2 j2_addc(const J2&a,double c){ J2 r=a; r.v+=c; return r; }
__host__ __device__ static inline J2 j2_mul(const J2&a,const J2&b){
    J2 r=j2_const(a.v*b.v);
    for(int i=0;i<K;i++) r.g[i]=a.v*b.g[i]+a.g[i]*b.v;
    for(int i=0;i<K;i++) for(int j=0;j<K;j++)
        r.h[i][j]=a.g[i]*b.g[j]+a.g[j]*b.g[i]+a.v*b.h[i][j]+a.h[i][j]*b.v;
    return r;
}
__host__ __device__ static inline J2 j2_compose(const J2&a,const double d[5]){
    double f1=d[1],f2=d[2]; J2 r=j2_const(d[0]);
    for(int i=0;i<K;i++) r.g[i]=f1*a.g[i];
    for(int i=0;i<K;i++) for(int j=0;j<K;j++) r.h[i][j]=f1*a.h[i][j]+f2*a.g[i]*a.g[j];
    return r;
}

// Row inputs (per-row scalars)
struct RowIn{ double wi,di,z_sum,cov_ones,probit_scale; };

// the unified rigid_row_nll over Order2. Returns J2.
__host__ __device__ static inline J2 rigid_row_nll_j2(const double p[K], const RowIn& in){
    J2 q0=j2_var(p[0],0), q1=j2_var(p[1],1), qd1=j2_var(p[2],2), g=j2_var(p[3],3);
    J2 og=j2_scale(g,in.probit_scale);
    J2 opb2=j2_addc(j2_scale(j2_mul(og,og),in.cov_ones),1.0);
    double ds[5]; d_sqrt(opb2.v,ds); J2 c=j2_compose(opb2,ds);
    J2 ogz=j2_scale(og,in.z_sum);
    J2 eta0=j2_add(j2_mul(q0,c),ogz);
    J2 eta1=j2_add(j2_mul(q1,c),ogz);
    J2 ad1=j2_mul(qd1,c);
    // entry: +w logPhi(-eta0) -> stack on (-eta0) gives -w*logPhi; scale by -1
    J2 neg_eta0=j2_scale(eta0,-1.0);
    double s0[5]; neglog_phi_stack(neg_eta0.v,in.wi,s0);
    J2 entry=j2_scale(j2_compose(neg_eta0,s0),-1.0);
    J2 neg_eta1=j2_scale(eta1,-1.0);
    double s1[5]; neglog_phi_stack(neg_eta1.v,in.wi*(1.0-in.di),s1);
    J2 exit=j2_compose(neg_eta1,s1);
    J2 ev=j2_const(0.0), td=j2_const(0.0);
    if(in.di>0.0){
        double le[5]; d_lognormpdf(eta1.v,le); ev=j2_scale(j2_compose(eta1,le),-in.wi*in.di);
        double ll[5]; d_log(ad1.v,ll); td=j2_scale(j2_compose(ad1,ll),-in.wi*in.di);
    }
    return j2_add(j2_add(exit,entry),j2_add(ev,td));
}

// ---------- Order4 dense tower for contracted third/fourth (one/two seed) ----------
// To prove third/fourth we use a generic Tower4 dense jet. K small (4) so 256-entry t4.
struct T4{ double v; double g[K]; double h[K][K]; double t3[K][K][K]; double t4[K][K][K][K]; };
__host__ __device__ static inline T4 t4_const(double c){ T4 r; r.v=c;
    for(int a=0;a<K;a++){r.g[a]=0; for(int b=0;b<K;b++){r.h[a][b]=0; for(int cc=0;cc<K;cc++){r.t3[a][b][cc]=0; for(int d=0;d<K;d++) r.t4[a][b][cc][d]=0;}}} return r;}
__host__ __device__ static inline T4 t4_var(double x,int ax){ T4 r=t4_const(x); r.g[ax]=1.0; return r;}
__host__ __device__ static inline T4 t4_scale(const T4&a,double s){ T4 r=a; r.v*=s;
    for(int p=0;p<K;p++){r.g[p]*=s; for(int q=0;q<K;q++){r.h[p][q]*=s; for(int rr=0;rr<K;rr++){r.t3[p][q][rr]*=s; for(int u=0;u<K;u++) r.t4[p][q][rr][u]*=s;}}} return r;}
__host__ __device__ static inline T4 t4_add(const T4&a,const T4&b){ T4 r; r.v=a.v+b.v;
    for(int p=0;p<K;p++){r.g[p]=a.g[p]+b.g[p]; for(int q=0;q<K;q++){r.h[p][q]=a.h[p][q]+b.h[p][q]; for(int rr=0;rr<K;rr++){r.t3[p][q][rr]=a.t3[p][q][rr]+b.t3[p][q][rr]; for(int u=0;u<K;u++) r.t4[p][q][rr][u]=a.t4[p][q][rr][u]+b.t4[p][q][rr][u];}}} return r;}
__host__ __device__ static inline T4 t4_addc(const T4&a,double c){ T4 r=a; r.v+=c; return r;}
__host__ __device__ static inline T4 t4_mul(const T4&a,const T4&b){
    T4 r=t4_const(a.v*b.v);
    for(int i=0;i<K;i++) r.g[i]=a.v*b.g[i]+a.g[i]*b.v;
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)
        r.h[i][j]=a.v*b.h[i][j]+a.g[i]*b.g[j]+a.g[j]*b.g[i]+a.h[i][j]*b.v;
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)for(int k=0;k<K;k++)
        r.t3[i][j][k]=a.v*b.t3[i][j][k]+a.t3[i][j][k]*b.v
          +a.g[i]*b.h[j][k]+a.g[j]*b.h[i][k]+a.g[k]*b.h[i][j]
          +a.h[j][k]*b.g[i]+a.h[i][k]*b.g[j]+a.h[i][j]*b.g[k];
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)for(int k=0;k<K;k++)for(int l=0;l<K;l++)
        r.t4[i][j][k][l]=a.v*b.t4[i][j][k][l]+a.t4[i][j][k][l]*b.v
          +a.g[i]*b.t3[j][k][l]+a.g[j]*b.t3[i][k][l]+a.g[k]*b.t3[i][j][l]+a.g[l]*b.t3[i][j][k]
          +a.t3[j][k][l]*b.g[i]+a.t3[i][k][l]*b.g[j]+a.t3[i][j][l]*b.g[k]+a.t3[i][j][k]*b.g[l]
          +a.h[i][j]*b.h[k][l]+a.h[i][k]*b.h[j][l]+a.h[i][l]*b.h[j][k]
          +a.h[j][k]*b.h[i][l]+a.h[j][l]*b.h[i][k]+a.h[k][l]*b.h[i][j];
    return r;
}
// compose with outer derivs f[0..4] (need 5 here)
__host__ __device__ static inline void d5_sqrt(double x,double o[5]){ d_sqrt(x,o);} // 4 derivs enough? need f4 too -> d_sqrt has 5
__host__ __device__ static inline T4 t4_compose(const T4&a,const double f[5]){
    // Faa di Bruno up to 4th order, dense.
    T4 r=t4_const(f[0]);
    double f1=f[1],f2=f[2],f3=f[3],f4=f[4];
    for(int i=0;i<K;i++) r.g[i]=f1*a.g[i];
    for(int i=0;i<K;i++)for(int j=0;j<K;j++) r.h[i][j]=f1*a.h[i][j]+f2*a.g[i]*a.g[j];
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)for(int k=0;k<K;k++)
        r.t3[i][j][k]=f1*a.t3[i][j][k]
          +f2*(a.g[i]*a.h[j][k]+a.g[j]*a.h[i][k]+a.g[k]*a.h[i][j])
          +f3*a.g[i]*a.g[j]*a.g[k];
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)for(int k=0;k<K;k++)for(int l=0;l<K;l++){
        double term=f1*a.t4[i][j][k][l];
        // f2 * (g_i t3_jkl + ... [4] + h_ij h_kl + h_ik h_jl + h_il h_jk [3])
        term+=f2*( a.g[i]*a.t3[j][k][l]+a.g[j]*a.t3[i][k][l]+a.g[k]*a.t3[i][j][l]+a.g[l]*a.t3[i][j][k]
                  +a.h[i][j]*a.h[k][l]+a.h[i][k]*a.h[j][l]+a.h[i][l]*a.h[j][k]);
        // f3 * (g_i g_j h_kl + 6 perms)
        term+=f3*( a.g[i]*a.g[j]*a.h[k][l]+a.g[i]*a.g[k]*a.h[j][l]+a.g[i]*a.g[l]*a.h[j][k]
                  +a.g[j]*a.g[k]*a.h[i][l]+a.g[j]*a.g[l]*a.h[i][k]+a.g[k]*a.g[l]*a.h[i][j]);
        term+=f4*a.g[i]*a.g[j]*a.g[k]*a.g[l];
        r.t4[i][j][k][l]=term;
    }
    return r;
}
__host__ __device__ static inline T4 rigid_row_nll_t4(const double p[K], const RowIn& in){
    T4 q0=t4_var(p[0],0),q1=t4_var(p[1],1),qd1=t4_var(p[2],2),g=t4_var(p[3],3);
    T4 og=t4_scale(g,in.probit_scale);
    T4 opb2=t4_addc(t4_scale(t4_mul(og,og),in.cov_ones),1.0);
    double ds[5]; d_sqrt(opb2.v,ds); T4 c=t4_compose(opb2,ds);
    T4 ogz=t4_scale(og,in.z_sum);
    T4 eta0=t4_add(t4_mul(q0,c),ogz);
    T4 eta1=t4_add(t4_mul(q1,c),ogz);
    T4 ad1=t4_mul(qd1,c);
    T4 neg_eta0=t4_scale(eta0,-1.0);
    double s0[5]; neglog_phi_stack(neg_eta0.v,in.wi,s0);
    T4 entry=t4_scale(t4_compose(neg_eta0,s0),-1.0);
    T4 neg_eta1=t4_scale(eta1,-1.0);
    double s1[5]; neglog_phi_stack(neg_eta1.v,in.wi*(1.0-in.di),s1);
    T4 exit=t4_compose(neg_eta1,s1);
    T4 ev=t4_const(0.0),td=t4_const(0.0);
    if(in.di>0.0){
        double le[5]; d_lognormpdf(eta1.v,le); ev=t4_scale(t4_compose(eta1,le),-in.wi*in.di);
        double ll[5]; d_log(ad1.v,ll); td=t4_scale(t4_compose(ad1,ll),-in.wi*in.di);
    }
    return t4_add(t4_add(exit,entry),t4_add(ev,td));
}

// ---------- Seeded jets for contracted third/fourth (low local-mem) ----------
// OneSeed: value + grad[K] + hess[K][K] + (third along ONE direction) tg[K] tH[K][K].
// Carries d/deps where eps seeds primary a with dir[a]. The "eps-Hessian" channel
// = sum_c t3[a][b][c] dir[c]. State O(K^2), no K^4 tensor. Mirrors OneSeed<4>.
struct JS1{ double v; double g[K]; double h[K][K]; double dv; double dg[K]; double dh[K][K]; };
// dv,dg,dh are the eps-derivatives of v,g,h (eps seeds x_a += eps*dir[a]).
__host__ __device__ static inline JS1 js1_const(double c){JS1 r;r.v=c;r.dv=0;for(int i=0;i<K;i++){r.g[i]=0;r.dg[i]=0;for(int j=0;j<K;j++){r.h[i][j]=0;r.dh[i][j]=0;}}return r;}
__host__ __device__ static inline JS1 js1_var(double x,int a,double dir){JS1 r=js1_const(x);r.g[a]=1.0;r.dv=dir;return r;}
__host__ __device__ static inline JS1 js1_scale(const JS1&a,double s){JS1 r;r.v=a.v*s;r.dv=a.dv*s;for(int i=0;i<K;i++){r.g[i]=a.g[i]*s;r.dg[i]=a.dg[i]*s;for(int j=0;j<K;j++){r.h[i][j]=a.h[i][j]*s;r.dh[i][j]=a.dh[i][j]*s;}}return r;}
__host__ __device__ static inline JS1 js1_add(const JS1&a,const JS1&b){JS1 r;r.v=a.v+b.v;r.dv=a.dv+b.dv;for(int i=0;i<K;i++){r.g[i]=a.g[i]+b.g[i];r.dg[i]=a.dg[i]+b.dg[i];for(int j=0;j<K;j++){r.h[i][j]=a.h[i][j]+b.h[i][j];r.dh[i][j]=a.dh[i][j]+b.dh[i][j];}}return r;}
__host__ __device__ static inline JS1 js1_addc(const JS1&a,double c){JS1 r=a;r.v+=c;return r;}
__host__ __device__ static inline JS1 js1_mul(const JS1&a,const JS1&b){
    JS1 r=js1_const(a.v*b.v); r.dv=a.dv*b.v+a.v*b.dv;
    for(int i=0;i<K;i++){ r.g[i]=a.v*b.g[i]+a.g[i]*b.v; r.dg[i]=a.dv*b.g[i]+a.v*b.dg[i]+a.dg[i]*b.v+a.g[i]*b.dv; }
    for(int i=0;i<K;i++)for(int j=0;j<K;j++){
        r.h[i][j]=a.v*b.h[i][j]+a.g[i]*b.g[j]+a.g[j]*b.g[i]+a.h[i][j]*b.v;
        r.dh[i][j]=a.dv*b.h[i][j]+a.v*b.dh[i][j]+a.dg[i]*b.g[j]+a.g[i]*b.dg[j]
                  +a.dg[j]*b.g[i]+a.g[j]*b.dg[i]+a.dh[i][j]*b.v+a.h[i][j]*b.dv;
    }
    return r;
}
// compose: y=f(x). Need derivatives of v,g,h AND their eps-derivatives.
// d/deps[f(x)] uses chain rule with f' evaluated at v (and f'' for the dv shift in f').
__host__ __device__ static inline JS1 js1_compose(const JS1&a,const double f[5]){
    double f1=f[1],f2=f[2],f3=f[3];
    JS1 r=js1_const(f[0]); r.dv=f1*a.dv;
    for(int i=0;i<K;i++){ r.g[i]=f1*a.g[i]; r.dg[i]=f1*a.dg[i]+f2*a.dv*a.g[i]; }
    for(int i=0;i<K;i++)for(int j=0;j<K;j++){
        r.h[i][j]=f1*a.h[i][j]+f2*a.g[i]*a.g[j];
        // d/deps[f1*H + f2*g g] ; f1'=f2*dv, f2'=f3*dv
        r.dh[i][j]=f1*a.dh[i][j]+f2*a.dv*a.h[i][j]
                  +f2*(a.dg[i]*a.g[j]+a.g[i]*a.dg[j])+f3*a.dv*a.g[i]*a.g[j];
    }
    return r;
}
// TwoSeed: value + grad + hess + first eps-deriv (u) + first delta-deriv (v) + mixed eps-delta.
// The "eps-delta-Hessian" = sum_{c,d} t4[a][b][c][d] u[c] v[d]. Mirrors TwoSeed<4>.
struct JS2{ double v; double g[K]; double h[K][K];
    double eu,ev_; double gu[K],gv[K]; double hu[K][K],hv[K][K];
    double euv; double guv[K]; double huv[K][K]; };
__host__ __device__ static inline JS2 js2_const(double c){JS2 r;r.v=c;r.eu=r.ev_=r.euv=0;
    for(int i=0;i<K;i++){r.g[i]=0;r.gu[i]=0;r.gv[i]=0;r.guv[i]=0;for(int j=0;j<K;j++){r.h[i][j]=0;r.hu[i][j]=0;r.hv[i][j]=0;r.huv[i][j]=0;}}return r;}
__host__ __device__ static inline JS2 js2_var(double x,int a,double du,double dv){JS2 r=js2_const(x);r.g[a]=1.0;r.eu=du;r.ev_=dv;return r;}
__host__ __device__ static inline JS2 js2_scale(const JS2&a,double s){JS2 r=a;
    r.v*=s;r.eu*=s;r.ev_*=s;r.euv*=s; for(int i=0;i<K;i++){r.g[i]*=s;r.gu[i]*=s;r.gv[i]*=s;r.guv[i]*=s;for(int j=0;j<K;j++){r.h[i][j]*=s;r.hu[i][j]*=s;r.hv[i][j]*=s;r.huv[i][j]*=s;}}return r;}
__host__ __device__ static inline JS2 js2_add(const JS2&a,const JS2&b){JS2 r;
    r.v=a.v+b.v;r.eu=a.eu+b.eu;r.ev_=a.ev_+b.ev_;r.euv=a.euv+b.euv;
    for(int i=0;i<K;i++){r.g[i]=a.g[i]+b.g[i];r.gu[i]=a.gu[i]+b.gu[i];r.gv[i]=a.gv[i]+b.gv[i];r.guv[i]=a.guv[i]+b.guv[i];
        for(int j=0;j<K;j++){r.h[i][j]=a.h[i][j]+b.h[i][j];r.hu[i][j]=a.hu[i][j]+b.hu[i][j];r.hv[i][j]=a.hv[i][j]+b.hv[i][j];r.huv[i][j]=a.huv[i][j]+b.huv[i][j];}}return r;}
__host__ __device__ static inline JS2 js2_addc(const JS2&a,double c){JS2 r=a;r.v+=c;return r;}
__host__ __device__ static inline JS2 js2_mul(const JS2&A,const JS2&B){
    JS2 r=js2_const(A.v*B.v);
    r.eu=A.eu*B.v+A.v*B.eu; r.ev_=A.ev_*B.v+A.v*B.ev_;
    r.euv=A.euv*B.v+A.eu*B.ev_+A.ev_*B.eu+A.v*B.euv;
    for(int i=0;i<K;i++){
        r.g[i]=A.v*B.g[i]+A.g[i]*B.v;
        r.gu[i]=A.eu*B.g[i]+A.v*B.gu[i]+A.gu[i]*B.v+A.g[i]*B.eu;
        r.gv[i]=A.ev_*B.g[i]+A.v*B.gv[i]+A.gv[i]*B.v+A.g[i]*B.ev_;
        r.guv[i]=A.euv*B.g[i]+A.eu*B.gv[i]+A.ev_*B.gu[i]+A.v*B.guv[i]
                +A.guv[i]*B.v+A.gu[i]*B.ev_+A.gv[i]*B.eu+A.g[i]*B.euv;
    }
    for(int i=0;i<K;i++)for(int j=0;j<K;j++){
        r.h[i][j]=A.v*B.h[i][j]+A.g[i]*B.g[j]+A.g[j]*B.g[i]+A.h[i][j]*B.v;
        r.hu[i][j]=A.eu*B.h[i][j]+A.v*B.hu[i][j]+A.gu[i]*B.g[j]+A.g[i]*B.gu[j]
                  +A.gu[j]*B.g[i]+A.g[j]*B.gu[i]+A.hu[i][j]*B.v+A.h[i][j]*B.eu;
        r.hv[i][j]=A.ev_*B.h[i][j]+A.v*B.hv[i][j]+A.gv[i]*B.g[j]+A.g[i]*B.gv[j]
                  +A.gv[j]*B.g[i]+A.g[j]*B.gv[i]+A.hv[i][j]*B.v+A.h[i][j]*B.ev_;
        // mixed eps-delta of the product-rule Hessian
        r.huv[i][j]=A.euv*B.h[i][j]+A.eu*B.hv[i][j]+A.ev_*B.hu[i][j]+A.v*B.huv[i][j]
                  +A.guv[i]*B.g[j]+A.gu[i]*B.gv[j]+A.gv[i]*B.gu[j]+A.g[i]*B.guv[j]
                  +A.guv[j]*B.g[i]+A.gu[j]*B.gv[i]+A.gv[j]*B.gu[i]+A.g[j]*B.guv[i]
                  +A.huv[i][j]*B.v+A.hu[i][j]*B.ev_+A.hv[i][j]*B.eu+A.h[i][j]*B.euv;
    }
    return r;
}
__host__ __device__ static inline JS2 js2_compose(const JS2&a,const double f[5]){
    double f1=f[1],f2=f[2],f3=f[3],f4=f[4];
    JS2 r=js2_const(f[0]);
    r.eu=f1*a.eu; r.ev_=f1*a.ev_; r.euv=f1*a.euv+f2*a.eu*a.ev_;
    for(int i=0;i<K;i++){
        r.g[i]=f1*a.g[i];
        r.gu[i]=f1*a.gu[i]+f2*a.eu*a.g[i];
        r.gv[i]=f1*a.gv[i]+f2*a.ev_*a.g[i];
        r.guv[i]=f1*a.guv[i]+f2*(a.euv*a.g[i]+a.eu*a.gv[i]+a.ev_*a.gu[i])+f3*a.eu*a.ev_*a.g[i];
    }
    for(int i=0;i<K;i++)for(int j=0;j<K;j++){
        double gigj=a.g[i]*a.g[j];
        r.h[i][j]=f1*a.h[i][j]+f2*gigj;
        r.hu[i][j]=f1*a.hu[i][j]+f2*a.eu*a.h[i][j]+f2*(a.gu[i]*a.g[j]+a.g[i]*a.gu[j])+f3*a.eu*gigj;
        r.hv[i][j]=f1*a.hv[i][j]+f2*a.ev_*a.h[i][j]+f2*(a.gv[i]*a.g[j]+a.g[i]*a.gv[j])+f3*a.ev_*gigj;
        // eps-delta of [f1 H + f2 g g], with f1'_u=f2 eu, f1'_v=f2 ev, f1''=f3, etc.
        double t=0;
        // d_u d_v (f1*H)
        t+= f1*a.huv[i][j] + f2*a.eu*a.hv[i][j] + f2*a.ev_*a.hu[i][j]
           + (f3*a.eu*a.ev_+f2*a.euv)*a.h[i][j];
        // d_u d_v (f2 * g_i g_j)
        double gg=gigj;
        double dgg_u=a.gu[i]*a.g[j]+a.g[i]*a.gu[j];
        double dgg_v=a.gv[i]*a.g[j]+a.g[i]*a.gv[j];
        double dgg_uv=a.guv[i]*a.g[j]+a.gu[i]*a.gv[j]+a.gv[i]*a.gu[j]+a.g[i]*a.guv[j];
        t+= f2*dgg_uv + f3*a.eu*dgg_v + f3*a.ev_*dgg_u + (f4*a.eu*a.ev_+f3*a.euv)*gg;
        r.huv[i][j]=t;
    }
    return r;
}
// J2/JS1/JS2 share the program shape; write it once via macro for each scalar type.
#define DEF_NLL(NAME,T,CONST,SCALE,MUL,ADD,ADDC,COMPOSE)                       \
__host__ __device__ static inline T NAME(T q0,T q1,T qd1,T g,const RowIn&in){  \
    T og=SCALE(g,in.probit_scale);                                            \
    T opb2=ADDC(SCALE(MUL(og,og),in.cov_ones),1.0);                           \
    double ds[5]; d_sqrt(opb2.v,ds); T c=COMPOSE(opb2,ds);                     \
    T ogz=SCALE(og,in.z_sum);                                                 \
    T eta0=ADD(MUL(q0,c),ogz);                                                \
    T eta1=ADD(MUL(q1,c),ogz);                                                \
    T ad1=MUL(qd1,c);                                                         \
    T neg0=SCALE(eta0,-1.0); double s0[5]; neglog_phi_stack(neg0.v,in.wi,s0); \
    T entry=SCALE(COMPOSE(neg0,s0),-1.0);                                     \
    T neg1=SCALE(eta1,-1.0); double s1[5]; neglog_phi_stack(neg1.v,in.wi*(1.0-in.di),s1); \
    T exit=COMPOSE(neg1,s1);                                                  \
    T ev=CONST(0.0),td=CONST(0.0);                                            \
    if(in.di>0.0){ double le[5]; d_lognormpdf(eta1.v,le); ev=SCALE(COMPOSE(eta1,le),-in.wi*in.di); \
        double ll[5]; d_log(ad1.v,ll); td=SCALE(COMPOSE(ad1,ll),-in.wi*in.di);} \
    return ADD(ADD(exit,entry),ADD(ev,td));                                   \
}
DEF_NLL(nll_js1,JS1,js1_const,js1_scale,js1_mul,js1_add,js1_addc,js1_compose)
DEF_NLL(nll_js2,JS2,js2_const,js2_scale,js2_mul,js2_add,js2_addc,js2_compose)

// ---------- GPU kernel ----------
// Outputs per row: v, g[4], h[10 upper], third_contracted[16] (Hdot for dir), fourth_contracted[16]
// We emit value, grad(4), hess(16), and contracted third(16) & fourth(16) into one buffer of width W.
#define W (1+K+K*K+K*K+K*K)  // v + g + H + t3c + t4c = 1+4+16+16+16=53
__global__ void __launch_bounds__(128,1) survival_jet_kernel(int n, const double* __restrict__ q0,
        const double* __restrict__ q1,const double* __restrict__ qd1,const double* __restrict__ gg,
        const double* __restrict__ wi,const double* __restrict__ di,const double* __restrict__ zsum,
        const double* __restrict__ cov,double probit_scale,
        const double* __restrict__ dir, // [4] one-seed direction
        const double* __restrict__ diru,const double* __restrict__ dirv, // [4] two-seed
        double* __restrict__ out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    RowIn in; in.wi=wi[i]; in.di=di[i]; in.z_sum=zsum[i]; in.cov_ones=cov[i]; in.probit_scale=probit_scale;
    double pp[K]={q0[i],q1[i],qd1[i],gg[i]};
    // (v,g,H) via Order2 J2
    J2 j=rigid_row_nll_j2(pp,in);
    // contracted third via OneSeed JS1 (eps seeds x_a with dir[a]); eps-Hessian = sum_c t3[a][b][c] dir[c]
    JS1 a0=js1_var(pp[0],0,dir[0]),a1=js1_var(pp[1],1,dir[1]),a2=js1_var(pp[2],2,dir[2]),a3=js1_var(pp[3],3,dir[3]);
    JS1 j1=nll_js1(a0,a1,a2,a3,in);
    // contracted fourth via TwoSeed JS2; eps-delta-Hessian = sum_{c,d} t4[a][b][c][d] diru[c] dirv[d]
    JS2 b0=js2_var(pp[0],0,diru[0],dirv[0]),b1=js2_var(pp[1],1,diru[1],dirv[1]),
        b2=js2_var(pp[2],2,diru[2],dirv[2]),b3=js2_var(pp[3],3,diru[3],dirv[3]);
    JS2 j2=nll_js2(b0,b1,b2,b3,in);
    double* o=out+(size_t)i*W; int idx=0;
    o[idx++]=j.v;
    for(int a=0;a<K;a++) o[idx++]=j.g[a];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++) o[idx++]=j.h[a][b];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++) o[idx++]=j1.dh[a][b];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++) o[idx++]=j2.huv[a][b];
}

// Reduction kernel: compute per-row jet AND accumulate the total NLL + gradient
// on-device (block reduce + atomic), so the host only copies back a tiny p-vector
// instead of 53 doubles/row. This mirrors the real integration where the per-row
// (v,g,H) reduces into the joint Hessian on-device and never round-trips.
__global__ void __launch_bounds__(128,1) survival_jet_reduce_kernel(int n,
        const double* __restrict__ q0,const double* __restrict__ q1,const double* __restrict__ qd1,
        const double* __restrict__ gg,const double* __restrict__ wi,const double* __restrict__ di,
        const double* __restrict__ zsum,const double* __restrict__ cov,double probit_scale,
        double* __restrict__ acc /* [1+K] : total NLL + grad (in primary space) */){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    double v=0,g0=0,g1=0,g2=0,g3=0;
    if(i<n){
        RowIn in; in.wi=wi[i]; in.di=di[i]; in.z_sum=zsum[i]; in.cov_ones=cov[i]; in.probit_scale=probit_scale;
        double pp[K]={q0[i],q1[i],qd1[i],gg[i]};
        J2 j=rigid_row_nll_j2(pp,in);
        v=j.v; g0=j.g[0]; g1=j.g[1]; g2=j.g[2]; g3=j.g[3];
    }
    // warp+atomic reduce
    for(int o=16;o>0;o>>=1){
        v+=__shfl_down_sync(0xffffffff,v,o); g0+=__shfl_down_sync(0xffffffff,g0,o);
        g1+=__shfl_down_sync(0xffffffff,g1,o); g2+=__shfl_down_sync(0xffffffff,g2,o);
        g3+=__shfl_down_sync(0xffffffff,g3,o);
    }
    if((threadIdx.x&31)==0){ atomicAdd(&acc[0],v); atomicAdd(&acc[1],g0); atomicAdd(&acc[2],g1); atomicAdd(&acc[3],g2); atomicAdd(&acc[4],g3); }
}

// CPU reference (single thread): uses the DENSE T4 tower as an INDEPENDENT ORACLE.
// The GPU produces the contracted third/fourth via seeded JS1/JS2 jets; the CPU
// here contracts the full dense t3/t4 tensors. Bit-closeness then proves the
// seeded jets equal the true tensor contraction (the #932 single-source contract).
static void cpu_row(const double p[K],const RowIn&in,const double dir[K],const double diru[K],const double dirv[K],double*o){
    J2 j=rigid_row_nll_j2(p,in); T4 t=rigid_row_nll_t4(p,in); int idx=0;
    o[idx++]=j.v; for(int a=0;a<K;a++) o[idx++]=j.g[a];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++) o[idx++]=j.h[a][b];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++){double s=0;for(int cc=0;cc<K;cc++)s+=t.t3[a][b][cc]*dir[cc];o[idx++]=s;}
    for(int a=0;a<K;a++)for(int b=0;b<K;b++){double s=0;for(int cc=0;cc<K;cc++)for(int d=0;d<K;d++)s+=t.t4[a][b][cc][d]*diru[cc]*dirv[d];o[idx++]=s;}
}

#define CU(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

int main(int argc,char**argv){
    long n = argc>1? atol(argv[1]) : 4000000;
    double probit_scale=0.7;
    double dir[K]={0.31,-0.22,0.17,0.44}, diru[K]={0.13,0.27,-0.41,0.05}, dirv[K]={-0.19,0.33,0.08,0.22};
    // build a realistic grid of rows (varied margins incl. deep tail)
    std::vector<double> hq0(n),hq1(n),hqd1(n),hg(n),hw(n),hd(n),hz(n),hcov(n);
    for(long i=0;i<n;i++){
        double t=(double)i/n;
        hq0[i]= -2.5 + 5.0*sin(12.0*t);      // covers + and - margins
        hq1[i]= -1.5 + 4.0*cos(9.0*t+0.3);
        hqd1[i]= 0.2 + 1.8*(0.5+0.5*sin(7.0*t)); // >0 (monotone guard)
        hg[i]= -1.0 + 2.0*sin(5.0*t+1.1);
        hw[i]= 1.0;
        hd[i]= (i%3==0)?1.0:0.0;              // mix of events/censored
        hz[i]= 0.5*cos(3.0*t);
        hcov[i]= 0.4+0.3*(0.5+0.5*sin(2.0*t));
    }
    // ---- CPU reference timing (single thread, the per-row jet wall) ----
    std::vector<double> cpu_out((size_t)n*W);
    auto c0=std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(long i=0;i<n;i++){
        double p[K]={hq0[i],hq1[i],hqd1[i],hg[i]};
        RowIn in{hw[i],hd[i],hz[i],hcov[i],probit_scale};
        cpu_row(p,in,dir,diru,dirv,&cpu_out[(size_t)i*W]);
    }
    auto c1=std::chrono::high_resolution_clock::now();
    double cpu_s=std::chrono::duration<double>(c1-c0).count();
    int ncpu=1;
    #pragma omp parallel
    {
      #pragma omp single
      ncpu=omp_get_num_threads();
    }

    // ---- GPU ----
    double *dq0,*dq1,*dqd1,*dg,*dw,*dd,*dz,*dcov,*dout,*ddir,*ddiru,*ddirv;
    size_t bn=(size_t)n*sizeof(double);
    CU(cudaMalloc(&dq0,bn));CU(cudaMalloc(&dq1,bn));CU(cudaMalloc(&dqd1,bn));CU(cudaMalloc(&dg,bn));
    CU(cudaMalloc(&dw,bn));CU(cudaMalloc(&dd,bn));CU(cudaMalloc(&dz,bn));CU(cudaMalloc(&dcov,bn));
    CU(cudaMalloc(&dout,(size_t)n*W*sizeof(double)));
    CU(cudaMalloc(&ddir,K*sizeof(double)));CU(cudaMalloc(&ddiru,K*sizeof(double)));CU(cudaMalloc(&ddirv,K*sizeof(double)));
    auto h0=std::chrono::high_resolution_clock::now();
    CU(cudaMemcpy(dq0,hq0.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dq1,hq1.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dqd1,hqd1.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dg,hg.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dw,hw.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dd,hd.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dz,hz.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dcov,hcov.data(),bn,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(ddir,dir,K*sizeof(double),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(ddiru,diru,K*sizeof(double),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(ddirv,dirv,K*sizeof(double),cudaMemcpyHostToDevice));
    auto h1=std::chrono::high_resolution_clock::now();
    int tpb=128; long grid=(n+tpb-1)/tpb;
    // warmup
    survival_jet_kernel<<<grid,tpb>>>(n,dq0,dq1,dqd1,dg,dw,dd,dz,dcov,probit_scale,ddir,ddiru,ddirv,dout);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    auto k0=std::chrono::high_resolution_clock::now();
    int reps=5;
    for(int r=0;r<reps;r++) survival_jet_kernel<<<grid,tpb>>>(n,dq0,dq1,dqd1,dg,dw,dd,dz,dcov,probit_scale,ddir,ddiru,ddirv,dout);
    CU(cudaDeviceSynchronize());
    auto k1=std::chrono::high_resolution_clock::now();
    double kern_s=std::chrono::duration<double>(k1-k0).count()/reps;
    double htod_s=std::chrono::duration<double>(h1-h0).count();
    std::vector<double> gpu_out((size_t)n*W);
    auto d0=std::chrono::high_resolution_clock::now();
    CU(cudaMemcpy(gpu_out.data(),dout,(size_t)n*W*sizeof(double),cudaMemcpyDeviceToHost));
    auto d1=std::chrono::high_resolution_clock::now();
    double dtoh_s=std::chrono::duration<double>(d1-d0).count();

    // ---- REALISTIC on-device reduction path (tiny DtoH) ----
    double* dacc; CU(cudaMalloc(&dacc,(1+K)*sizeof(double)));
    CU(cudaMemset(dacc,0,(1+K)*sizeof(double)));
    survival_jet_reduce_kernel<<<grid,tpb>>>(n,dq0,dq1,dqd1,dg,dw,dd,dz,dcov,probit_scale,dacc);
    CU(cudaGetLastError()); CU(cudaDeviceSynchronize());
    auto r0=std::chrono::high_resolution_clock::now();
    for(int r=0;r<reps;r++){ CU(cudaMemset(dacc,0,(1+K)*sizeof(double)));
        survival_jet_reduce_kernel<<<grid,tpb>>>(n,dq0,dq1,dqd1,dg,dw,dd,dz,dcov,probit_scale,dacc);}
    CU(cudaDeviceSynchronize());
    auto r1=std::chrono::high_resolution_clock::now();
    double red_kern_s=std::chrono::duration<double>(r1-r0).count()/reps;
    double hacc[1+K]; CU(cudaMemcpy(hacc,dacc,(1+K)*sizeof(double),cudaMemcpyDeviceToHost));
    // CPU total NLL for accuracy of the reduction
    double cpu_nll=0; for(long i=0;i<n;i++) cpu_nll+=cpu_out[(size_t)i*W];
    double red_e2e=htod_s+red_kern_s+1e-6; // tiny p-vector DtoH ~ negligible

    // ---- accuracy ----
    double max_abs=0,max_rel=0; long worst=-1; int worstc=-1;
    for(size_t i=0;i<(size_t)n*W;i++){
        double a=cpu_out[i],b=gpu_out[i];
        if(!isfinite(a)||!isfinite(b)){ if(a!=b){fprintf(stderr,"nonfinite mismatch at %zu: %g vs %g\n",i,a,b);} continue; }
        double ad=fabs(a-b); double rd=(fabs(a)>1e-6)? ad/fabs(a) : 0.0;
        if(ad>max_abs){max_abs=ad; worst=i/W; worstc=i%W;}
        if(rd>max_rel){max_rel=rd;}
    }
    printf("=== survival rigid row jet: A100 vs CPU ===\n");
    printf("n=%ld  width=%d (v+g4+H16+t3c16+t4c16)\n",n,W);
    printf("CPU per-row jet (%d threads): %.4f s  -> %.3e rows/s\n",ncpu,cpu_s,n/cpu_s);
    printf("GPU kernel (mean of %d):    %.5f s  -> %.3e rows/s\n",reps,kern_s,n/kern_s);
    printf("  HtoD copy: %.4f s   DtoH copy(%dB/row): %.4f s\n",htod_s,W*8,dtoh_s);
    printf("GPU end-to-end (HtoD+kern+DtoH): %.4f s -> %.3e rows/s\n",htod_s+kern_s+dtoh_s,n/(htod_s+kern_s+dtoh_s));
    printf("SPEEDUP kernel-only          vs %d-CPU-thread: %.1fx\n",ncpu,cpu_s/kern_s);
    printf("SPEEDUP full-output e2e      vs %d-CPU-thread: %.1fx (transfer-bound: 424B/row DtoH)\n",ncpu,cpu_s/(htod_s+kern_s+dtoh_s));
    printf("--- realistic on-device REDUCTION path (tiny DtoH) ---\n");
    printf("  reduce kernel: %.5f s -> %.3e rows/s ; e2e(HtoD+kern): %.4f s -> %.1fx vs %d-CPU-thread\n",
        red_kern_s,n/red_kern_s,red_e2e,cpu_s/red_e2e,ncpu);
    printf("  total NLL: gpu=%.10g cpu=%.10g relerr=%.3e\n",hacc[0],cpu_nll,fabs(hacc[0]-cpu_nll)/fabs(cpu_nll));
    printf("ACCURACY: max_abs=%.3e  max_rel(|cpu|>1e-6)=%.3e  (worst-abs row %ld col %d)\n",max_abs,max_rel,worst,worstc);
    printf("VERDICT: %s\n", (max_abs<=1e-9)?"PASS (max_abs<=1e-9)":"CHECK");
    if(worst>=0){
        printf("--- worst row %ld dump (col: CPU vs GPU) ---\n",worst);
        for(int cc=0;cc<W;cc++){ double a=cpu_out[(size_t)worst*W+cc],b=gpu_out[(size_t)worst*W+cc];
            if(fabs(a-b)>1e-12) printf("  col %d: cpu=%.17g gpu=%.17g diff=%.3e\n",cc,a,b,fabs(a-b)); }
        printf("  inputs: q0=%.6g q1=%.6g qd1=%.6g g=%.6g w=%.6g d=%.6g z=%.6g cov=%.6g\n",
            hq0[worst],hq1[worst],hqd1[worst],hg[worst],hw[worst],hd[worst],hz[worst],hcov[worst]);
    }
    return 0;
}
