// NVRTC device source for the survival marginal-slope rigid per-row NLL jet.
// Compiled by `survival_rowjet.rs` via `cudarc::nvrtc::compile_ptx` (full f64,
// no fast-math). Byte-faithful port of the CPU `rigid_row_nll` seeded-jet
// arithmetic. K is fixed to 4 (rigid survival primaries q0,q1,qd1,g).
//
// Provenance: derived from the measured standalone prototype
// `src/gpu/proto/survival_marginal_slope_jet_932.cu` (device == CPU to 4.7e-12
// on the A100). Seeded jets (JS1/JS2) carry O(K^2) state — a dense Tower4<4>
// spills 41KB/thread and OOMs the launch.

#define K 4

// NVRTC does NOT include <math.h>/<cmath>, so the math.h constant macros
// (M_PI, M_SQRT2, ...) are undefined here — defining them explicitly (full f64
// values) is required or the NVRTC compile fails with "identifier M_PI is
// undefined" and the dispatcher silently falls back to the CPU. The intrinsics
// erfc / erfcx / exp / log / sqrt / isfinite / isnan / fmin / fmax and the
// constants INFINITY / NAN ARE provided by NVRTC.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
// NVRTC also leaves INFINITY / NAN undefined (they come from <math.h>). Define
// them as the exact IEEE-754 f64 bit patterns via the NVRTC-provided bit-cast
// intrinsic: +inf = 0x7ff0…0, quiet NaN = 0x7ff8…0.
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))
#endif
#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000LL))
#endif

// ─── statrs Boost-derived erf polynomial, ported verbatim ───────────────
// These coefficient tables and the erf_impl control flow are a byte-faithful
// port of statrs 0.18 `function::erf` (the CPU oracle's erfc). The device
// must call THIS, not CUDA's intrinsic erfc: the two rational approximations
// differ ~1e-10 relative, and the deep probit-tail derivative jet amplifies
// that to ~5e-8 — blowing the 1e-9 GPU↔CPU parity gate (measured on a V100).
// Horner evaluation with --fmad=false reproduces the CPU rounding exactly.
__constant__ double EI_AN[8] = { 0.00337916709551257388990745, -0.00073695653048167948530905, -0.374732337392919607868241, 0.0817442448733587196071743, -0.0421089319936548595203468, 0.0070165709512095756344528, -0.00495091255982435110337458, 0.000871646599037922480317225 };
__constant__ double EI_AD[8] = { 1.0, -0.218088218087924645390535, 0.412542972725442099083918, -0.0841891147873106755410271, 0.0655338856400241519690695, -0.0120019604454941768171266, 0.00408165558926174048329689, -0.000615900721557769691924509 };
__constant__ double EI_BN[6] = { -0.0361790390718262471360258, 0.292251883444882683221149, 0.281447041797604512774415, 0.125610208862766947294894, 0.0274135028268930549240776, 0.00250839672168065762786937 };
__constant__ double EI_BD[6] = { 1.0, 1.8545005897903486499845, 1.43575803037831418074962, 0.582827658753036572454135, 0.124810476932949746447682, 0.0113724176546353285778481 };
__constant__ double EI_CN[7] = { -0.0397876892611136856954425, 0.153165212467878293257683, 0.191260295600936245503129, 0.10276327061989304213645, 0.029637090615738836726027, 0.0046093486780275489468812, 0.000307607820348680180548455 };
__constant__ double EI_CD[7] = { 1.0, 1.95520072987627704987886, 1.64762317199384860109595, 0.768238607022126250082483, 0.209793185936509782784315, 0.0319569316899913392596356, 0.00213363160895785378615014 };
__constant__ double EI_DN[7] = { -0.0300838560557949717328341, 0.0538578829844454508530552, 0.0726211541651914182692959, 0.0367628469888049348429018, 0.00964629015572527529605267, 0.00133453480075291076745275, 0.778087599782504251917881e-4 };
__constant__ double EI_DD[8] = { 1.0, 1.75967098147167528287343, 1.32883571437961120556307, 0.552528596508757581287907, 0.133793056941332861912279, 0.0179509645176280768640766, 0.00104712440019937356634038, -0.106640381820357337177643e-7 };
__constant__ double EI_EN[7] = { -0.0117907570137227847827732, 0.014262132090538809896674, 0.0202234435902960820020765, 0.00930668299990432009042239, 0.00213357802422065994322516, 0.00025022987386460102395382, 0.120534912219588189822126e-4 };
__constant__ double EI_ED[7] = { 1.0, 1.50376225203620482047419, 0.965397786204462896346934, 0.339265230476796681555511, 0.0689740649541569716897427, 0.00771060262491768307365526, 0.000371421101531069302990367 };
__constant__ double EI_FN[7] = { -0.00546954795538729307482955, 0.00404190278731707110245394, 0.0054963369553161170521356, 0.00212616472603945399437862, 0.000394984014495083900689956, 0.365565477064442377259271e-4, 0.135485897109932323253786e-5 };
__constant__ double EI_FD[8] = { 1.0, 1.21019697773630784832251, 0.620914668221143886601045, 0.173038430661142762569515, 0.0276550813773432047594539, 0.00240625974424309709745382, 0.891811817251336577241006e-4, -0.465528836283382684461025e-11 };
__constant__ double EI_GN[6] = { -0.00270722535905778347999196, 0.0013187563425029400461378, 0.00119925933261002333923989, 0.00027849619811344664248235, 0.267822988218331849989363e-4, 0.923043672315028197865066e-6 };
__constant__ double EI_GD[7] = { 1.0, 0.814632808543141591118279, 0.268901665856299542168425, 0.0449877216103041118694989, 0.00381759663320248459168994, 0.000131571897888596914350697, 0.404815359675764138445257e-11 };
__constant__ double EI_HN[6] = { -0.00109946720691742196814323, 0.000406425442750422675169153, 0.000274499489416900707787024, 0.465293770646659383436343e-4, 0.320955425395767463401993e-5, 0.778286018145020892261936e-7 };
__constant__ double EI_HD[6] = { 1.0, 0.588173710611846046373373, 0.139363331289409746077541, 0.0166329340417083678763028, 0.00100023921310234908642639, 0.24254837521587225125068e-4 };
__constant__ double EI_IN[5] = { -0.00056907993601094962855594, 0.000169498540373762264416984, 0.518472354581100890120501e-4, 0.382819312231928859704678e-5, 0.824989931281894431781794e-7 };
__constant__ double EI_ID[6] = { 1.0, 0.339637250051139347430323, 0.043472647870310663055044, 0.00248549335224637114641629, 0.535633305337152900549536e-4, -0.117490944405459578783846e-12 };
__constant__ double EI_JN[5] = { -0.000241313599483991337479091, 0.574224975202501512365975e-4, 0.115998962927383778460557e-4, 0.581762134402593739370875e-6, 0.853971555085673614607418e-8 };
__constant__ double EI_JD[5] = { 1.0, 0.233044138299687841018015, 0.0204186940546440312625597, 0.000797185647564398289151125, 0.117019281670172327758019e-4 };
__constant__ double EI_KN[5] = { -0.000146674699277760365803642, 0.162666552112280519955647e-4, 0.269116248509165239294897e-5, 0.979584479468091935086972e-7, 0.101994647625723465722285e-8 };
__constant__ double EI_KD[5] = { 1.0, 0.165907812944847226546036, 0.0103361716191505884359634, 0.000286593026373868366935721, 0.298401570840900340874568e-5 };
__constant__ double EI_LN[5] = { -0.583905797629771786720406e-4, 0.412510325105496173512992e-5, 0.431790922420250949096906e-6, 0.993365155590013193345569e-8, 0.653480510020104699270084e-10 };
__constant__ double EI_LD[5] = { 1.0, 0.105077086072039915406159, 0.00414278428675475620830226, 0.726338754644523769144108e-4, 0.477818471047398785369849e-6 };
__constant__ double EI_MN[4] = { -0.196457797609229579459841e-4, 0.157243887666800692441195e-5, 0.543902511192700878690335e-7, 0.317472492369117710852685e-9 };
__constant__ double EI_MD[5] = { 1.0, 0.052803989240957632204885, 0.000926876069151753290378112, 0.541011723226630257077328e-5, 0.535093845803642394908747e-15 };
__constant__ double EI_NN[4] = { -0.789224703978722689089794e-5, 0.622088451660986955124162e-6, 0.145728445676882396797184e-7, 0.603715505542715364529243e-10 };
__constant__ double EI_ND[4] = { 1.0, 0.0375328846356293715248719, 0.000467919535974625308126054, 0.193847039275845656900547e-5 };

// Horner polynomial evaluation, identical order to statrs `evaluate::polynomial`:
// sum = coeff[n-1]; for c in coeff[0..n-1] reversed: sum = c + z*sum.
__device__ __forceinline__ double ei_poly(double z, const double* c, int n){
    if(n==0) return 0.0;
    double sum=c[n-1];
    for(int i=n-2;i>=0;--i) sum = c[i] + z*sum;
    return sum;
}

// Byte-faithful port of statrs 0.18 `erf_impl(z, inv)` for z >= 0. The negative
// branch of statrs's `erf_impl` is a pure sign/reflection of the z>=0 result, so
// it is handled inline by the callers (`erf_statrs`/`erfc_statrs`) WITHOUT device
// recursion: recursion under this deep-jet kernel (heavy per-thread stack +
// `__launch_bounds__(128,1)`) overflows the launch's local-memory stack and
// faults with CUDA_ERROR_ILLEGAL_ADDRESS. This formulation is recursion-free.
__device__ double erf_impl_pos(double z, bool inv){
    // z is guaranteed >= 0 here (callers reflect negatives).
    double result;
    if(z<0.5){
        if(z<1e-10){
            result = z*1.125 + z*0.003379167095512573896158903121545171688;
        }else{
            result = z*1.125 + z*ei_poly(z,EI_AN,8)/ei_poly(z,EI_AD,8);
        }
    }else if(z<110.0){
        double r,b;
        if(z<0.75){ r=ei_poly(z-0.5,EI_BN,6)/ei_poly(z-0.5,EI_BD,6); b=0.3440242112; }
        else if(z<1.25){ r=ei_poly(z-0.75,EI_CN,7)/ei_poly(z-0.75,EI_CD,7); b=0.419990927; }
        else if(z<2.25){ r=ei_poly(z-1.25,EI_DN,7)/ei_poly(z-1.25,EI_DD,8); b=0.4898625016; }
        else if(z<3.5){ r=ei_poly(z-2.25,EI_EN,7)/ei_poly(z-2.25,EI_ED,7); b=0.5317370892; }
        else if(z<5.25){ r=ei_poly(z-3.5,EI_FN,7)/ei_poly(z-3.5,EI_FD,8); b=0.5489973426; }
        else if(z<8.0){ r=ei_poly(z-5.25,EI_GN,6)/ei_poly(z-5.25,EI_GD,7); b=0.5571740866; }
        else if(z<11.5){ r=ei_poly(z-8.0,EI_HN,6)/ei_poly(z-8.0,EI_HD,6); b=0.5609807968; }
        else if(z<17.0){ r=ei_poly(z-11.5,EI_IN,5)/ei_poly(z-11.5,EI_ID,6); b=0.5626493692; }
        else if(z<24.0){ r=ei_poly(z-17.0,EI_JN,5)/ei_poly(z-17.0,EI_JD,5); b=0.5634598136; }
        else if(z<38.0){ r=ei_poly(z-24.0,EI_KN,5)/ei_poly(z-24.0,EI_KD,5); b=0.5638477802; }
        else if(z<60.0){ r=ei_poly(z-38.0,EI_LN,5)/ei_poly(z-38.0,EI_LD,5); b=0.5640528202; }
        else if(z<85.0){ r=ei_poly(z-60.0,EI_MN,4)/ei_poly(z-60.0,EI_MD,5); b=0.5641309023; }
        else { r=ei_poly(z-85.0,EI_NN,4)/ei_poly(z-85.0,EI_ND,4); b=0.5641584396; }
        double g=exp(-z*z)/z;
        result = g*b + g*r;
    }else{
        result = 0.0;
    }
    if(inv && z>=0.5) return result;
    else if(z>=0.5 || inv) return 1.0 - result;
    else return result;
}

// statrs `erfc` wrapper. NaN/Inf branches mirror statrs's `erfc` exactly; the
// finite path reproduces statrs `erf_impl(x, inv=true)` including its z<0
// reflection, but inlined (no device recursion). For x<0:
//   z<-0.5 : erfc(x) = 2 - erfc(-x)        = 2 - erf_impl_pos(-x, true)
//   -0.5<=x<0 : erfc(x) = 1 + erf(-x)      = 1 + erf_impl_pos(-x, false)
__device__ __forceinline__ double erfc_statrs(double x){
    if(isnan(x)) return NAN;
    if(x==INFINITY) return 0.0;
    if(x==-INFINITY) return 2.0;
    if(x>=0.0) return erf_impl_pos(x,true);
    if(x<-0.5) return 2.0 - erf_impl_pos(-x,true);
    return 1.0 + erf_impl_pos(-x,false);
}

// ---- transcendentals, bit-mirroring the Rust f64 ops ----
// erfcx_nonnegative (src/inference/probability.rs)
__device__ __forceinline__ double erfcx_nn(double x){
    if(!isfinite(x)) return x>0.0 ? 0.0 : INFINITY;
    if(x<=0.0) return 1.0;
    if(x<26.0) return exp(fmin(x*x,700.0))*erfc_statrs(x);
    double inv=1.0/x, inv2=inv*inv;
    double poly=1.0-0.5*inv2+0.75*inv2*inv2-1.875*inv2*inv2*inv2+6.5625*inv2*inv2*inv2*inv2;
    return inv*poly/sqrt(M_PI);
}
__device__ __forceinline__ double normal_pdf(double x){
    const double INV_SQRT_2PI=0.3989422804014327;
    return INV_SQRT_2PI*exp(-0.5*x*x);
}
__device__ __forceinline__ double normal_cdf(double x){ return 0.5*erfc_statrs(-x/M_SQRT2); }
__device__ __forceinline__ void sp_logcdf_mills(double x, double*lc, double*lam){
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
// signed_probit_neglog_unary_stack: [-w logPhi(m), w k1..k4]
__device__ __forceinline__ void neglog_phi_stack(double m,double w,double out[5]){
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
__device__ __forceinline__ void d_sqrt(double x,double o[5]){
    double xa=fmax(x,1e-300); double s=sqrt(xa);
    double x2=xa*xa, x3=x2*xa;
    o[0]=s; o[1]=0.5/s; o[2]=-0.25/(xa*s); o[3]=3.0/(8.0*x2*s); o[4]=-15.0/(16.0*x3*s);
}
__device__ __forceinline__ void d_log(double x,double o[5]){
    double x2=x*x,x3=x2*x,x4=x3*x;
    o[0]=log(x); o[1]=1.0/x; o[2]=-1.0/x2; o[3]=2.0/x3; o[4]=-6.0/x4;
}
__device__ __forceinline__ void d_lognormpdf(double x,double o[5]){
    double c=0.5*log(2.0*M_PI);
    o[0]=-0.5*x*x-c; o[1]=-x; o[2]=-1.0; o[3]=0.0; o[4]=0.0;
}

struct RowIn{ double wi,di,z_sum,cov_ones,probit_scale; };

// ---- Order-2 jet: exactly value + gradient + Hessian, no seeded high-order state ----
struct J2{ double v; double g[K]; double h[K][K]; };
__device__ __forceinline__ J2 j2_const(double c){
    J2 r; r.v=c;
    for(int i=0;i<K;i++){ r.g[i]=0.0; for(int j=0;j<K;j++) r.h[i][j]=0.0; }
    return r;
}
__device__ __forceinline__ J2 j2_var(double x,int a){ J2 r=j2_const(x); r.g[a]=1.0; return r; }
__device__ __forceinline__ J2 j2_scale(const J2&a,double s){
    J2 r; r.v=a.v*s;
    for(int i=0;i<K;i++){ r.g[i]=a.g[i]*s; for(int j=0;j<K;j++) r.h[i][j]=a.h[i][j]*s; }
    return r;
}
__device__ __forceinline__ J2 j2_add(const J2&a,const J2&b){
    J2 r; r.v=a.v+b.v;
    for(int i=0;i<K;i++){ r.g[i]=a.g[i]+b.g[i]; for(int j=0;j<K;j++) r.h[i][j]=a.h[i][j]+b.h[i][j]; }
    return r;
}
__device__ __forceinline__ J2 j2_addc(const J2&a,double c){ J2 r=a; r.v+=c; return r; }
__device__ __forceinline__ J2 j2_mul(const J2&a,const J2&b){
    J2 r=j2_const(a.v*b.v);
    for(int i=0;i<K;i++) r.g[i]=a.v*b.g[i]+a.g[i]*b.v;
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)
        r.h[i][j]=a.v*b.h[i][j]+a.g[i]*b.g[j]+a.g[j]*b.g[i]+a.h[i][j]*b.v;
    return r;
}
__device__ __forceinline__ J2 j2_compose(const J2&a,const double f[5]){
    double f1=f[1],f2=f[2]; J2 r=j2_const(f[0]);
    for(int i=0;i<K;i++) r.g[i]=f1*a.g[i];
    for(int i=0;i<K;i++)for(int j=0;j<K;j++)
        r.h[i][j]=f1*a.h[i][j]+f2*a.g[i]*a.g[j];
    return r;
}

// ---- OneSeed jet: value+grad+hess + eps-derivatives (eps seeds x_a += eps*dir[a]) ----
// dh = the eps-Hessian channel = sum_c t3[a][b][c] dir[c].
struct JS1{ double v; double g[K]; double h[K][K]; double dv; double dg[K]; double dh[K][K]; };
__device__ __forceinline__ JS1 js1_const(double c){JS1 r;r.v=c;r.dv=0;for(int i=0;i<K;i++){r.g[i]=0;r.dg[i]=0;for(int j=0;j<K;j++){r.h[i][j]=0;r.dh[i][j]=0;}}return r;}
__device__ __forceinline__ JS1 js1_var(double x,int a,double dir){JS1 r=js1_const(x);r.g[a]=1.0;r.dv=dir;return r;}
__device__ __forceinline__ JS1 js1_scale(const JS1&a,double s){JS1 r;r.v=a.v*s;r.dv=a.dv*s;for(int i=0;i<K;i++){r.g[i]=a.g[i]*s;r.dg[i]=a.dg[i]*s;for(int j=0;j<K;j++){r.h[i][j]=a.h[i][j]*s;r.dh[i][j]=a.dh[i][j]*s;}}return r;}
__device__ __forceinline__ JS1 js1_add(const JS1&a,const JS1&b){JS1 r;r.v=a.v+b.v;r.dv=a.dv+b.dv;for(int i=0;i<K;i++){r.g[i]=a.g[i]+b.g[i];r.dg[i]=a.dg[i]+b.dg[i];for(int j=0;j<K;j++){r.h[i][j]=a.h[i][j]+b.h[i][j];r.dh[i][j]=a.dh[i][j]+b.dh[i][j];}}return r;}
__device__ __forceinline__ JS1 js1_addc(const JS1&a,double c){JS1 r=a;r.v+=c;return r;}
__device__ __forceinline__ JS1 js1_mul(const JS1&a,const JS1&b){
    JS1 r=js1_const(a.v*b.v); r.dv=a.dv*b.v+a.v*b.dv;
    for(int i=0;i<K;i++){ r.g[i]=a.v*b.g[i]+a.g[i]*b.v; r.dg[i]=a.dv*b.g[i]+a.v*b.dg[i]+a.dg[i]*b.v+a.g[i]*b.dv; }
    for(int i=0;i<K;i++)for(int j=0;j<K;j++){
        r.h[i][j]=a.v*b.h[i][j]+a.g[i]*b.g[j]+a.g[j]*b.g[i]+a.h[i][j]*b.v;
        r.dh[i][j]=a.dv*b.h[i][j]+a.v*b.dh[i][j]+a.dg[i]*b.g[j]+a.g[i]*b.dg[j]
                  +a.dg[j]*b.g[i]+a.g[j]*b.dg[i]+a.dh[i][j]*b.v+a.h[i][j]*b.dv;
    }
    return r;
}
__device__ __forceinline__ JS1 js1_compose(const JS1&a,const double f[5]){
    double f1=f[1],f2=f[2],f3=f[3];
    JS1 r=js1_const(f[0]); r.dv=f1*a.dv;
    for(int i=0;i<K;i++){ r.g[i]=f1*a.g[i]; r.dg[i]=f1*a.dg[i]+f2*a.dv*a.g[i]; }
    for(int i=0;i<K;i++)for(int j=0;j<K;j++){
        r.h[i][j]=f1*a.h[i][j]+f2*a.g[i]*a.g[j];
        r.dh[i][j]=f1*a.dh[i][j]+f2*a.dv*a.h[i][j]
                  +f2*(a.dg[i]*a.g[j]+a.g[i]*a.dg[j])+f3*a.dv*a.g[i]*a.g[j];
    }
    return r;
}

// ---- TwoSeed jet: value+grad+hess + u-deriv + v-deriv + mixed uv-deriv ----
// huv = eps-delta-Hessian = sum_{c,d} t4[a][b][c][d] u[c] v[d].
struct JS2{ double v; double g[K]; double h[K][K];
    double eu,ev_; double gu[K],gv[K]; double hu[K][K],hv[K][K];
    double euv; double guv[K]; double huv[K][K]; };
__device__ __forceinline__ JS2 js2_const(double c){JS2 r;r.v=c;r.eu=r.ev_=r.euv=0;
    for(int i=0;i<K;i++){r.g[i]=0;r.gu[i]=0;r.gv[i]=0;r.guv[i]=0;for(int j=0;j<K;j++){r.h[i][j]=0;r.hu[i][j]=0;r.hv[i][j]=0;r.huv[i][j]=0;}}return r;}
__device__ __forceinline__ JS2 js2_var(double x,int a,double du,double dv){JS2 r=js2_const(x);r.g[a]=1.0;r.eu=du;r.ev_=dv;return r;}
__device__ __forceinline__ JS2 js2_scale(const JS2&a,double s){JS2 r=a;
    r.v*=s;r.eu*=s;r.ev_*=s;r.euv*=s; for(int i=0;i<K;i++){r.g[i]*=s;r.gu[i]*=s;r.gv[i]*=s;r.guv[i]*=s;for(int j=0;j<K;j++){r.h[i][j]*=s;r.hu[i][j]*=s;r.hv[i][j]*=s;r.huv[i][j]*=s;}}return r;}
__device__ __forceinline__ JS2 js2_add(const JS2&a,const JS2&b){JS2 r;
    r.v=a.v+b.v;r.eu=a.eu+b.eu;r.ev_=a.ev_+b.ev_;r.euv=a.euv+b.euv;
    for(int i=0;i<K;i++){r.g[i]=a.g[i]+b.g[i];r.gu[i]=a.gu[i]+b.gu[i];r.gv[i]=a.gv[i]+b.gv[i];r.guv[i]=a.guv[i]+b.guv[i];
        for(int j=0;j<K;j++){r.h[i][j]=a.h[i][j]+b.h[i][j];r.hu[i][j]=a.hu[i][j]+b.hu[i][j];r.hv[i][j]=a.hv[i][j]+b.hv[i][j];r.huv[i][j]=a.huv[i][j]+b.huv[i][j];}}return r;}
__device__ __forceinline__ JS2 js2_addc(const JS2&a,double c){JS2 r=a;r.v+=c;return r;}
__device__ __forceinline__ JS2 js2_mul(const JS2&A,const JS2&B){
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
        r.huv[i][j]=A.euv*B.h[i][j]+A.eu*B.hv[i][j]+A.ev_*B.hu[i][j]+A.v*B.huv[i][j]
                  +A.guv[i]*B.g[j]+A.gu[i]*B.gv[j]+A.gv[i]*B.gu[j]+A.g[i]*B.guv[j]
                  +A.guv[j]*B.g[i]+A.gu[j]*B.gv[i]+A.gv[j]*B.gu[i]+A.g[j]*B.guv[i]
                  +A.huv[i][j]*B.v+A.hu[i][j]*B.ev_+A.hv[i][j]*B.eu+A.h[i][j]*B.euv;
    }
    return r;
}
__device__ __forceinline__ JS2 js2_compose(const JS2&a,const double f[5]){
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
        double t=0;
        t+= f1*a.huv[i][j] + f2*a.eu*a.hv[i][j] + f2*a.ev_*a.hu[i][j]
           + (f3*a.eu*a.ev_+f2*a.euv)*a.h[i][j];
        double dgg_u=a.gu[i]*a.g[j]+a.g[i]*a.gu[j];
        double dgg_v=a.gv[i]*a.g[j]+a.g[i]*a.gv[j];
        double dgg_uv=a.guv[i]*a.g[j]+a.gu[i]*a.gv[j]+a.gv[i]*a.gu[j]+a.g[i]*a.guv[j];
        t+= f2*dgg_uv + f3*a.eu*dgg_v + f3*a.ev_*dgg_u + (f4*a.eu*a.ev_+f3*a.euv)*gigj;
        r.huv[i][j]=t;
    }
    return r;
}

// The unified NLL program, written ONCE per scalar type via this macro.
#define DEF_NLL(NAME,T,CONST,SCALE,MUL,ADD,ADDC,COMPOSE)                       \
__device__ __forceinline__ T NAME(T q0,T q1,T qd1,T g,const RowIn&in){         \
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
DEF_NLL(nll_j2,J2,j2_const,j2_scale,j2_mul,j2_add,j2_addc,j2_compose)
DEF_NLL(nll_js1,JS1,js1_const,js1_scale,js1_mul,js1_add,js1_addc,js1_compose)
DEF_NLL(nll_js2,JS2,js2_const,js2_scale,js2_mul,js2_add,js2_addc,js2_compose)

extern "C" __global__ void __launch_bounds__(128,1) survival_rowjet_vgh(
        int n,
        const double* __restrict__ q0,const double* __restrict__ q1,
        const double* __restrict__ qd1,const double* __restrict__ gg,
        const double* __restrict__ wi,const double* __restrict__ di,
        const double* __restrict__ zsum,const double* __restrict__ cov,
        double probit_scale,
        double* __restrict__ out_v, double* __restrict__ out_g,
        double* __restrict__ out_h){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    RowIn in; in.wi=wi[i]; in.di=di[i]; in.z_sum=zsum[i]; in.cov_ones=cov[i]; in.probit_scale=probit_scale;
    J2 out=nll_j2(j2_var(q0[i],0),j2_var(q1[i],1),j2_var(qd1[i],2),j2_var(gg[i],3),in);
    out_v[i]=out.v;
    for(int a=0;a<K;a++) out_g[(size_t)i*K+a]=out.g[a];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++)
        out_h[(size_t)i*K*K+a*K+b]=out.h[a][b];
}

extern "C" __global__ void __launch_bounds__(128,1) survival_rowjet(
        int n,
        const double* __restrict__ q0,const double* __restrict__ q1,
        const double* __restrict__ qd1,const double* __restrict__ gg,
        const double* __restrict__ wi,const double* __restrict__ di,
        const double* __restrict__ zsum,const double* __restrict__ cov,
        double probit_scale,
        const double* __restrict__ dir,const double* __restrict__ diru,const double* __restrict__ dirv,
        double* __restrict__ out_v, double* __restrict__ out_g,
        double* __restrict__ out_h, double* __restrict__ out_t3, double* __restrict__ out_t4){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    RowIn in; in.wi=wi[i]; in.di=di[i]; in.z_sum=zsum[i]; in.cov_ones=cov[i]; in.probit_scale=probit_scale;
    double q0v=q0[i],q1v=q1[i],qd1v=qd1[i],gv=gg[i];
    // Contracted third. The JS1 base channels are the same value/gradient/Hessian
    // program the removed J2 path computed, so they also feed out_v/out_g/out_h.
    JS1 j1=nll_js1(js1_var(q0v,0,dir[0]),js1_var(q1v,1,dir[1]),js1_var(qd1v,2,dir[2]),js1_var(gv,3,dir[3]),in);
    // contracted fourth
    JS2 j2=nll_js2(js2_var(q0v,0,diru[0],dirv[0]),js2_var(q1v,1,diru[1],dirv[1]),
                   js2_var(qd1v,2,diru[2],dirv[2]),js2_var(gv,3,diru[3],dirv[3]),in);
    out_v[i]=j1.v;
    for(int a=0;a<K;a++) out_g[(size_t)i*K+a]=j1.g[a];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++){ size_t o=(size_t)i*K*K+a*K+b;
        out_h[o]=j1.h[a][b]; out_t3[o]=j1.dh[a][b]; out_t4[o]=j2.huv[a][b]; }
}

extern "C" __global__ void __launch_bounds__(128,1) survival_rowjet_no_t4(
        int n,
        const double* __restrict__ q0,const double* __restrict__ q1,
        const double* __restrict__ qd1,const double* __restrict__ gg,
        const double* __restrict__ wi,const double* __restrict__ di,
        const double* __restrict__ zsum,const double* __restrict__ cov,
        double probit_scale,
        const double* __restrict__ dir,
        double* __restrict__ out_v, double* __restrict__ out_g,
        double* __restrict__ out_h, double* __restrict__ out_t3, double* __restrict__ out_t4){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    RowIn in; in.wi=wi[i]; in.di=di[i]; in.z_sum=zsum[i]; in.cov_ones=cov[i]; in.probit_scale=probit_scale;
    double q0v=q0[i],q1v=q1[i],qd1v=qd1[i],gv=gg[i];
    JS1 j1=nll_js1(js1_var(q0v,0,dir[0]),js1_var(q1v,1,dir[1]),js1_var(qd1v,2,dir[2]),js1_var(gv,3,dir[3]),in);
    out_v[i]=j1.v;
    for(int a=0;a<K;a++) out_g[(size_t)i*K+a]=j1.g[a];
    for(int a=0;a<K;a++)for(int b=0;b<K;b++){ size_t o=(size_t)i*K*K+a*K+b;
        out_h[o]=j1.h[a][b]; out_t3[o]=j1.dh[a][b]; out_t4[o]=0.0; }
}
