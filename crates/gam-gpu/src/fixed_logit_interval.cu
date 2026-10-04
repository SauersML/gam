// Optional checked fixed-input scalar/KL pilot. No vendor exp/log, no acceptance.
// Each basic arithmetic operation below uses an explicit directed f64 intrinsic.
#define CI_BLOCK 256
#define CI_INF __longlong_as_double(0x7ff0000000000000LL)
#define CI_NINF __longlong_as_double(0xfff0000000000000LL)
typedef unsigned long long ci_u64;
struct CI { double lo, hi; unsigned int reason; };
__device__ CI ci_point(double x) { return CI{x,x,0}; }
__device__ CI ci_unknown(unsigned int reason) { return CI{0.0,0.0,reason}; }
__device__ CI ci_checked(double lo, double hi) {
    return (isnan(lo) || isnan(hi) || lo > hi) ? ci_unknown(4) : CI{lo,hi,0};
}
__device__ unsigned int ci_reason(CI a, CI b) { return a.reason > b.reason ? a.reason : b.reason; }
__device__ CI ci_add(CI a, CI b) {
    if (a.reason || b.reason) return ci_unknown(ci_reason(a,b));
    return ci_checked(__dadd_rd(a.lo,b.lo),__dadd_ru(a.hi,b.hi));
}
__device__ CI ci_sub(CI a, CI b) {
    if (a.reason || b.reason) return ci_unknown(ci_reason(a,b));
    return ci_checked(__dadd_rd(a.lo,-b.hi),__dadd_ru(a.hi,-b.lo));
}
__device__ double ci_product(double a,double b,bool upper) {
    // Zero times an unbounded endpoint is the limiting zero of this real
    // interval product; other endpoint pairs still retain unbounded ranges.
    if (a == 0.0 || b == 0.0) return 0.0;
    return upper ? __dmul_ru(a,b) : __dmul_rd(a,b);
}
__device__ CI ci_mul(CI a,CI b) {
    if (a.reason || b.reason) return ci_unknown(ci_reason(a,b));
    double lo=fmin(fmin(ci_product(a.lo,b.lo,false),ci_product(a.lo,b.hi,false)),
                   fmin(ci_product(a.hi,b.lo,false),ci_product(a.hi,b.hi,false)));
    double hi=fmax(fmax(ci_product(a.lo,b.lo,true),ci_product(a.lo,b.hi,true)),
                   fmax(ci_product(a.hi,b.lo,true),ci_product(a.hi,b.hi,true)));
    return ci_checked(lo,hi);
}
__device__ CI ci_div_positive(CI a,CI b) {
    if (a.reason || b.reason) return ci_unknown(ci_reason(a,b));
    if (!(b.lo > 0.0)) return ci_unknown(2);
    return ci_mul(a,ci_checked(__ddiv_rd(1.0,b.hi),__ddiv_ru(1.0,b.lo)));
}
__device__ CI ci_ln2() {
    // Adjacent dyadic endpoints certified by independent exact rational tests.
    return CI{__longlong_as_double(0x3fe62e42fefa39efLL),
              __longlong_as_double(0x3fe62e42fefa39f0LL),0};
}
__device__ double ci_abs(CI a) { return fmax(fabs(a.lo),fabs(a.hi)); }
__device__ double ci_power_two(int exponent) {
    return exponent < -1022 ? __longlong_as_double((ci_u64)1 << (exponent+1074))
                            : __longlong_as_double((ci_u64)(exponent+1023) << 52);
}
__device__ bool ci_arithmetic_guard() {
    volatile double tiny=__longlong_as_double(1LL);
    volatile double half=0.5;
    volatile double one=1.0;
    volatile double half_ulp=__longlong_as_double(0x3ca0000000000000LL);
    // A failed directed/gradual-underflow guard refuses the row. It never
    // selects a CPU fallback or substitutes an empirical epsilon.
    return __dmul_rd(tiny,one)==tiny && __dmul_ru(tiny,one)==tiny
        && __dmul_rd(tiny,half)==0.0 && __dmul_ru(tiny,half)==tiny
        && __ddiv_rd(tiny,2.0)==0.0 && __ddiv_ru(tiny,2.0)==tiny
        && __dadd_rd(one,half_ulp)==one
        && __dadd_ru(one,half_ulp)==__longlong_as_double(0x3ff0000000000001LL);
}
__device__ CI ci_exp(double x) {
    if (!isfinite(x)) return ci_unknown(1);
    if (x==0.0) return ci_point(1.0);
    CI cutoff=ci_mul(ci_ln2(),ci_point(-1074.0));
    if (x<=cutoff.lo) return CI{0.0,__longlong_as_double(1LL),0};
    // Refuse far positive input before converting a potentially huge quotient
    // to an integer. The input itself is never clamped.
    if (x>710.0) return ci_unknown(3);
    int k=__double2int_rn(__ddiv_rn(x,ci_ln2().lo));
    k=k < -1074 ? -1074 : (k > 1023 ? 1023 : k);
    CI r=ci_sub(ci_point(x),ci_mul(ci_ln2(),ci_point((double)k)));
    double radius=ci_abs(r);
    if (r.reason || !isfinite(radius) || radius>1.0) return ci_unknown(3);
    CI term=ci_point(1.0),sum=term;
    for (unsigned int degree=1;degree<=18;++degree) {
        term=ci_div_positive(ci_mul(term,r),ci_point((double)degree));
        sum=ci_add(sum,term);
    }
    CI first=ci_div_positive(ci_mul(ci_point(ci_abs(term)),ci_point(radius)),ci_point(19.0));
    CI ratio=ci_div_positive(ci_point(radius),ci_point(20.0));
    CI tail=ci_div_positive(first,ci_sub(ci_point(1.0),ratio));
    if (term.reason || sum.reason || tail.reason || !isfinite(tail.hi)) return ci_unknown(4);
    CI polynomial=ci_add(sum,CI{-tail.hi,tail.hi,0});
    CI value=ci_mul(polynomial,ci_point(ci_power_two(k)));
    value.lo=fmax(value.lo,0.0); // proven mathematical exp range
    if (value.reason || !isfinite(value.lo) || !isfinite(value.hi)) return ci_unknown(4);
    return value;
}
__device__ CI ci_log(double x) {
    if (!isfinite(x)) return ci_unknown(1);
    if (!(x>0.0)) return ci_unknown(2);
    if (x==1.0) return ci_point(0.0);
    ci_u64 bits=__double_as_longlong(x),fraction=bits&(((ci_u64)1<<52)-1);
    int exponent_bits=(int)((bits>>52)&2047),exponent;
    ci_u64 mantissa_bits;
    if (exponent_bits==0) {
        int highest=63-__clzll(fraction);
        ci_u64 normalized=fraction<<(52-highest);
        mantissa_bits=((ci_u64)1023<<52)|(normalized-((ci_u64)1<<52));
        exponent=highest-1074;
    } else {
        mantissa_bits=((ci_u64)1023<<52)|fraction;
        exponent=exponent_bits-1023;
    }
    CI m=ci_point(__longlong_as_double(mantissa_bits));
    CI u=ci_div_positive(ci_sub(m,ci_point(1.0)),ci_add(m,ci_point(1.0)));
    CI u2=ci_mul(u,u),power=u,sum=u;
    for (unsigned int j=1;j<18;++j) {
        power=ci_mul(power,u2);
        sum=ci_add(sum,ci_div_positive(power,ci_point((double)(2*j+1))));
    }
    CI omitted=ci_mul(power,u2);
    double radius=ci_abs(u);
    CI denom=ci_mul(ci_point(37.0),ci_sub(ci_point(1.0),ci_mul(ci_point(radius),ci_point(radius))));
    CI tail=ci_div_positive(ci_mul(ci_point(2.0),ci_point(ci_abs(omitted))),denom);
    if (u.reason || tail.reason || !isfinite(tail.hi)) return ci_unknown(4);
    CI mantissa_log=ci_add(ci_mul(sum,ci_point(2.0)),CI{-tail.hi,tail.hi,0});
    CI result=ci_add(mantissa_log,ci_mul(ci_ln2(),ci_point((double)exponent)));
    if (result.reason || !isfinite(result.lo) || !isfinite(result.hi)) return ci_unknown(4);
    return result;
}
__device__ CI ci_exp_shift(CI x) {
    if (x.reason) return x;
    CI lo=x.lo==CI_NINF ? ci_point(0.0) : ci_exp(x.lo);
    CI hi=ci_exp(fmin(x.hi,0.0));
    if (lo.reason || hi.reason) return ci_unknown(ci_reason(lo,hi));
    return ci_checked(lo.lo,fmin(hi.hi,1.0)); // exact shifted max implies exp<=1
}
__device__ CI ci_log_interval(CI x) {
    if (x.reason) return x;
    CI lo=ci_log(x.lo),hi=ci_log(x.hi);
    if (lo.reason || hi.reason) return ci_unknown(ci_reason(lo,hi));
    return ci_checked(lo.lo,hi.hi);
}
__device__ double ci_reduce(double value,double* shared,bool upper) {
    shared[threadIdx.x]=value; __syncthreads();
    for (unsigned int stride=CI_BLOCK/2;stride>0;stride>>=1) {
        if (threadIdx.x<stride) shared[threadIdx.x]=upper
            ? __dadd_ru(shared[threadIdx.x],shared[threadIdx.x+stride])
            : __dadd_rd(shared[threadIdx.x],shared[threadIdx.x+stride]);
        __syncthreads();
    }
    double result=shared[0]; __syncthreads(); return result;
}
__device__ double ci_reduce_max(double value,double* shared) {
    shared[threadIdx.x]=value; __syncthreads();
    for (unsigned int stride=CI_BLOCK/2;stride>0;stride>>=1) {
        if (threadIdx.x<stride) shared[threadIdx.x]=fmax(shared[threadIdx.x],shared[threadIdx.x+stride]);
        __syncthreads();
    }
    double result=shared[0]; __syncthreads(); return result;
}
__device__ void ci_store(double* output,ci_u64 row,CI value) {
    if (!value.reason && (!isfinite(value.lo) || !isfinite(value.hi) || value.lo>value.hi)) value=ci_unknown(4);
    output[row*3]=value.lo; output[row*3+1]=value.hi; output[row*3+2]=(double)value.reason;
}
extern "C" __global__ void checked_scalar_interval(ci_u64 count,unsigned int mode,const double* input,double* output) {
    for (ci_u64 i=(ci_u64)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(ci_u64)blockDim.x*gridDim.x) {
        CI result=!ci_arithmetic_guard() ? ci_unknown(5) : (mode==0 ? ci_exp(input[i]) : ci_log(input[i]));
        ci_store(output,i,result);
    }
}
extern "C" __global__ void checked_kl_interval(unsigned int rows,unsigned int cols,
    const double* teacher,const double* explained,double* output) {
    __shared__ double shared[CI_BLOCK];
    unsigned int row=blockIdx.x;
    if (row>=rows) return;
    const double* z=teacher+(ci_u64)row*cols;
    const double* w=explained+(ci_u64)row*cols;
    double bad=ci_arithmetic_guard() ? 0.0 : 5.0;
    CI offset=ci_sub(ci_point(z[0]),ci_point(w[0]));
    double changed=(!offset.reason && offset.lo==offset.hi && isfinite(offset.lo)) ? 0.0 : 1.0;
    double mz=CI_NINF,mw=CI_NINF;
    for (unsigned int c=threadIdx.x;c<cols;c+=CI_BLOCK) {
        if (!isfinite(z[c]) || !isfinite(w[c])) bad=fmax(bad,1.0);
        CI diff=ci_sub(ci_point(z[c]),ci_point(w[c]));
        if (diff.reason || diff.lo!=offset.lo || diff.hi!=offset.hi) changed=1.0;
        mz=fmax(mz,z[c]); mw=fmax(mw,w[c]);
    }
    double invalid=ci_reduce_max(bad,shared);
    double mismatch=ci_reduce_max(changed,shared);
    mz=ci_reduce_max(mz,shared); mw=ci_reduce_max(mw,shared);
    if (invalid>0.0) { if (threadIdx.x==0) ci_store(output,row,ci_unknown((unsigned int)invalid)); return; }
    if (mismatch==0.0) { if (threadIdx.x==0) ci_store(output,row,ci_point(0.0)); return; }
    CI sz=ci_point(0.0),sw=sz,weighted=sz;
    for (unsigned int c=threadIdx.x;c<cols;c+=CI_BLOCK) {
        CI a=ci_sub(ci_point(z[c]),ci_point(mz)),b=ci_sub(ci_point(w[c]),ci_point(mw));
        CI ep=ci_exp_shift(a),eq=ci_exp_shift(b);
        sz=ci_add(sz,ep); sw=ci_add(sw,eq);
        weighted=ci_add(weighted,ci_mul(ep,ci_sub(a,b)));
    }
    bad=(double)ci_reason(sz,ci_unknown(ci_reason(sw,weighted)));
    invalid=ci_reduce_max(bad,shared);
    if (invalid>0.0) { if (threadIdx.x==0) ci_store(output,row,ci_unknown((unsigned int)invalid)); return; }
    double szlo=ci_reduce(sz.lo,shared,false),szhi=ci_reduce(sz.hi,shared,true);
    double swlo=ci_reduce(sw.lo,shared,false),swhi=ci_reduce(sw.hi,shared,true);
    double ulo=ci_reduce(weighted.lo,shared,false),uhi=ci_reduce(weighted.hi,shared,true);
    if (threadIdx.x==0) {
        CI p=ci_checked(fmax(szlo,1.0),szhi),q=ci_checked(fmax(swlo,1.0),swhi);
        CI result=ci_sub(ci_add(ci_div_positive(ci_checked(ulo,uhi),p),ci_log_interval(q)),ci_log_interval(p));
        result.lo=fmax(result.lo,0.0); // exact KL nonnegativity
        ci_store(output,row,result);
    }
}
