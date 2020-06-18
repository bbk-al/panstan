// Model type m
data {
  int<lower=0> N;          // number of data points
  real lval[N];        // observed data points
  int datenum[N];          // dates (as day counts) for above
  int<lower=0> M;          // number of suspect points
  int missing[M];          // suspect data points
  int<lower=1> I;          // number of intervals (interventions + 1)
  int<lower=0,upper=1> S;  // 1 = number of SD values is I; 0 = number is 1
  int<lower=0,upper=1> C;  // 1 = constant SD; 0 = variable SD with mum below
  int<lower=0,upper=1> G;  // 1 = apply geometric dist; 0 = use normal only
  int<lower=min(datenum),upper=max(datenum)> d[I-1];  // param change dates
  real tol;                // Tolerance (minimum permitted SD)
  real worldpop;           // Maximum peak value permitted
}
transformed data {
  real logworldpop = log(worldpop); // Log of above, avoid recalculating
  real klogi = 5.764;               // Chosen for Worldpop constraint
  real logsqrtpi = log(sqrt(pi())); // used in Gaussian peak/total calculation
  real ones[I] = rep_array(1.0,I);  // conditioners for population control
}
parameters {
  real<lower=0> a[I];              // square term -> shape
  real b[I];                       // linear term -> position
  real c[I];                       // height of peak
  real<lower=0> sigmam[S*(I-1)+1]; // shape:  x height to get sd
  vector[M] logdiffi;              // imputed points
  vector[I-1] mud;                 // means for dates of interventions
}
model{
  vector[N] x = to_vector(datenum); // vector copy of date
  vector[N] mum;        // means for fitted Gaussian
  vector[N] sgm;        // sd's for fitted Gaussian
  vector[N] logdiffm;   // merged observed and imputed data points
  int s;                // number of sigmam values
  real p[I];            // population control factor
  int mix[N];           // multi-index used to select vector entries for update
  int ix;               // index into mix and then length of selector
  int abl[N];           // counts of above/below mum

  // baseline parameter draws
  for (i in 1:I) {
    // These three will correlate because of the tight fit of the data to the
    // curve, and not because of an underlying collinearity.
    a[i] ~ cauchy(0, 1);
    b[i] ~ normal(0, 1);
    c[i] ~ normal(0,12);
    if (i < I) {
      // Expect divisions to be accurate to a day or so
      d[i] ~ normal(mud[i],1);
    }
  }
  s = S*(I-1)+1;
  for (i in 1:s) {
    sigmam[i] ~ cauchy(0,1); // 4-3.5*C);
  }

  // regression
  // interventions
  for (i in 1:I) {
    // Factor to reduce likelihood of wilder results with too many cases
    p[i] = inv_logit(-klogi*(logsqrtpi -log(a[i])/2 +b[i]*b[i]/(4*a[i]) -c[i]
                             -logworldpop));
    ix = 0;
    for (n in 1:N) {
      if ((i == 1 || x[n] >= mud[i-1]) && (i == I || x[n] < mud[i])) {
        ix = ix + 1;
        mix[ix] = n;
      }
    }
    if (ix > 0) {
      // Construct index selector
      int sel[ix];  // Local definition
      sel = head(mix,ix);
      // Calculate per-interval mum and sgm for logdiffm distribution
      mum[sel] = -c[i] - a[i]*square(x[sel]) + b[i]*x[sel];
      sgm[sel] = rep_vector(fmax(sigmam[min(i,s)],tol),ix);
      // Following increases processing time by roughly 50-100%
      if (C != 0) {  // Not mathematically necessary but quicker for C==0
        // New mum merits updated sgm
        // NB Logistic is used to tame extremes (negative mums): min -0.28 ish
        sgm[sel] = sgm[sel] .* exp(-C*mum[sel] .* inv_logit(mum[sel]));
      }
    }
  }

  // Merge missing and observed for imputation.
  if (G != 0) {
    // Count run lengths for values above/below mum;  ensure equal sample sizes.
    // The latter adds approx 50% to processing time.
    int m = 1;            // missing value index
    int rl = 0;           // run length
    int r = 1;            // run index
    for (n in 1:N) {
      if (m <= M && n == missing[m]) {
        logdiffm[n] = logdiffi[m];
        m = m + 1;
      } else {
        logdiffm[n] = lval[n];
        for (sgn in -1:1) {
          if (sgn == 0) continue;
          if (sgn*lval[n] > sgn*mum[n]) {
            if (sgn*rl < 0) {
              if (n > r+1) abl[r:(n-2)] = rep_array(0,n-r-1);
              abl[n-1] = abs(rl);
              rl = sgn;
              r = n;
            } else {
              rl = rl + sgn;
            }
            break;
          }
        }
      }
    }
    abl[N] = abs(rl);
    if (N > r) abl[r:(N-1)] = rep_array(0,N-r);
  } else {
    // No above/below geometric dist: copy and overwrite as fastest option
    logdiffm = to_vector(lval);
    for (m in 1:M) logdiffm[missing[m]] = logdiffi[m];
  }

  // fit
  ones ~ normal(p,0.1);
  if (G != 0) abl ~ neg_binomial(1,1);
  logdiffm ~ normal(mum,sgm);
}
generated quantities{
  real mu[I];
  real sigma[I];
  real peak[I];
  for (i in 1:I) {
    real lpk;
    sigma[i] = 1/sqrt(2*a[i]);
    mu[i] = b[i]/(2*a[i]);
    lpk = a[i]*mu[i]*mu[i] - c[i];
    //if (lpk > logworldpop)
    //  peak[i] = worldpop;
    //else
    peak[i] = exp(lpk);
  }
}
