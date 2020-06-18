Experiments with Stan
---------------------

In addition to using `PyStan
<https://mc-stan.org/users/interfaces/pystan>`_ in place of
`RStan <https://mc-stan.org/users/interfaces/rstan>`_ (and the consequent
need for
`Pandas <https://pandas.pydata.org/pandas-docs/stable/reference/index.html>`_
and `Numpy <https://numpy.org/doc/stable/reference/index.html>`_ ) this
project has explored some of the possibilities of Stan programming for
MCMC analyses.

Limiting population
^^^^^^^^^^^^^^^^^^^

For example, fitting Gaussian splines to epidemic data, rather than a
true epidemic curve, carries the problem of there being no susceptible
population limits:  some of the wilder results would have more people
infected than have ever existed.  It turns out that a limit can be
cleanly imposed with no significant processing or convergence cost.

The technique used is to estimate the total caseload implied by a
given Gaussian and then to use this total in the construction of a
logistic curve.  That logistic curve is then used as the mean of a normal
distribution with a small SD (e.g. 0.1), such that 1 is a very likely
sample value and 0 very unlikely.  Then an array of ones is fitted to
this normal distribution, applying a strong potential for infeasible
caseloads and none for smaller ones.
 
.. code-block:: Stan

  data {
    //...
    real worldpop;           // Maximum peak value permitted
  }
  transformed_data {
    real logworldpop = log(worldpop); // Log of above, avoid recalculating
    real klogi = 5.764;               // Chosen for Worldpop constraint
    real logsqrtpi = log(sqrt(pi())); // used in Gaussian peak/total calculation
    real ones[I] = rep_array(1.0,I);  // conditioners for population control
  }
  //...
  model {
    //...
    // For each spline
    for (i in 1:I) {
      // Factor to reduce likelihood of wilder results with too many cases
      p[i] = inv_logit(-klogi*(logsqrtpi -log(a[i])/2 +b[i]*b[i]/(4*a[i]) -c[i]
                               -logworldpop));
    //...
    // fit
    ones ~ normal(p,0.1);
  }

In the example above, ``klogi`` has been chosen to shape the logistic,
making the transition between acceptable and unacceptable caseloads
steep but not too steep.  The logistic (``inv_logit``) has a positive
argument when the Gaussian total is less than the prescribed limit
(``logworldpop``, the log of the data value ``worldpop`` supplied to
Stan).  The Gaussian total happens to be very easy to calculate based
on the curve's parameters (but beware of the sign choices which make
parameter ``a`` positive and whilst the choice for ``c`` is historical).

The technique above is potentially useful for encoding any boundary
into the model of a Stan program rather than as a data or parameter
constraint.  It may be extensible to wider situations in which a more
complex true/false constraint applies.  This can help with avoiding
convergence and rejection issues, as well as real-world infeasibilities.

Alternative shape potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This success encouraged some exploration of ways to tackle one of the
weaknesses of the Gaussian spline approach to modelling epidemic data.
In particular, the pressure exerted on the shape of the fitted curves is
weakest for the constant SD model, where, paradoxically, the log-world
SD is strongly dependent on all three primary parameters, ``a``, ``b``
and ``c``.  In contrast, for the variable SD model, in the log-world,
only parameter ``a`` counts, and all the pressure drives its convergence
to an accurate estimate.

To increase the potential, and so steepen gradients to favour rapid
convergence, an independent criterion affecting the shape is required.
An additional submodel counts the length of runs of data points above
or below the fitted curve, and then fits these runs to a geometric
distribution.  The idea is that a curve which shows only short runs of
points above or below it should be better than a curve which has lots
of initial points below it and the rest above it.

The implementation requires the following components.

.. code-block:: Stan

  model {
    //...
    int abl[N];           // counts of above/below mum
    //...
      // Count run lengths for values above/below mum; ensure equal sample sizes
      // The latter adds approx 50% to processing time.
      int m = 1;            // missing value index
      int rl = 0;           // run length
      int r = 1;            // run index
      for (n in 1:N) {
        //...
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
    //...
      abl ~ neg_binomial(1,1);
    //...
  }

The decision on whether or not to apply the geometric distribution fit
is controlled via the data value ``G`` in the full model.  The snippet
above is focussed solely on the construction of the ``abl`` above/below
list and its fit to the distribution.  Note that in Stan, the geometric
distribution for probability 0.5 is spelt ``neg_binomial(1,1)``, the
second parameter being the odds rather than the probability.  In contrast,
in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_,
the spelling is ``nbinom.pmf(values,1,0.5)`` (though the ``logpmf``
is used in practice).

The ``-1:1`` loop above merely avoids a lengthy and repetitive if-else case.
It effectively assigns probability 0.5 to even the first value in a run,
which is to avoid penalising exact matches.  Zeroes are inserted in
between runs in order to avoid penalising a long series of short runs
relative to a short series of long ones.  The zeroes will still have a
cost, but the same cost as an ideal run, so  that the full impact of an
above/below run is properly accounted for.

The effects of this code were, on balance, negative.  There was an
improvement in convergence, especially for weakly convergent parameters,
adding of the order of 10% to :math:`n_{eff}`.  But this came at the
cost of a 50% increase in processing time, and there was no consistent
improvement in uncorrected WAIC scores.  When the WAIC calculation
accounted for the extra geometric probabilities, there was still no
improvement and the pWAIC values were too large (:math:`O(100)`) and ran
in the wrong direction relative to the model complexity (pWAIC increased
as the number of parameters decreased).

The behaviour of the 'corrected' WAIC as implemented could always be due
to an error, but it is reasonable that it simply reflects the patterns
in the log likelihood.  The correction has lowered all likelihoods,
amplifying the variance involved and hence the pWAIC has increased.
That this has affected the simpler models more strongly is no surprise,
since these have less scope for balancing the twin objectives of
minimising both variance and bias.

This outcome reflects that the potential applied is against a different
data set, not the original data points but the lengths of consecutive runs
of data consistently above/below the model.  Its effect on parameter
estimation for models of the original data must therefore be weak.
Indeed, instead of just calculating the probability, we could estimate
the second parameter of the negative binomial distribution, and then
see how far that is from the ideal probablity 0.5.

As a result, the geometric distribution (or directed variance) is only
applied in WAIC comparisons if any current model makes use of it.  That
way, it is easy to avoid the consequences of this particular experiment.


Variants of the main model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Aside from experimenting with additional potential features, the model
includes two other variants affecting the shape parameter.  One requires
a single shape to apply across all splines, the idea being that with
more data applied to this one parameter, convergence might improve.
The second attempts to model a constant shape across each spline (or
across the whole model if combined with the first variant).  Details of
this second variant are provided with the :const:`panstan.ModelCode`
description.

In both cases, fits confirm that the variable multi-spline shape is the
best modelling option, which is why it is the default.

Futures
^^^^^^^

The current model applies the interventions as cut-offs, albeit with
some fuzziness applied via a normal distribution for the intervention
date itself.  A possibly better approach would be to use logistic curves
to define the intervals and to sum these to produce a single potential
with no hard cut-off.  However, in the absence of serious performance or
convergence issues, this approach has dropped in priority for evaluation.

Whilst the Gaussian spline-fitting approach appears to work reasonably
well, the real question is how it compares to a true epidemic curve.
This is next on the list.

A Few Highlights
----------------

An epidemiological analysis of the pandemic was not the motivation of
this project and absolutely no claim to a capability to perform such
an analysis is being made.  However, a few results are worth noting
as indicators of the program's success and limitations.  For a true
epidemiological analysis, refer to the relevant experts!

Typical run times using three cores on an 8GB laptop vary from 40s for
models with no interventions to 25 minutes for three interventions and
over half an hour if a constant SD is sought (for approximately 110 data
points).  It is not practical to determine accurate time dependencies,
but with the limited data available, very roughly the processing time
varies with the number of data points and the log of the number of
intervals (i.e. interventions plus one).  Python's multithreading is weak
owing to problems with the Global Interpreter Lock, and, consequently,
PyStan runs multiprocessing with separate full processes.  In this regard,
the RStan version is a little faster.  However, pandas and numpy demand
careful coding to achieve good performance, and, on occasions, this has
enabled the Python program to outperform the R version overall.

For the simpler models, there are no divergences and no saturation
of the tree depth (defaulting to 10).  For the more complex models,
there is progressive deterioration.  The pairs plot turns from blue to
yellow as saturation occurs, and the trace plots gradually indicate
weaker mixing between chains.  Eventually, the number of effective
samples (:math:`n_{eff}`) drops and the Gelman-Rubin convergence ratio
(:math:`\hat{R}`) starts to give values above 1.00.  Despite this,
the resulting plots and tables are generally plausible.

The first indication that the Gaussian spline strategy could work came
with the end of the first wave of the Chinese epidemic.  Models with
interventions showed a second peak forming soon after, with this peak
initially predicted as severe and then progressively reducing as more
data became available.  As of June 2020, the Chinese data is problematic
for the program because it rejects a large number of data points (34,
or 20%), with another 6 missing from the ECDC dataset.  This discourages
convergence.  However, sensible plots with 2 or more interventions
are possible.

The British response to the pandemic has produced some notable features.
The raw data demonstrate sharp changes in early April and again at the
end of April or early May.  These correspond quite credibly with the
impact of the lockdown implemented in late March (allowing for a week or
two delay) and the rapid increase in testing that occurred in late April.
The minimum number of interventions needed for a decent fit is therefore
2, but marginal gains are made at 3 according to the WAIC comparison.
The implications table provides hints of what the caseload numbers might
have been had the higher level of testing been in place earlier (though
this would not substantially have affected the hospitalisation rate
and certainly not the deaths).  Since then (as of 17th June 2020), the
predicted end to the outbreak with a 2 or 3 intervention model has moved
from mid-July to November, suggesting either a slowing rate of decline or,
possibly, a feature of a Gaussian fit as opposed to a true epidemic curve.

The US data is problematic for several reasons, including the early
weekday-only reporting and the treatment of the USA as a single entity
when it appears to be better represented as a series of linked large-scale
clusters.  This results in a large number of missing data points being
identified, and in a fairly complex pattern.  However, in the round, the
data is reasonably well described by a model with just one intervention,
and further interventions focus on early details, perhaps indicative of
successive clusters.

