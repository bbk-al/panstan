Installation
------------

Installation should be straightforward following standard module
installation methods.


Program Usage
-------------

The command line is documented in the usual manner:  ``panstan.py -h``
will summarise the options available.  Note that options (expecting
data values) are always lower case, and flags (always false by default)
are in upper case.  This section looks at typical uses by way of further
explanation.

Data
^^^^

File and data storage has not been a major focus, so the capabilities
are narrow.

It is assumed that a ``data`` directory exists locally.  This will be
used to store all required data files.

By default, the program pulls data from the ECDC site, and this has been
the main source of test data.  ECDC data is automatically refreshed from
10:00CET onwards and only once per day (with at most hourly checks if
data is not available).  The program also retains some ability to analyse
data in ODS files, and is coded to be flexible about the file layout,
but the further this drifts from the ECDC structure the more likely it
is that problems will arise.

To use the ECDC site, apply no ``-f`` option;  otherwise, use this and
specify the file location.

.. code-block:: console

  panstan.py                      # ECDC data to be used
  panstan.py -f data/myfile.ODS   # Use a locally constructed data set

All output data is likewise stored in the ``data`` directory.  For the
most part, this data is in gzipped JSON format, with some specialisations
to support non-standard structures (see the :ref:`Serialisation`
functions).  This serialisation is safe for exchange (unlike pickle data),
fairly portable and moderately compact, but it is far from optimal for
dataframe storage.

To avoid saving data from a model run specify ``-N``.

Keys, hashes and pickled code
"""""""""""""""""""""""""""""

The only exception to compressed JSON is with compiled model code.  Here,
the only practical option is pickle, made more acceptable by the limited
portability of compiled code.  Some protection against 'corruption'
is provided by encrypted hashes of the compiled code held separately
and specific to the user.  A failure to verify the integrity of compiled
code will only result in recompilation of the model, a significant time
cost but no more than that.

Data selection
^^^^^^^^^^^^^^

Two options control the selection of data:

* ``-c column`` to select the column containing the data to analyse;
* ``-d data`` to filter the data by 'data'.

The data filter works by identifying the most common occurrence of the
filter terms and then using the associated column to filter against.
There is no need to specify the filter column.

For ECDC data, the country required can be specified by its abbreviations
or full name (with underscores).  When matched, the program reports on
its aliases and supergroup(s) (i.e. continent).  The column defaults to
cases, but deaths can be selected.  The data defaults to UK.

In addition, the ``-T`` flag specifies that the data is of totals not
new cases.  Totals are converted to daily new cases.

When the program runs, it summarises the requested and available
models and indicates which of these will be run.  When a number of
model specifications for the same country have been built up, the ``-A``
flag indicates a requirement to run or analyse all previously run models.

Data corrections
^^^^^^^^^^^^^^^^

Sometimes the ECDC data includes wild swings in data owing to corrections
being applied.  It is possible to smooth these out by constructing a CSV
containing the 'true' data values for the affected dates and countries.
To implement such a set of corrections:

.. code-block:: console

  panstan.py -f data/ECDC_Corrections.csv -M

The ``-M`` flag tells the program to apply the csv as corrections the
main data set and to save these corrections.  Subsequent runs (without
``-f`` or ``-M``) will incorporate the corrections automatically.
If it is required to run the original, uncorrected data, used the ``-M``
flag on its own.

Interventions, intervals and junctions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An intervention is a point at which a change in parameters is
required to improve the fit to the data, likely signifying the impact
of an intervention (such as lockdown or rapidly expanded testing).
Such points can be located automatically (``-i``) or can be specified
as dates (``-j``).  In the latter case, interventions are referred to
as junctions.  A combination of automatically located intervention dates
and fixed junctions can be used.  The start and end of the pandemic are
unspecified (but for plotting these are calculated as the dates on which
the number of cases falls below 0.5 according to the plotted model).
Each side of an intervention is an interval, so there is one more interval
than interventions.

.. code-block:: console

  panstan.py -i 0 1 2            # Fit models for 0, 1 and 2 interventions
  panstan.py -i 3 -j 8-May-2020  # Fit 3 interventions one of which is fixed

Automatic assignment of interventions is achieved by seeking points where
the variance is at a maximum.  This is done during the determination of
initial values:  it can be done entirely within the MCMC analysis, but
this is slow and weakens convergence.  The initial value is subject to
variation within Stan via a normal distribution with an SD of one day.
A minimum number of points is required within an interval, which is
currently 9 (3 for a quadratic fit, and 3 either side to allow for
adjustments made within the Stan model).  This is necessary to avoid
a situation where the quadratic is underfit, permitting very wild
parameter changes.  The probability of this occurring is reduced to
about 1 in 50000 for the minimum interval size.

Shape parameter variants
^^^^^^^^^^^^^^^^^^^^^^^^

Constant SD
"""""""""""

The default is to vary the SD proportionately with the mean.  To run a
constant-SD model as well as the default case, specify ``-C``.  See also
:const:`panstan.ModelCode`.

Single SD
"""""""""

The default is to allow the SD to change for each interval (Gaussian
spline).  To run a single-SD model as well as the default, specify ``-S``.

Directed
""""""""

An experimental model attempts to detect bias in the fit and use this
to drive improvements in convergence (see
:ref:`Alternative shape potentials`).  To run models with this enabled,
use ``-D``.

Note.  The pWAIC is suspect when the directed models are used, but the
geometric distribution correction is automatically disapplied when there
are no geometric models in the comparison.

Displays options and flags
^^^^^^^^^^^^^^^^^^^^^^^^^^

When an MCMC analysis is run, the trace and pairs plots are run
automatically, unless ``-P`` is specified.  These diagnostics are skipped
if no MCMC is run, unless ``+P`` is specified.

Unless ``-P`` is specified, a plot of the highest posterior density interval
(HPDI) is always displayed.  This consists of the median model in blue,
the upper credibility interval boundary in red, and the lower boundary
in green.  In between the two boundaries, the background is shaded grey.
However, this grey shading is often fully obscured by pink or green
shading for the HPDI boundaries :math:`\pm 1` SD.  These shaded areas help
to indicate the level of uncertainty in the model.

The HPDI plot will automatically switch to log-scale for the daily new
cases if the range of values is great enough (between upper HPDI + 1SD
and median).  This can be overridden by ``-L`` to never use a log-scale,
and ``+L`` to always use a log-scale.

``-O`` will save each plot to a page in the file '{column}{data}.pdf'
where column and data come from the ``-c`` and ``-d`` options.

These options are mostly self-explanatory from the help:

* ``-n num`` display num sample curves on top of the standard HPDI plot;
* ``-s num`` use num samples in calculations and graphs.

MCMC control
^^^^^^^^^^^^

The main flags are ``-R`` to use only stored data from previous runs
(i.e. do not rerun MCMC even if the data is out of date), and ``+R``
to force a rerun of MCMC.

There are several Stan-specific options prefixed ``--mc``, which
do not have short forms.  These override the stated defaults.
See `PyStan <https://mc-stan.org/users/interfaces/pystan>`_ and `Stan
<https://mc-stan.org/users/documentation/>`_ documentation for details.

Textual reporting
^^^^^^^^^^^^^^^^^

The program produces several pieces of textual report with no options
to modify these.

* Information on the program's interpretation of the command line and
  data set.  This culminates in a table showing which models are available
  and which will be processed.
* For each model processed:

  * Initial values supplied to Stan.
  * Status information, including Stan timing information, warnings and
    progress statements every ``mcwarmup`` iterations.
  * Summary statistics for each model parameter.  The summary uses
    highest density intervals (HDI) rather than quantiles, and adds in
    data for :math:`n_{eff}` and :math:`\hat{R}`.
  * A table of the model's implications listing Peak (new daily cases),
    Total cases, Duration of the outbreak, and the Start, Peak and End
    dates.  These are specified by HDI for different credibilities and
    for the median.

* A final table displaying the WAIC comparison.  See :func:`panstan.mywaic`
  for details.

