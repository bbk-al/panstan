
API
===

Classes Summary
---------------

.. autosummary::
   :toctree: classes
   :template: class.rst

   panstan.ModelList
   panstan.MyStanModel
   panstan.JsonEnc


Functions Summary
-----------------
Serialisation
^^^^^^^^^^^^^

.. autosummary::
   :toctree: functions

   panstan.jsdump
   panstan.jsdumps
   panstan.jsload
   panstan.jsloads
   panstan.jsplus

Statistics
^^^^^^^^^^^

.. autosummary::
   :toctree: functions


   panstan.hdi
   panstan.lrfit

Posterior Sample Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: functions

   panstan.gencases
   panstan.implications
   panstan.implications_quantile
   panstan.mywaic
   panstan.precis
   panstan.str2mi

Plotting functions
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: functions

   panstan.traceplot
   panstan.pairs
   panstan.hpdiplot
   panstan.pdfplot

Utilities
^^^^^^^^^

.. autosummary::
   :toctree: functions

   panstan.inv_logit
   panstan.inv_logit_scaled

Argument parsers
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: functions

   panstan.parse_date
   panstan.parse_non_negative_int
   panstan.parse_non_negative


Constants Summary
-----------------

.. autosummary::
   :toctree: data

   panstan.Auto
   panstan.CredMass
   panstan.ModelCode


