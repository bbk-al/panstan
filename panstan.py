#!/usr/bin/env python
"""

This module explores the use of pandas rather than R in Stan modelling
of the covid-19 pandemic data.  The modelling is based upon Gaussian
splicing rather than true epidemic curves, but will auto-detect changes
in parameters and limit predicted caseloads to a population size.

"""

__version__ = '0.0.1'
__all__ = [
  'MyStanModel', 'JsonEnc', 'ModelList',
  'jsplus', 'jsdump', 'jsdumps', 'jsload', 'jsloads',
  'str2mi', 'hdi', 'traceplot', 'pairs', 'pdfplot', 'hpdiplot',
  'mywaic', 'precis', 'implications', 'implications_quantile',
  'inv_logit', 'inv_logit_scaled', 'gencases', 'lrfit',
  'parse_date', 'parse_non_negative_int', 'parse_non_negative',
]

__author__ = "Adam Light dev@allight.plus.com"

#----------------------------------------------------------------------------
# Clean imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib.dates as pld
import matplotlib.lines as pll
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm,nbinom

import argparse
import re
from datetime import datetime,timedelta
import pytz
from time import sleep
from multiprocessing import cpu_count
import json
import gzip
from ast import literal_eval
#import codecs
from urllib.error import URLError
#from tempfile import TemporaryFile,mkstemp
import os,io

import collections.abc,six
from itertools import islice,cycle
from pathlib import Path
import hashlib,hmac,secrets
import pickle

#----------------------------------------------------------------------------
# Package fixes

# Pkg: pandas
# To read ODS, install odfpy;  then pd.read_excel engine='odf' will work.

#............................................................................
# Pkg: pystan
# Annoying piece of code needed to stop pystan issuing unwanted INFO
# Alternative is to preconfigure logging and suppress INFO until after import
os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count()-1)
# pacaur -Sy python-pystan  # Not AUR
from pystan import StanModel
from pystan.misc import stan_rdump,read_rdump
# Pystan touts Arviz for plots and data storage, but it does not work with
# pystan output in python.  Pystan's own plots are deprecated by itself.
# The ony remaining option is to write bespoke plot functons...

#----------------------------------------------------------------------------
# Serialisation choice:
# Pickle is supported, but is inherently unsafe (eval) and Python-specific.
# RData as supported, but for numeric data only.
# Efficient data file formats (feather,HD5 etc) are data frame only.
# Instead, compressed json offers reasonable resource usage and generality.
# NB json-tricks does not work with pandas;  json-pickle has pickle issues.

#............................................................................

class MyStanModel(StanModel):
  """
  Subclass of `pystan.StanModel <https://mc-stan.org/users/interfaces/pystan>`_
  providing additional support for serialisation.

  Methods
  -------
  load
  save

  Attributes
  ----------
  None

  See Also
  --------
  MyStanModel.load : load previously saved compiled model and check hash.
  MyStanModel.save : save compiled model and return hash.

  Notes
  -----
  The greater part of the serialisation is provided via class :class:`JsonEnc`.
  The only exception is for the compiled models themselves, which use pickle
  secured via encrypted hashes.  This is suitable only for local use, as is
  the compiled model itself.

  The secret key is stored in the ``__file__`` parent path in `.stankey`,
  which is set read only by user.  The pickled models are stored in
  `.stan-<mt>` where `<mt>` is the model type string.  The hash returned by
  `save()` should be separately stored by the caller (see class
  :class:`ModelList`).

  Examples
  --------
  >>> m = MyStanModel(model_code=mt_str)
  """
  _path = Path(__file__).parent
  _keypath = _path / '.stankey'
  _hashkey = None

  # Utility to ensure private class attribute _hashkey is set.
  # mt parameter is not currently used:  same secret key for all models.
  @classmethod
  def _setkey(cls,mt):
    if cls._hashkey is None:
      if cls._keypath.is_file():
        with open(cls._keypath,'rb') as f: cls._hashkey = f.read()
      else:
        cls._hashkey = secrets.token_bytes(nbytes=1024)
        with open(cls._keypath,'wb') as f: f.write(cls._hashkey)
        cls._keypath.chmod(0o400)

  @classmethod
  def load(cls,mt,mash):
    """
    Load previously saved compiled model and check hash.

    Parameters
    ----------
    mt : str
      The model type (identifying the model code).
    mash : str
      Hex representation of expected digest / hash.

    Returns
    -------
    MyStanModel object or None if hashes did not match.

    See Also
    --------
    MyStanModel.save : save compiled model and return hash.

    Examples
    --------
    >>> m = MyStanModel.load("m",model_dict['python'])
    """
    # Set up 
    cls._setkey(mt)
    m = None
    # Load model from file
    modpath = cls._path / (".stan-" + mt)
    try:
      with open(modpath, 'rb') as f: ms = f.read()
    except:
      pass
    else:
      h = hmac.new(cls._hashkey,msg=ms,digestmod=hashlib.sha256)
      if hmac.compare_digest(h.hexdigest(),mash):
        m = pickle.loads(ms) #,protocol=pickle.HIGHEST_PROTOCOL)
    return m

  # Constructor: pass through to subclass
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)

  # Save model to file
  def save(self,mt):
    """
    Save compiled model and return hash.

    Parameters
    ----------
    mt : str
      The model type (identifying the model code).

    Returns
    -------
    dict of 'python' set to the hex representation of the hash.

    See Also
    --------
    MyStanModel.load : load previously saved compiled model and check hash.

    Examples
    --------
    >>> m = MyStanModel(model_code=mt_str)
    >>> model_dict = m.save("m")
    """
    MyStanModel._setkey(mt)
    ms = pickle.dumps(self,protocol=pickle.HIGHEST_PROTOCOL)
    h = hmac.new(self._hashkey,msg=ms,digestmod=hashlib.sha256)
    modpath = self._path / (".stan-" + mt)
    with open(modpath, 'wb') as f: f.write(ms)
    return dict(python=h.hexdigest())

#............................................................................

# Pkg: json - is too limited in its capabilities, so subclassed here
# Subclass Encoder and Decoder to add capabilities
class JsonEnc(json.JSONEncoder):
  """
  Subclass of `json.JsonEncoder
  <https://docs.python.org/3/library/json.html#json.JSONEncoder>`_
  extending serialisation support to include datetime, dataframe, series and
  any other class with a `to_json()` method.

  Methods
  -------
  default

  Attributes
  ----------
  None

  Notes
  -----
  JsonEnc.default is called via `json.JsonEncoder
  <https://docs.python.org/3/library/json.html#json.JSONEncoder>`_ to encode
  new types, and should not be called directly.  Whilst `pandas.DataFrame
  <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
  and `pandas.Series
  <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_
  have to_json() methods, these do not round-trip with a `pandas.MultiIndex
  <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html>`_
  .  Explicit support is provided here.

  Examples
  --------
  >>> s = json.dumps(obj,cls=JsonEnc)
  """
  def default(self, obj):
    """
    Specialised from `json.JsonEncoder
    <https://docs.python.org/3/library/json.html#json.JSONEncoder>`_.
    """
    # Is it a datetime? (Can add others)
    if isinstance(obj, (datetime,)): return obj.isoformat()
    # Is it a convertible numpy?
    try:
      if isinstance(obj, np.generic): return obj.item()
      if isinstance(obj, np.ndarray): obj = pd.Series(obj) # pass to next case
    except Exception as e:
      #print("Exception raised: " + str(e))
      pass
    # Is it a pd.DataFrame or pd.Series
    # This is to handle MultiIndex, 7 yrs and counting from pandas request
    if isinstance(obj, (pd.DataFrame,pd.Series)):
      axl = [('index',obj.index)]
      if isinstance(obj, (pd.DataFrame,)): axl += [('columns',obj.columns)]
      for ax in axl:
        if isinstance(ax[1],pd.MultiIndex):
          values = [repr(v) for v in ax[1].ravel()]
          obj = obj.set_axis(pd.Index(values),axis=ax[0])
      return obj.to_json(orient='split')
    # Is it another class with a to_json method?
    try:
      return getattr(obj.__class__, "to_json")(obj)
    except AttributeError as e:
      #print("AttributeError raised:"+str(e))
      pass
    # See if json can do this by default, else it will raise TypeError
    return json.JSONEncoder.default(self, obj)
    #raise TypeError("Type {0:s} not serialisable".format(type(obj).__name__))

def jsplus(obj):
  """
  Support function to decode specialist json strings.

  Parameters
  ----------
  obj : Any
    The object to be serialised.

  Returns
  -------
  A decoded object if one string was passed;  a list of objects if a\
  non-string iterable was passed;  otherwise the object is returned.

  Notes
  -----
  This supports decodes of json dumps of a list of variables.
  jsplus runs through a list of predefined possible types and attempts to
  interpret each entry using each predefined type in turn.

  Examples
  --------
  >>> partial_decode = json.loads(js_str)
  >>> obj = jsplus(partial_decode)
  """
  # Local utility to decode tuples to a MultiIndex, for both index and columns.
  def pd_from_json(obj):
    df = pd.read_json(obj,orient='split')
    for ax in [('index',df.index),('columns',df.columns)]:
      mi = False
      try:
        mi = ax[1][0].startswith('(')
      except:
        pass
      if mi:
        df = df.set_axis(pd.MultiIndex.from_tuples(
                         (literal_eval(t) for t in ax[1])),axis=ax[0])
    return df

  # Localised decoder utility for MyStanModel
  def model_from_json(obj):
    return ModelList.from_json(obj)

  # The list of supported types and the applicable decode method
  trial = dict(datetime=datetime.fromisoformat,pandas=pd_from_json,
               model=model_from_json)
  # Main course
  rv = list()
  for v in [obj] if (not isinstance(obj,collections.abc.Iterable) or
                     isinstance(obj,six.string_types)) else obj:
    if isinstance(v,six.string_types):
      for t,f in trial.items():
        try:
          rv.append(f(v))
          break
        except ValueError:
          pass
        except Exception as e:
          print("jsplus: unhandled exception: " + str(e))
          raise(e)
    else:
      rv.append(v)
  return rv[0] if (not isinstance(obj,collections.abc.Iterable) or
                   isinstance(obj,six.string_types)) else rv

def jsdump(obj,fp,**kw):
  """
  Convenience function to serialise a list of objects to file using the
  extended JsonEnc capabilities.

  Parameters
  ----------
  obj : Any
    The object(s) to be serialised.
  fp : a .write()-supporting file-like object that works with str.
    The output file stream to use.
  **kw : Any
    Other keyword parameters to pass to json.dump

  Returns
  -------
  No return value.

  See Also
  --------
  JsonEnc : subclass of JSONEncoder extending serialisation support to include
    datetime, dataframe, series and any other class with a to_json() method.
  jsdumps : convenience function to serialise a list of objects to str
    using the extended JsonEnc capabilities.
  jsload : convenience function to decode objects from file, with jsplus
    support.
  jsloads : convenience function to decode objects from str, with jsplus
    support.

  Examples
  --------
  >>> df = pandas.DataFrame([1,2,3,4])
  >>> ts = datetime.now()
  >>> with gzip.open(filename,'wt') as f: jsdump([df,ts],f)
  """
  json.dump(obj,fp,cls=JsonEnc,**kw)

def jsdumps(obj,**kw):
  """
  Convenience function to serialise a list of objects to str using the
  extended JsonEnc capabilities.

  Parameters
  ----------
  obj : Any
    The object(s) to be serialised.
  **kw : Any
    Other keyword parameters to pass to json.dump

  Returns
  -------
  str for the serialised object(s).

  See Also
  --------
  JsonEnc : subclass of JSONEncoder extending serialisation support to include
    datetime, dataframe, series and any other class with a to_json() method.
  jsdump : convenience function to serialise a list of objects to file
    using the extended JsonEnc capabilities.
  jsload : convenience function to decode objects from file, with jsplus
    support.
  jsloads : convenience function to decode objects from str, with jsplus
    support.

  Examples
  --------
  >>> df = pandas.DataFrame([1,2,3,4])
  >>> ts = datetime.now()
  >>> s = jsdumps([df,ts])
  """
  return json.dumps(obj,cls=JsonEnc,**kw)

def jsload(fp,**kw):
  """
  Convenience function to decode objects from file, with jsplus support.

  Parameters
  ----------
  fp : a .read()-supporting file-like object that works with str.
    The input file stream to use.
  **kw : Any
    Other keyword parameters to pass to json.load

  Returns
  -------
  The decoded object(s).

  See Also
  --------
  JsonEnc : subclass of JSONEncoder extending serialisation support to include
    datetime, dataframe, series and any other class with a to_json() method.
  jsplus : support function to decode specialist json strings.
  jsloads : convenience function to decode objects from str, with jsplus
    support.
  jsdump : convenience function to serialise a list of objects to file
    using the extended JsonEnc capabilities.
  jsdumps : convenience function to serialise a list of objects to str
    using the extended JsonEnc capabilities.

  Examples
  --------
  >>> with gzip.open(jsfile) as f: df,ts = jsload(f)
  """
  rv = json.load(fp,**kw)
  return jsplus(rv)

def jsloads(s,**kw):
  """
  Convenience function to decode objects from str, with jsplus support.

  Parameters
  ----------
  s : str
    The serialisation string to be decoded.
  **kw : Any
    Other keyword parameters to pass to json.load

  Returns
  -------
  The decoded object(s).

  See Also
  --------
  JsonEnc : subclass of JSONEncoder extending serialisation support to include
    datetime, dataframe, series and any other class with a to_json() method.
  jsplus : support function to decode specialist json strings.
  jsload : convenience function to decode objects from file, with jsplus
    support.
  jsdump : convenience function to serialise a list of objects to file
    using the extended JsonEnc capabilities.
  jsdumps : convenience function to serialise a list of objects to str
    using the extended JsonEnc capabilities.

  Examples
  --------
  >>> df = pandas.DataFrame([1,2,3,4])
  >>> ts = datetime.now()
  >>> df1,ts1 = jsloads(jsdumps([df,ts]))
  """
  rv = json.loads(s,**kw)
  return jsplus(rv)

#----------------------------------------------------------------------------

# ModelList - a key structure for model output
class ModelList(dict):
  """
  Primary storage class for model data, parameters and status information.

  The ModelList should be built first as a dictionary or loaded from file.

  Attributes
  ----------
  None

  Methods
  -------
  read_json
  from_json
  to_json

  See Also
  --------
  ModelList.read_json : load previously saved ModelList.
  ModelList.from_json : load previously saved ModelList.
  ModelList.to_json : save ModelList to file.

  Notes
  -----
  For a given <model type> and <model name> the dictionary is structured as:

  +-------+-------+-------+--------------------------------------------+
  |Level 0|Level 1|Level 2|Content                                     |
  +=======+=======+=======+============================================+
  |Index  |<model |table  |dataframe of models and their statuses      |
  +       + type> +-------+--------------------------------------------+
  |       |       |pres   |list of columns in table to present         |
  +       +       +-------+--------------------------------------------+
  |       |       |model  |previously compiled model: dict(python=hash)|
  +-------+-------+-------+--------------------------------------------+
  |<model |Fit    |Code   |str, Stan code                              |
  + name> +       +-------+--------------------------------------------+
  |       |       |Post   |dataframe, fit of model mname, or None      |
  +       +       +-------+--------------------------------------------+
  |       |       |Summary|dataframe, fit summary including diagnostics|
  +       +-------+-------+--------------------------------------------+
  |       |Data           |data used for fit                           |
  +       +-------+-------+--------------------------------------------+
  |       |Args           |args used for fit                           |
  +-------+-------+-------+--------------------------------------------+

  Examples
  --------
  >>> mml = ModelList()
  >>> # Work through model types and their sub-dictionaries (table,pres,model)
  >>> for mt,mtd in mml[Index].items():
  >>> # Test if model exists
  >>> if mmn in mml:
  >>> # Add to Index:
  >>> mml[Index][mt] = {Table:pd.DataFrame(columns=mtv)}
  >>> # For load and save see methods below
  """

  # Specialisation from decoded string
  @staticmethod
  def read_json(s):
    """
    ModelList factory method to read from file or string.
  
    Parameters
    ----------
    s : subclass of io.IOBase or str
      The serialisation stream or string to be decoded.
  
    Returns
    -------
    The decoded ModelList.
  
    See Also
    --------
    ModelList : primary storage class for model data, parameters and status
      information.
    from_json : read from file or string.
    to_json : encode ModelList as json file or string.
  
    Examples
    --------
    >>> with gzip.open(jgz) as jz: mml = ModelList.read_json(jz)
    """
    # Generic interpretation of string
    if isinstance(s,io.IOBase):
      obj = json.load(s)
    elif isinstance(s,six.string_types):
      obj = json.loads(s)
    else:
      raise ValueError("ModelList cannot read json from "+type(obj).__name__)

    # Override native structures where special classes are expected
    for k1,v1 in obj.items():
      #print(k1)
      if k1 == Index:
        for k2,v2 in v1.items():
          #print(k2)
          #print(v2)
          obj[k1][k2][Table] = jsplus(v2[Table])
          # Model and pres are native structures
      else:
        if Fit not in v1: obj[k1][Fit] = dict(Post=None,Code=None,Summary=None)
        obj[k1][Fit][Post] = jsplus(v1[Fit][Post])
        obj[k1][Fit][Summary] = jsplus(v1[Fit][Summary])
        # Astonshingly, pd.Series json split does not do round-trip
        for sn,dt in [('lval','f8'),('datenum','i4'),('missing','i4')]:
          if isinstance(v1[Data][sn],six.string_types):  # json string
            jd = json.loads(v1[Data][sn])  # json dictionary
            obj[k1][Data][sn] = pd.Series(jd['data'],name=jd['name'],
                                          index=jd['index'],dtype=dt)
          else:  # Hopefully it's an ndarray (historical case)
            obj[k1][Data][sn] = pd.Series(v1[Data][sn],name=sn,dtype=dt)
          #  Will not work: pd.read_json(v1[Data][sn],orient='split',dtype=dt)
        # Code and Args are native dict structures
    return obj

  def __init__(self, *args, **kwargs):
    self[Index] = dict()
    self.update(*args, **kwargs)

  def __getitem__(self, key):
    val = dict.__getitem__(self, key)
    return val

  def __setitem__(self, key, val):
    dict.__setitem__(self, key, val)

  # Use standard format for repr
  def __repr__(self):
    drepr = dict.__repr__(self)
    return "{0:s}({1:s})".format(type(self).__name__, drepr)

  def update(self, *args, **kwargs):
    """
    Inherited from ``dict``, along with ``__getitem__``, ``__setitem__`` etc.
    """
    for k, v in dict(*args, **kwargs).items():
      self[k] = v

  # Defining this method delivers encoding via JsonEnc
  def to_json(self,s=None):
    """
    Encode ModelList as json file or string.
  
    Parameters
    ----------
    s : subclass of io.IOBase or None
      The serialisation stream to write to, or None if only the string is
      required.
  
    Returns
    -------
    The json-encoded ModelList as str.
  
    See Also
    --------
    ModelList : primary storage class for model data, parameters and status
      information.
    read_json : ModelList factory method to read from file or string.
    from_json : read from file or string.
  
    Examples
    --------
    >>> with gzip.open(jgz,'wt') as jz: mml.to_json(jz)
    """
    if s is None:
      obj = jsdumps(self)
    elif isinstance(s,io.IOBase):
      # Either all or nothing must be written
      err = None
      try:
        obj = jsdump(self,s)
      except Exception as e:
        #print("ModelList to_json exception: " + str(e))
        err = e
      if err is not None:
        s.close()
        os.unlink(s.name)
        raise err
      #fd,fn = mkstemp()
      #with gzip.open(fn,'wt') as f: obj = jsdump(self,f)
      ## Got this far, so read back what was dumped and put it 
      #with gzip.open(fn) as f:
      #  js = f.read()
      #  s.write(js)
      ## Tidy up
      #fd.close()    # Delayed until here in case it helps prevent fn deletion
      #os.unlink(fn)
    else:
      raise ValueError("ModelList to_json doesn't work with "+type(s).__name__)
    return obj

  # Full decoding from string or file-like
  def from_json(self,s):
    """
    Read ModelList from file or string.
  
    Parameters
    ----------
    s : subclass of io.IOBase or str
      The serialisation stream or string to be decoded.
  
    Returns
    -------
    Self.

    See Also
    --------
    ModelList : primary storage class for model data, parameters and status
      information.
    read_json : ModelList factory method to read from file or string.
    to_json : Encode ModelList as json file or string.

    Notes
    -----
    This method will only overwrite pre-existing model entries with the same
    name.  Use :meth:`ModelList.read_json` to construct a wholly new object.

    Examples
    --------
    >>> with gzip.open(jgz) as jz: mml.from_json(jz)
    """
    # Specialise the generic structure
    for k,v in ModelList.read_json(s).items():
      self[k] = v
    return self

#............................................................................

def hdi(x,lead="Mean",credMass=None,median=True):
  """
  Calculate highest density interval.

  Parameters
  ----------
  x : dataframe, series or list of values
    Data to calculate the HDI for.
  lead : list of str or str or None
    Column(s) or level value on which to base interval.  Non-`lead` columns
    will have their values reported that align to the `lead` column HDI, not 
    their own HDI.  A lead of `None` will treat all columns as `lead`.
  credMass : list of float
    Credible intervals to estimate (in [0,1]).  Default ``[CredMass]``.
  median :  bool
    ``True`` to include median values, else ``False``.

  Returns
  -------
  Series indexed with two extra levels over `x` columns (or `lead`):
  * 'Upper','Lower' and optionally 'Median' (if `median` is ``True``)\
    and 'Mode' (if `credMass` includes ``0.0``).
  * `credMass` values, giving HDI levels.

  Warnings
  --------
  The approach used is not designed to handle multimodal distributions.

  Examples
  --------
  >>> x = [1,2,3,4,5,6,7,8,9,10]
  >>> hdi(x,"X")
  Lower   0.9  X     1.0
  Median  NA   X     5.5
  Upper   0.9  X    10.0
  dtype: float64
  """
  # Prep the data frame to work on
  if credMass is None: credMass = [CredMass]
  if sum([cr > 1 or cr < 0 for cr in credMass]) > 0:
    raise ValueError("Credibility mass must be between 0 and 1")
  credMass.sort()
  # Force lead to be None or iterable (but not string)
  if lead is not None and (not isinstance(lead,collections.abc.Iterable) or
    isinstance(lead,six.string_types)): lead = [lead]
  # Get the data x into a consistent form to simplify the HDI extraction.
  # This means using multiindexes for columns.
  # If x is a data frame, use that but preserve the original
  if isinstance(x,pd.DataFrame):
    top = "DataFrame"
    dx = x.copy()     # To be treated as a read-only view
    if lead is None:  # meaning 'all'
      lead = [top]
    elif sum([l in x.columns for l in lead]) != len(lead):
      raise ValueError("Some lead column(s) "+str(lead)+" not in dataframe")
    xc = [(top,*c) if x.columns.nlevels > 1 else (top,c) for c in list(x)]
    dx.columns = pd.MultiIndex.from_tuples(xc)
  # If x is a series, make it a data frame with MultiIndex so easier later
  elif isinstance(x,pd.Series):
    top = "Series"
    dx = x.to_frame()
    if lead is None:
      lead = [top]
    elif len(lead) > 1:
      raise ValueError("Only one lead column permitted for Series")
    dx.columns = pd.MultiIndex.from_product([[top],lead])
  # Otherwise assume it is a single list or other iterable
  else:
    top = "Data"
    dx = pd.DataFrame(x)
    if lead is None:
      lead = [top]
    elif len(lead) > dx.shape[1]:
      raise ValueError("Too many lead columns for "+type(x).__name__)
    elif dx.shape[1] > 1 and sum([l in dx.columns for l in lead]) != len(lead):
      raise ValueError("Some lead column(s) "+str(lead)+" not in "+dx.columns)
    dx.columns = pd.MultiIndex.from_product([[top],lead])
  # Columns must be ordered for multiindex efficiency and to match rv later
  colord,columner = dx.columns.sortlevel()
  dx = dx.reindex(columns=colord)

  # Find the level corresponding to the lead column names - must be only one
  for clv in range(dx.columns.nlevels):
    if sum([l in dx.columns.get_level_values(clv) for l in lead]) == len(lead):
      break
  if clv >= dx.columns.nlevels:
    raise ValueError("Lead column(s) not in same level")

  # Set up credivals
  lenx = dx.shape[0]
  mode = False
  if 0 in credMass:
    credMass = [cr for cr in credMass if cr != 0]
    mode = True
  credival = [int(np.ceil(cr*lenx)) for cr in credMass]

  # Prep return data - using rows woudl need .loc to access - ensure lex order
  hcols = [('Lower',-cr) for cr in credMass[::-1]] +\
          ([('Median','NA')] if median else []) +\
          ([('Mode',0.0)] if mode else []) +\
          [('Upper',cr) for cr in credMass]
  mit = [(*h,*c[1:]) for h in hcols for c in dx.columns]
  miln = ['level','credMass'] + dx[top].columns.names
  rv = pd.Series(index=pd.MultiIndex.from_tuples(mit,names=miln),dtype='f8')
  # Sort the multiindex to avoid performance warnings
  roword,indexer = rv.index.sortlevel()
  rv = rv.reindex(index=roword)

  ## Not tested with multiple lead columns...
  # For each subcolumn, sort the data and identify range sizes
  for ct in ((*i,dx.xs(l,level=clv,axis=1).columns.nlevels)
             for l in lead for i in dx.xs(l,level=clv,axis=1).items()):
    c,ds,n = ct
    if n == 1: c = (c,) # Get c as consistent tuple even if 1 column
    so = ds.sort_values().index
    t = tuple([cc for cc in c[:clv]] + [slice(None)] + [cc for cc in c[clv:]])
    # Allow for some features to be absent from dx (missing indices)
    f = [ft[1:] for ft in dx.loc[:,t].columns]

    for cri in range(len(credMass)):
      # Scan through the series for optimal range of noted size
      wi = np.argmax(ds[so].diff(-credival[cri])) # -ve diff for lower, so max
      wl,wu = (so[wi],so[wi+credival[cri]])

      # Construct the return data frame specifying the HDI bounds
      rv[[('Lower',-credMass[cri],*ft) for ft in f]] = dx.loc[wl,t].values
      rv[[('Upper',credMass[cri],*ft) for ft in f]] = dx.loc[wu,t].values
    if median:
      medvals = dx.loc[[so[int((lenx-1)/2)],so[int((lenx)/2)]],t].mean()
      rv[[('Median','NA',*ft) for ft in f]] = medvals.values
    if mode:
      modevals = dx.loc[np.min(ds[so].diff(1)),t].values
      rv[[('Mode',0.0,*ft) for ft in f]] = modevals
  # Finally, fix signage and restore order
  rv.index = pd.MultiIndex.from_tuples([(t[0],-t[1],*t[2:])
               if t[0] == 'Lower' else (t[0],t[1],*t[2:]) for t in rv.index])
  roword = np.array(rv.index)      # To get the right type of array
  roword[indexer] = rv.index       # Reverse the earlier sort
  return rv.reindex(index=roword)  # Return the reordered result

#............................................................................

def lrfit(mr,dr,cv=1):
  r"""
  Linear regression fit against a negatively curved quadratic.

  Parameters
  ----------
  mr : slice
    Specifies indices of `dr` to be used.
  dr : dataframe
    Containing 'x' independents and 'y' data.
  cv : int 0 or 1
    1 for constant variance, 0 for variable (shape times mean).

  Returns
  -------
  list of variance, coefficients :math:`\times 3`, standard deviation.

  Examples
  --------
  >>> import pandas as pd, numpy as np
  >>> dr = pd.DataFrame(dict(x=np.arange(100),y=-np.arange(100)**2))
  >>> lrmr = [2,75]
  >>> lrfit(slice(*lrmr),dr)
  [1e-05, -5.400124791776761e-13, -1.971756091734278e-12, -0.999999999999967,\
  0.0031622776601683794]
  """
  # Choice of packages:  scipy.stats, sklearn.linear_model, or statsmodels.api.
  # Settled on the last because of better pandas integration and R support.

  ds = dr.iloc[mr]  # Get view of the slice required
  lrm = smf.ols("y~x+np.power(x,2)",data=ds).fit()
  # If the result is the wrong way up, use a linear fit and a small fix
  if lrm.params.iat[2] > 0: lrm = smf.ols("y~x",data=ds).fit()
  # Get a consistent set of coefficients including the small fix for a
  lrcoef = np.append(lrm.params,[-Tol])[0:3]
  # Apply restraints - in particular, c-b^2/4a cannot be more than about 27
  if (lrcoef[0] - np.power(lrcoef[1],2)/(4*lrcoef[2])) > 30:
    lrcoef = np.array([ds.y.mean(),Tol,-Tol])

  # For variable variance model, estimate shape from the raw residuals
  if cv == 0:
    lrem = 1
  # Otherwise, for constant SD, use a safe version of the residuals
  else:
    lx = len(range(*mr.indices(dr.shape[0])))
    lre = np.sum(lrcoef*np.transpose(np.array([np.full(lx,1),ds.x,ds.x**2])))
    lrem = np.abs(lre)             # Magnitude of estimated ds.y...
    #lres = lrm.resid*np.exp(lrem)  # ...made safe
    #lres = lrm.resid*lrem          # ...made safe

  # Get the variance of the residuals and ensure it is above 0
  v = np.var(lrm.resid)*(lrem*lrem)
  if v < Tol: v = Tol

  # Return a list of values in known order: variance, coefficients, shape
  return [v] + list(lrcoef) + [np.sqrt(v)]


#----------------------------------------------------------------------------
# Posterior data processing functions

def str2mi(d,pars,gqs,before,after,axis=0):
  """
  Convert PyStan index or column with string labels to MultiIndex.

  Parameters
  ----------
  d : dataframe
    The dataframe whose index or columns are to be converted
  pars : list
    The names of the parameters to convert.
  gqs : list
    The names of generated quantities to convert (treated same as pars).
  before: str
    The level 0 name for labels prior to any in pars or gqs.
  after: str
    The level 0 name to use for labels found after pars and gqs.
  axis : int
    0 to convert index, 1 to convert columns.

  Returns
  -------
  Dataframe with new axis labels.

  Examples
  --------
  >>> #Where post includes columns 'a[1]','a[2]','b[1]' and 'b[2]', 'mu[1]' etc.
  >>> ppars = ("a","b")
  >>> gqs = ("mu","sigma","peak")
  >>> post = fit.to_dataframe(inc_warmup=False)
  >>> post = str2mi(post,ppars,gqs,'Indices','Diagnostics',axis=1)
  >>> #post now has MultiIndex columns, level 0 includes 'a','b', level 1 1,2.
  """
  pl = list()
  cl = list()
  diag = 0
  for c in d.axes[axis]:
    if c[-1] == "]":
      for p in pars + gqs:
        if c[0:(len(p)+1)] == p + "[":
          pl = pl + [(p,int(c[(len(p)+1):-1]))]
          diag = -1
    if diag > 0:
      pl = pl + [(after,c)]
    elif diag == 0:
      pl = pl + [(before,c)]
    elif diag < 0:
      diag = -diag
  # Rename first to keep data
  rv = d.rename(mapper=dict([(d.axes[axis][i],pl[i]) for i in range(len(pl))]),
                axis=axis)
  # Then uplift to multiindex
  return(rv.reindex(labels=pd.MultiIndex.from_tuples(pl),axis=axis))

#............................................................................

def mywaic(samsize,d,mml,review=True):
  """
  Calculate WAIC for model comparison.

  Parameters
  ----------
  d : dataframe
    With 'datenum','lval', and 'missing' columns.
  mml : ModelList
    The models to compare.
  review : bool
    False to check if model data is current, True to only check if code is.

  Returns
  -------
  Dataframe sorted by WAIC value and listing for each qualifying model:
  * WAIC: widely applicable information criterion value.
  * pWAIC: estimated number of effective parameters.
  * dWAIC: difference between each WAIC and the lowest WAIC.
  * weight: Akaike estimate of probability a model makes best predictions.
  * SE: standard error of WAIC estimate.
  * dSE: standard error of difference in WAIC with top-ranked model.

  Examples
  --------
  >>> dw = mywaic(1000,dd,mml,False)
             WAIC    pWAIC    dWAIC  weight       SE     dSE
  m2.1.0     7.07    10.70     0.00     1.0    19.05    1.96
  m2.0.0    32.85    10.80    25.79     0.0    20.35    1.92
  m3.1.0    80.43    52.55    73.36     0.0    73.21    7.23
  m2.1.1    90.48    12.25    83.41     0.0    27.89    2.69
  m2.0.1    90.72    11.08    83.65     0.0    28.28    2.71
  m1.1.1   129.06     6.02   121.99     0.0    26.37    2.47
  m1.1.0   183.47     7.28   176.41     0.0    15.66    0.00
  """
  # Globals:  ModelCode = list of models as code

  # Set up target table
  WAIC,pWAIC,dWAIC,weight,SE,dSE =('WAIC','pWAIC','dWAIC','weight','SE','dSE')
  dw = pd.DataFrame(columns=(WAIC,pWAIC,dWAIC,weight,SE,dSE),dtype='f8')
  ddw = pd.DataFrame(index=d.index)
  mwa = list()
  mwi = 0

  # Build list of models to be compared
  mlist = []
  geometric = 0
  for mt,mtd in mml[Index].items():
    dm = mtd[Table]
    #dm = pd.DataFrame(mt=character(),I=integer(),S=integer(),C=integer())
    for mi in range(len(dm)):
      # Convenient references
      mp = dm.iloc[mi,:]       # model parameters
      mmn = dm.index[mi]       # model name

      # Check this model exists
      if mmn not in mml: continue  # Silently ignore (already reported)
      mm = mml[mmn]            # list of model fit and data

      # Check if model qualifies
      if mm[Fit][Post] is None or mm[Fit][Post].shape[0] == 0:
       continue                                      # No fit available
      if not review and not (
        mm[Data]['M'] == d.missing.sum() and
        mm[Data]['N'] == d.lval.shape[0] and
        mm[Data]['missing'].equals(pd.Series(name='missing',dtype='i4',
          data=d.missing.reset_index().index[d.missing].values+1)) and
        mm[Data]['datenum'].equals(d.datenum) and
        mm[Data]['lval'].round(6).equals(d.lval.round(6))):
        continue                                     # Data changed
      if mm[Fit][Code] != ModelCode[mt]: continue    # Code changed

      # Add model to list for WAIC processing
      mlist.append((mmn,mp))

      # Check if geometric distribution applies to any model
      if 'G' in mp and mp.G != 0: geometric = 1 # Apply to all WAIC calculations

  # For each qualifying model, compute WAIC
  for mmn,mp in mlist:
    # Get sampled parameters and lose warmup
    p = mml[mmn][Fit][Post].loc[(mm[Fit][Post].Indices.warmup == 0.0),:]

    # Obtain log likelihoods by using model to generate sample probabilities
    # Will need mt to set correct ll when mt not m
    ll = gencases(samsize,d,p,constantSD=mp.C,geometric=geometric).LL

    # Calculate WAIC for this model:  lppd and pwaic get overwritten per loop
    ddw['lppd'] = np.log(np.exp(ll).mean())
    ddw['lppd'].fillna(ddw.lppd.min()+Unlikely)
    ddw['pwaic'] = ll.var()
    ddw['pwaic'].fillna(ddw.pwaic.max()-Unlikely)
    ddw[mmn] = -2*(ddw.lppd - ddw.pwaic)  # Retained across loops

    # Save to table
    dw.loc[mmn,"WAIC"] = ddw[mmn].sum()
    dw.loc[mmn,"SE"] = np.sqrt(ddw.shape[0]*ddw[mmn].var())
    dw.loc[mmn,"pWAIC"] = ddw.pwaic.sum()

  # Perform comparisons
  if dw.shape[0] > 0:
    dw.sort_values(by='WAIC',inplace=True)
    if 'lppd' in ddw.columns: ddw.drop(['lppd','pwaic'],axis=1,inplace=True)
    dw['dWAIC'] = dw.WAIC - dw.WAIC.iat[0]
    dw['weight'] = np.exp((-dw.dWAIC/2))
    dw['weight'] = dw.weight/dw.weight.sum()
    ddw = ddw[dw.index]  # match ddw column order with revised dw row order
    dw['dSE'] = ddw.subtract(ddw.iloc[:,0],axis=0).std()

  # Return the table
  return dw

#............................................................................

def precis(post,credMass=None,fitsum=None):
  """
  Display posterior sample data in neat form.

  Parameters
  ----------
  post : dataframe
    The posterior data without warmup and with MultiIndex columns.
  credMass : list of float
    The HPDI credivals to display.  Default ``[CredMass,0.5]``.
  fitsum: dataframe or None
    The summary fit table from PyStan, must include 'n_eff' and 'Rhat'.

  Returns
  -------
  Dataframe with 'Statistics' level 0 column containing 'Mean', 'SE_Mean',\
  and 'SD' level 1 columns;  'Upper', 'Lower' and 'Median' level 0 columns\
  with credMass value level 1 columns;  and 'Diagnostics' level 0 column\
  with 'n_eff' and 'Rhat' level 1 columns, if a suitable fitsum was passed.\
  Index is taken from post parameter MultiIndex columns.

  See Also
  --------
  implications : tabulate model implications using HDI.
  implications_quantile : tabulate model implications using quantiles.

  Notes
  -----
  'Median' is always included, and 'Mode' if credMass includes 0.

  Examples
  --------
  >>> precis(post,fitsum=mml[mmn][Fit][Summary])
        Statistics                  Lower          Median   Upper          \
              Mean SE_Mean     SD     0.9     0.5      NA     0.5     0.9   
  a   1       0.00    0.00   0.00    0.00    0.00    0.00    0.00    0.00   
  b   1       0.13    0.00   0.01    0.12    0.12    0.13    0.13    0.13   
  c   1      -4.81    0.00   0.15   -5.06   -4.92   -4.81   -4.71   -4.56   
  .
        Diagnostics       
              n_eff Rhat  
  a   1    20414.57 1.00  
  b   1    19441.05 1.00  
  c   1    20805.49 1.00  
  """
  if credMass is None: credMass = [CredMass,0.5]
  tpars = post.columns[~post.columns.isin(['Indices','Diagnostics'],level=0)]
  statl = ['Mean','SE_Mean','SD']
  stats,diag = ('Statistics','Diagnostics')
  mix = pd.MultiIndex.from_tuples([(stats,st) for st in statl])
  dp = pd.DataFrame(index=tpars,columns=mix)
  dp[(stats,'Mean')] = post.loc[:,tpars].mean().values
  dp[(stats,'SE_Mean')] = post.loc[:,tpars].sem().values
  dp[(stats,'SD')] = post.loc[:,tpars].std().values
  dh = hdi(post[tpars],lead=None,credMass=credMass,median=True)
  for c in dh.index: dp.loc[c[2:],c[:2]] = dh[c]
  if fitsum is not None and 'n_eff' in list(fitsum) and 'Rhat' in list(fitsum):
    dp[[(diag,'n_eff'),(diag,'Rhat')]] = fitsum[['n_eff','Rhat']]
  return dp

#............................................................................

def implications(post,credMass=None,median=True):
  """
  Tabulate model implications using HDI.

  Parameters
  ----------
  post : dataframe
    Posterior sample without warmup and with MultiIndex columns.
  credMass : list of float
     Credibility intervals.
  median : boolean
     True to include 'Median', False to exclude.

  Returns
  -------
  Dataframe with 'Peak', 'Total', 'Duration', 'Start date', 'Peak date',\
  and 'End date' columns.  Index is a MultiIndex with 'Lower' and 'Upper'\
  level 0 values along with 'Median' if requested and 'Mode' if `credMass`\
  includes 0;  level 1 specifies the relevant `credMass` value.

  See Also
  --------
  precis : display posterior sample data in neat form.
  implications_quantile : tabulate model implications using quantiles.

  Examples
  --------
  >>> implications(post)
               Peak  Total  Duration   Start date    Peak date     End date
  Level  Cred                                                              
  Lower  0.9   5056 280539       190  23-Jan-2020  27-Apr-2020  31-Jul-2020
         0.5   5195 281565       185  25-Jan-2020  27-Apr-2020  29-Jul-2020
  Median NA    5297 295113       191  22-Jan-2020  27-Apr-2020  01-Aug-2020
  Upper  0.5   5396 293882       187  24-Jan-2020  27-Apr-2020  29-Jul-2020
         0.9   5551 306286       190  22-Jan-2020  26-Apr-2020  30-Jul-2020
  """
  if credMass is None: credMass = [CredMass,0.5]
  # Model knowledge
  iv = post.a.columns.max()
  pc = ['mu','sigma','peak'] + (['mud'] if 'mud' in post.columns else [])
  ic = ['lim','offs','sdate','edate']
  rc = ['Peak','Total','Duration','Start date','Peak date','End date']

  # Get the HDI of peak values, pulling all other columns along
  ## This version aimed at preventing fascination with artificial density
  #pkseli = np.array([post.peak[i] < Worldpop for i in range(1,iv+1)])
  #pksel = np.sum(pkseli,axis=0) == iv
  #dh = hdi(post.loc[pksel,pc],lead='peak',credMass=credMass,median=median)
  dh = hdi(post[pc],lead='peak',credMass=credMass,median=median)
  rix = pd.MultiIndex.from_tuples(pd.unique([c[:2] for c in dh.index]))

  # Issues with dtype make it best to deal only with floats and convert to
  # dates at one go when the calculations are done.
  hdicols = [c for c in ['Lower','Median','Mode','Upper'] if c in dp.index]
  mix = pd.MultiIndex.from_tuples(list(pd.unique([c[2:] for c in dh.index]))
                              + [(p,i) for p in ic+rc for i in range(1,iv+1)])
  mix,indexer = mix.sortlevel()  # Needed for slicing later
  di = pd.DataFrame(index=rix,columns=mix,dtype='f8')
  for c in dh.index: di.loc[c[:2],c[2:]] = dh[c]

  #di[['mu','sigma','peak']] = dp[['mu','sigma','peak']].values
  # Find where only half a case occurs (safely: allow for peak < 0.5)
  di['lim'] = (0.5/(0.5+di['peak'])).values
  di['offs'] = (np.sqrt(-2*np.log(di['lim'].values))*di['sigma'].values)
  # Half a case provides beginning and end of overall range
  di[('sdate',1)] = (di[('mu',1)] - di[('offs',1)]).values
  di[('edate',iv)] = (di[('mu',iv)] + di[('offs',iv)]).values
  if iv > 1:
    di.loc[:,('sdate',slice(2,iv))] = di['mud'].values
    di.loc[:,('edate',slice(1,iv-1))] = di['mud'].values
  # Mixing credivals and offsets can make start > end: mud is more reliable
  di.loc[di[('edate',iv)] < di[('sdate',iv)],('edate',iv)] =\
    di.loc[di[('edate',iv)] < di[('sdate',iv)],('sdate',iv)]
  for i in range(1,iv):
    di.loc[di[('sdate',i)] > di[('edate',i)],'sdate'] =\
      di.loc[di[('sdate',i)] > di[('edate',i)],'edate'].values
  di['Duration'] = (di['edate'] - di['sdate']).values
  for i in range(1,iv+1):
    dmp = di.xs(i,level=1,axis=1).apply(lambda r: r.peak*np.exp(
             -(np.arange(r.sdate,r.edate+1)-r.mu)**2/(2*r.sigma**2)),axis=1)
    di.loc[:,(['Peak','Peak date','Total'],i)] = dmp.apply(lambda r:
             [np.max(r),np.argmax(r),np.sum(r)]).to_list()
  
  # Construct return dataframe
  dr = pd.DataFrame(index=di.index,columns=rc)
  dr.index.set_names(['Level','Cred'],inplace=True)
  dr['Peak'] = di['Peak'].max(axis=1)
  dr['Total'] = di['Total'].sum(axis=1)
  dr['Start date'] = md + pd.TimedeltaIndex(di[('sdate',1)],'D')
  for ix,i in di['Peak'].idxmax(axis=1).items():
    dr.loc[ix,'Peak date'] = (md +
          pd.Timedelta(di.loc[ix,('sdate',i)]+di.loc[ix,('Peak date',i)],'D'))
  dr['End date'] = md + pd.TimedeltaIndex(di[('edate',iv)],'D')
  dr['Duration'] = pd.TimedeltaIndex(dr['End date'] - dr['Start date']).days
  # Alas, there is no date format option available, so have to resort to str
  dtfmt = '%d-%b-%Y'
  dr['Start date'] = pd.DatetimeIndex(dr['Start date']).strftime(dtfmt)
  dr['Peak date'] = pd.DatetimeIndex(dr['Peak date']).strftime(dtfmt)
  dr['End date'] = pd.DatetimeIndex(dr['End date']).strftime(dtfmt)
  return dr

#............................................................................

# di is a possibly useful intermediate dataframe for per-intervention stats.
def implications_quantile(post,qs=(0.025,0.25,0.5,0.75,0.975)):
  """
  Tabulate model implications using quantiles.

  Parameters
  ----------
  post : dataframe
    Posterior sample without warmup and with MultiIndex columns.
  qs : list of float
    Quantiles to include.

  Returns
  -------
  Dataframe with 'Peak', 'Total', 'Duration', 'Start date', 'Peak date',\
  and 'End date' columns.  Index lists quantiles requested.

  See Also
  --------
  precis : display posterior sample data in neat form.
  implications : tabulate model implications using HDI.

  Notes
  -----
  This function is retained for historical comparison only.  Most of the
  panstan capability is based upon HDI and not quantiles.  Differences
  against the HDI result will include:
  * Minor variations in Peak Median owing to different methods being used.
  * Wider variations in other columns, as the HDI version pulls these values
  along with the Peak HDI, whereas quantiles are displayed per column.
  * A 0.95 HDI will only match quantiles 0.025 and 0.975 if the density is
  strictly symmetric.

  Examples
  --------
  >>> implications_quantile(post)
               Peak  Total  Duration   Start date    Peak date     End date
  Quantile                                                                    
  0.025        4998 260368       178  27-Jan-2020  25-Apr-2020  23-Jul-2020
  0.25         5195 280301       185  25-Jan-2020  27-Apr-2020  28-Jul-2020
  0.5          5296 291121       188  23-Jan-2020  26-Apr-2020  30-Jul-2020
  0.75         5397 302468       192  22-Jan-2020  27-Apr-2020  02-Aug-2020
  0.975        5592 325212       200  19-Jan-2020  28-Apr-2020  07-Aug-2020
  """
  iv = post.a.columns.max()
  qss = [str(q) for q in qs]
  ic = ['mu','sigma','peak','lim','offs','sdate','edate']
  rc = ['Peak daily','Total','Duration','Start date','Peak date','End date']

  # Issues with dtype make it best to deal only with floats and convert to
  # dates at one go when the calculations are done.
  mix = pd.MultiIndex.from_product([range(1,iv+1),qss],names=['I','Quantile'])
  di = pd.DataFrame(index=mix,columns=ic+rc,dtype='f8')
  for i in range(1,iv+1):
    di.loc[i,'mu'] = post.mu[i].quantile(qs).values
    di.loc[i,'sigma'] = post.sigma[i].quantile(qs).values
    di.loc[i,'peak'] = post.peak[i].quantile(qs).values  # see pt below
    # Start/End = latest/earliest cases <= 0.5 before/after peak
    di.loc[i,'lim'] = (0.5/(0.5+di.loc[i,'peak'])).values
    di.loc[i,'offs'] = (np.sqrt(-2*np.log(di.loc[i,'lim'].values))*
                                di.loc[i,'sigma']).values
    if i == 1:
      di.loc[i,'sdate'] = (di.loc[i,'mu'] - di.loc[i,'offs']).values
    else:
      di.loc[i,'sdate'] = post.mud[i-1].quantile(qs).values
    # Mixing quantiles and offsets can make start > end: mud more reliable
    if i == iv:
      di.loc[i,'edate'] = (di.loc[i,'mu'] + di.loc[i,'offs']).values
      di.loc[i,'edate'][di.loc[i,'edate'] < di.loc[i,'sdate']] = \
        di.loc[i,'sdate'] + 1
    else:
      di.loc[i,'edate'] = post.mud[i].quantile(qs).values
      di.loc[i,'sdate'][di.loc[i,'sdate'] > di.loc[i,'edate']] = \
        di.loc[i,'edate'] - 1
    dmp = di.loc[i,:].apply(lambda r: r.peak*np.exp(
             -(np.arange(r.sdate,r.edate+1)-r.mu)**2/(2*r.sigma**2)),axis=1)
    di.loc[i,['Peak date','Peak daily','Total']] = dmp.apply(lambda r:
             [np.argmax(r),np.max(r),np.sum(r)]).to_list()
    di.loc[i,'Duration'] = (di.loc[i,'edate'] - di.loc[i,'sdate']).values

  # Construct return dataframe
  dr = pd.DataFrame(index=qss,columns=rc)
  dr.index.set_names('Quantile',inplace=True)
  dr['Peak daily'] = di['Peak daily'].max(level=1)
  dr['Total'] = di['Total'].sum(level=1)
  dr['Start date'] = md + pd.TimedeltaIndex(di.loc[1,'sdate'],'D')
  dr['Peak date'] = md + pd.TimedeltaIndex(
   di['sdate'].loc[di['Peak daily'].groupby(level=1).idxmax()] +
   di['Peak date'].loc[di['Peak daily'].groupby(level=1).idxmax()],'D')
  dr['End date'] = md + pd.TimedeltaIndex(di.loc[iv,'edate'],'D')
  dr['Duration'] = pd.TimedeltaIndex(dr['End date'] - dr['Start date']).days
  # Alas, there is no date format option available, so have to resort to str
  dtfmt = '%d-%b-%Y'
  dr['Start date'] = pd.DatetimeIndex(dr['Start date']).strftime(dtfmt)
  dr['Peak date'] = pd.DatetimeIndex(dr['Peak date']).strftime(dtfmt)
  dr['End date'] = pd.DatetimeIndex(dr['End date']).strftime(dtfmt)
  return dr

#............................................................................
def inv_logit(x):
  """
  Returns logistic of vector provided.

  The result is calculated in a way to avoid numpy warnings of overflow.

  Parameters
  ----------
  x : numpy array or pandas Series
    Real data values.

  Returns
  -------
  Same type as `x` with logistic values.

  Examples
  --------
  >>> inv_logit(np.arange(-100,100,20))
  array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.06115369e-09, 5.00000000e-01, 9.99999998e-01, 1.00000000e+00,
         1.00000000e+00, 1.00000000e+00])
  """
  return 0.5*(1. + np.sign(x)*(2./(1. + np.exp(-np.abs(x))) - 1.))

def inv_logit_scaled(x):
  """
  Returns `x` times the logistic of `x`, using :func:`inv_logit`.

  Parameters
  ----------
  x : numpy array or pandas Series
    Real data values.

  Returns
  -------
  Same type as `x` with scaled logistic values.

  Examples
  --------
  >>> inv_logit_scaled(np.arange(-100,100,20))
  array([-0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         -4.12230738e-08,  0.00000000e+00,  2.00000000e+01,  4.00000000e+01,
          6.00000000e+01,  8.00000000e+01])
  """
  return x*inv_logit(x)

#............................................................................
def gencases(samsize,dd,post,constantSD=0,geometric=0):
  """
  Generate sample case data.

  Parameters
  ----------
  samsize : float
    Number of samples of parameters to return.
  dd : dataframe
    Contains 'datenum', and optionally 'lval' and 'missing' to get log
    likelihood.
  post : dataframe
    From PyStan to_dataframe() with :func:`str2mi` MultiIndex columns
    containing a,b,c,mud and sigmam.
  constantSD : int 0 or 1
    0 for variable SD (default) or 1 for constant SD.

  Returns
  -------
  Dataframe with 'Mean', 'SD' and optional 'LL' level 0 columns with 'datenum'\
  from dd forming level 1 of a MultiIndex.
  'Mean' and 'SD' can be used to construct a Gaussian spline for each 'datenum'.

  Examples
  --------
  >>> post = fit.to_dataframe(inc_warmup=False)
  >>> post = str2mi(post,ppars,gqs,'Indices','Diagnostics',axis=1)
  >>> dr = pd.DataFrame(dict(datenum=np.arange(0,100)),dtype='i4')
  >>> cases = gencases(1000,dr,post,constantSD=1)
  """
  # The easier route is to vectorise by datenum, figuring out for each post
  # row how all the datenums relate.  Even this is quite involved, building
  # a set of parameters from the post row for each datenum.
  # More efficient is vectorisation by post, i.e. by parameter estimate.
  # But both are slow, so vectorisation should be complete.
  # The problem is with variable mud: the columns needed are ~per datenum.
  # It's straightforward to identify which set of parameters applies to a dn.
  # Input is dd which includes datenum and if ll needed also lval.

  # NB the LL is worked out in log world, and then Mean and SD are converted
  # to real world:  slightly more work, but probably a better result.

  # First, get the sample size right and select the useful parameters
  samb = ~post.columns.isin(['Indices','Diagnostics'],level=0)
  sam = post.loc[(post.Indices.warmup == 0.0),samb]
  sam.columns = pd.MultiIndex.from_tuples(sam.columns) # Clean up columns mi
  samlen = sam.shape[0]
  # If samsize is large enough, ensure all of post is used, via non-replacement
  sl = [sam.sample(samlen) for x in range(samsize//samlen)]
  sam = pd.concat(sl+[sam.sample(samsize-len(sl)*samlen)])
  ##! Remove following
  if samsize != sam.shape[0]: raise Exception("pd.concat didn't work!")
  # Set up the parameters table from sam
  pars = np.array(['a','b','c','sigmam'])  #sam.columns.levels[0]
  pmix = pd.MultiIndex.from_product([pars,dd.datenum],names=['par','datenum'])
  par = pd.DataFrame(index=sam.index,columns=pmix)
  par1 = pars[[sam[p].shape[1] == 1 for p in pars]] # e.g. sigmam & S=0, or I=1
  parn = pars[[sam[p].shape[1] > 1 for p in pars]]
  dnlen = par[pars[0]].shape[1]
  ## How to extend a dataframe by empty columns:
  # part = [('par',p) for p in pars]
  # sam = sam.reindex(columns=sam.columns.to_list() + part,copy=False)
  # Initialise par anticipating first parameter will dominate
  ##! This is clumsy...
  par[pars] = pd.concat([s for l in [[s]*dnlen for n,s in
                   sam.loc[:,(pars,1)].items()] for s in l],axis=1).values
  sam[('tmp','i')] = 1
  sam[('mud',sam.a.shape[1])] = np.inf
  # For each datenum in numerical order:
  # In practice, some gain by grouping datenums into common post sets.
  # If there is more than one of any parameter, has to be handled per row
  for dn in dd.datenum:
    sami = (dn >= sam.mud).sum(axis=1).astype('i8') + 1
    if sam.tmp.i.eq(sami).sum() != sami.shape[0]: # Not same pars as last time
      #print("Calculating parameters for datenum " + str(dn))
      sam[('tmp','i')] = sami
      ##! This method is fast but is np.choose faster?
      for i in sam.mud.columns[sam.mud.columns != 1]:  # 1 was the default
        par.loc[sam.tmp.i == i,(parn,dn)] = \
                                 sam.loc[sam.tmp.i == i,(parn,i)].values
    elif sam.tmp.i.eq(1).sum() != sami.shape[0]:  # Some pars not the first
      par.loc[:,(parn,dn)] = par.loc[:,(parn,dn-1)].values
  
  # Set up the output structure
  values = ['Mean','SD'] + (['LL'] if 'lval' in list(dd) else [])
  dmix = pd.MultiIndex.from_product([values,dd.datenum],names=['v','datenum'])
  dr = pd.DataFrame(index=sam.index,columns=dmix)
  if 'LL' in values:
    # Log likelihood: if needed, set up copies of values to be assessed
    dr['LL'] = pd.concat([dd.lval.to_frame().T]*samsize).values
    # Replace missing values with imputed ones before gauging likelihood
    if 'missing' in list(dd):
      dr.loc[:,('LL',dd.datenum[dd.missing])] = sam.logdiffi.values

  # Calculate outputs
  # This is pandasic:
  #x = pd.DataFrame(pd.concat([dd.datenum.to_frame().T]*samsize).values,
  # But this has to be more efficient:
  x,x2 = [pd.DataFrame(dx.to_frame().T.values.repeat(samsize,axis=0),index=
         par.index,columns=par.a.columns) for dx in [dd.datenum,dd.datenum**2]]
  # Annoyingly, rhs must have same multiindex as lhs even when slices match.
  # But still, it doesn't completely spoil the benefit from all the above.
  dr['Mean'] = (-par.c +par.b*x -par.a*x2).values
  # For SD, use logistic to tame extreme means (strongly negative)
  rwmean = np.exp(inv_logit_scaled(dr.Mean))
  dr['SD'] = par.sigmam/(constantSD*rwmean+1-constantSD) # ok to plot +/-1SD

  # Produce the log likelihood in if needed
  if 'LL' in values:
    if geometric == 0:
      dr['LL'] = norm.logpdf(dr.LL,loc=dr.Mean,scale=dr.SD)
    else:
      # Required to account for runs of points above/below Mean
      dz = (dr.LL > dr.Mean).astype('i4') -(dr.LL < dr.Mean).astype('i4')
      dnz = -dz.diff(-1,axis=1).fillna(-1)
      # Difficult to avoid python loop with diff of nonzero
      #dnz.where(dnz != 0,
      #dd.apply(lambda r: np.diff(np.insert(np.nonzero(r.values),0,-1))),axis=1)
      #di = dd.where(dd == 0,list(dd.T.reset_index().T))
      for ix,r in dnz.iterrows():
        r[r != 0] = np.diff(np.insert(np.nonzero(r.values),0,-1))

      dr['LL'] = norm.logpdf(dr.LL,loc=dr.Mean,scale=dr.SD) +\
                 nbinom.logpmf((dnz*dz).abs(),1,0.5)

    # Precaution: make 0 probability finitely unlikely and fill NaNs
    with pd.option_context('mode.use_inf_as_null', True):
      dr['LL'] = dr['LL'].fillna(dr.LL.min().min()+Unlikely)

  # Now convert to real world from log scale
  dr['Mean'] = rwmean
  dr['SD'] = dr.SD*rwmean
  return dr

#----------------------------------------------------------------------------
# Plot functions

# This utility no longer works as in R, which disables useful functionality
def pdfplot(fig=None,pdfile=None,prompt=None):
  """
  Convenience function to display a previously built figure with notice
  and optional save to PDF.

  Parameters
  ----------
  fig : matplotlib.figure.Figure
    The figure to be displayed.
  pdfile : str
    Name of the PDF file to save the figure to.
  prompt : str
    The notice to display along with the figure (no input is expected).

  Returns
  -------
  None.

  See Also
  --------
  traceplot: display variability and trend in draws made by MCMC.
  pairs : display scatter plots of each parameter value against each other.
  precis : tabulate parameter estimate statistics.
  hpdiplot : display model results as an epidemic curve.

  Examples
  --------
  >>> pdfplot(prompt="trace plot",pdfile=cpdf,
  ...         fig=traceplot(post,ppars,(detail1,detail2))[0])
  Displaying trace plot
  """
  if prompt is not None:
    print("Displaying " + prompt)
  if fig is not None:
    plt.pause(1) # Side effect is show()
    if pdfile is not None: fig.savefig(pdfile,bbox_inches='tight',format='pdf')
    #plt.waitforbuttonpress(timeout=0)
    #plt.ginput(n=1,timeout=0,show_clicks=False)
    #plt.close()

def traceplot(post,pars,details):
  """
  Display variability and trend in draws made by MCMC.

  Parameters
  ----------
  post : dataframe
    As extracted from Pystan fit and should contain warmup.
  pars : list
    Only lists parameter names not indices e.g. 'a' but not 'a[1]'.
  details : list of two strings
    [0] is used with title;  [1] is captioned at bottom of figure.

  Returns
  -------
  tuple of figure and axes.

  See Also
  --------
  pairs : display scatter plots of each parameter value against each other.
  precis : tabulate parameter estimate statistics.
  hpdiplot : display model results as an epidemic curve.

  Examples
  --------
  >>> post = fit.to_dataframe(inc_warmup=True)
  >>> detail1 = "Model 2"
  >>> detail2 = "10000 warmup, 20000 iterations, 3 chains"
  >>> ppars = ("a","b","c")
  >>> fig,ax = traceplot(post,ppars,(detail1,detail2))
  """
  pl = post.columns[post.columns.isin(pars,level=0)]
  lp = min(56,pl.shape[0])  ## Limit to 8x7 display
  slp = np.sqrt(lp)
  h,w = (int(np.ceil(slp)),int(np.floor(slp)))
  if w*h < lp: h += 1  # Worst case is n^2-1; (n-1)(n+1) = n^2-1  -- enough
  chains = post.Indices.chain.unique()
  fig,ax = plt.subplots(h,w)
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt.subplots_adjust(left=0.07,bottom=0.08,right=0.97,top=0.9,
                      wspace=0.25,hspace=0.4)
  fig.suptitle("Trace plot " + (details[0] if len(details) > 0 else ""))
  plt.figtext(x=0.5,y=0.03,s=details[1],size='small',ha='center',va='top')
  colour = list(islice(cycle(['b','r','g','y','k']),0,len(chains)))
  # Plot each chain across the subplots to avoid excess large data selections
  for c in chains:
    chain = post.loc[(post.Indices.chain == c),:]  # Do only once for all pars
    for p in range(w*h):
      axis = ax[p//w,p%w]
      if p >= lp:
        if c == chains[0]: fig.delaxes(axis)
        continue
      chain.plot(x=('Indices','draw'),y=pl[p],
                              color=colour[c],alpha=0.5,ax=axis,legend=False)
      # For the last chain, set up the subplot display details
      if c == chains[-1]:
        # Hide x-axis except on lowest plots
        if p < lp-w: plt.setp(axis.get_xticklabels(), visible=False)
        axis.set_title("${0:s}_{{{1:d}}}$".format(*(pl[p])),pad=1.0,
                  fontdict={'fontsize':'medium','verticalalignment':'bottom'})
        axis.set(ylabel="",xlabel="")
        ylim = axis.get_ylim()
        x = post.loc[(post.Indices.warmup == 1.0),('Indices','draw')]
        axis.fill_between(x,ylim[0],ylim[1],color='grey',alpha=0.2,lw=0)
  return (fig,ax)

#............................................................................

def pairs(post,pars,detail,maxtreedepth=10,samsize=2000):
  """
  Display scatter plot of each parameter against each other.

  Parameters
  ----------
  post : dataframe
    As extracted from Pystan fit.  Warmup will be excluded, but post
    must include diagnostics.
  pars : list
    Only lists parameter names not indices e.g. 'a' but not 'a[1]'.
  detail : string
    Used with title.
  maxtreedepth : int
    As passed to Pystan, needed to identify excesses in yellow.
  samsize : int / optional (default 2000)
    Number of points to plot (can significantly reduce display time).

  Returns
  -------
  tuple of figure and axes.

  See Also
  --------
  traceplot : display variability and trend in draws made by MCMC.
  precis : tabulate parameter estimate statistics.
  hpdiplot : display model results as an epidemic curve.

  Examples
  --------
  >>> post = fit.to_dataframe(inc_warmup=True)
  >>> ppars = ("a","b","c")
  >>> detail1 = "Model 2"
  >>> fig,ax = pairs(post,ppars,detail1)
  """
  pl = post.columns[post.columns.isin(pars,level=0)].to_list()
  h,w = [min(30,len(pl))]*2  ## Limit to 30x30 plots...
  chains = post.Indices.chain.unique()
  fig,ax = plt.subplots(h,w)
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt.subplots_adjust(left=0.03,bottom=0.08,right=0.97,top=0.9)
  fig.suptitle("Pairs plot " + detail)
  # colours will be b (ok), y (treedepth exceeded), r (non-convergent)
  pcols = pl + [('Diagnostics','divergent__'),('Diagnostics','treedepth__')]
  draw = post.sample(samsize,replace=True).loc[(post.Indices.warmup == 0.0),
                                               pcols].copy()
  draw[('pairs','colour')] = (draw.Diagnostics.divergent__  + \
          (1-draw.Diagnostics.divergent__)*\
          (draw.Diagnostics.treedepth__ >= maxtreedepth)).astype('i4')
  draw.sort_values(by=('pairs','colour'),inplace=True)
  colour = pd.Series(['b','y','r'],name='colour')
  draw[('pairs','colour')] = colour[draw.pairs.colour].values
  dhalf = int(draw.shape[0]/2)
  for p in range(w):
    for q in range(h):
      axis = ax[q,p]
      if p == q:
        draw.hist(column=pl[p],color='lightblue',ax=axis)
        axis.set_title("")
        pos = axis.get_position()
        plt.figtext(x=pos.x0+pos.width/2,y=pos.y0+0.9*pos.height,
                    s="${0:s}_{{{1:d}}}$".format(*(pl[p])),
                    size='small',ha='center',va='top')
      elif p > q:
        draw.iloc[:dhalf,:].plot.scatter(x=pl[q],y=pl[p],marker='.',
               colorbar=False,c=('pairs','colour'),s=1.0,ax=axis,legend=False)
      else:
        draw.iloc[dhalf:,:].plot.scatter(x=pl[q],y=pl[p],marker='.',
               colorbar=False,c=('pairs','colour'),s=1.0,ax=axis,legend=False)
      axis.set_ylabel("")
      axis.set_xlabel("")
      axis.set_yticklabels([])
      axis.set_xticklabels([])
  return (fig,ax)

#............................................................................

def hpdiplot(x,cases,dd=None,credMass=None,title="",subtitle="",scheme=None,
             xlabel="date",ylabel="new cases per day",
             Log=False,Exlog=False,ncurves=0,**kw):
  """
  Display model results as an epidemic curve.

  Parameters
  ----------
  x : Series
    Dates for x-axis.
  cases : dataframe
    As output from :func:`panstan.gencases` used to construct y-values.
  dd : dataframe or None
    Original points in 'data' column against 'date' column.  If provided,
    these are plotted as small circles.  If 'missing' is listed in `dd` then
    these points are highlighted.
  credMass : float or None
    Credibility mass in range [0,1].  Default :const:`panstan.CredMass`.
  title, subtitle : format str
    Used for captions.
  scheme : dict
    Colour scheme.  Tuple of two colours (line and shading) for Upper, Lower
    and Median;  single colour for Data and Imputed.  None for default scheme.
  xlabel,ylabel : format str
    Used for axes labels.
  Log,Exlog : boolean
    Always,Never use log-scale plots.  Only one should be asserted!
    The default is log-scale only when maximum:median ratio exceeds 10.
  ncurves : int
    Number of sample curves to display.
  **kw : keywords
    Other keywords including those to be used in format string expansions.

  Returns
  -------
  tuple of figure and axes.

  See Also
  --------
  traceplot : display variability and trend in draws made by MCMC.
  pairs : display scatter plots of each parameter value against each other.

  Examples
  --------
  >>> mmn = "m2"
  >>> mpartext = "details of plot"
  >>> tt = "New {Data} {Column} per day"
  >>> st = "Model {mmn} - {mpartext} over {samples:d} samples."
  >>> ylabel="number of new {Column} per day"
  >>> fig,ax = hpdiplot(dr.x,cases,title=tt,subtitle=st,ylabel=ylabel,
  ...                    **argv,mmn=mmn,mpartext=mpartext)
  """

  # Check arg defaults
  if credMass is None: credMass = [CredMass]
  if scheme is None:
    scheme = dict(Upper=('r','pink'),Lower=('forestgreen','palegreen'),
                  Median=('b','lightblue'),Data='b',Imputed='r')

  # Check arg viability
  if not isinstance(credMass,collections.abc.Iterable) or\
    isinstance(credMass,six.string_types): credMass = [credMass]
  if len(credMass) > 1: raise ValueError("hpdiplot accepts only one credMass")
  credMass = [float(credMass[0])]

  # Set up the plot figure and axes
  fig,ax = plt.subplots()
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()
  plt.subplots_adjust(left=0.06,bottom=0.10,right=0.97,top=0.9)
  ax.set(ylabel=ylabel.format(**kw),xlabel=xlabel.format(**kw))
  ax.xaxis.set_major_formatter(pld.DateFormatter('%d-%b-%Y'))
  ax.figure.autofmt_xdate()

  # Main plot: median, upper and lower HPDI, shading
  logplot = False
  hl = []
  if cases is not None:
    # Calculate HPDI and median
    hpdi = hdi(cases,credMass=credMass)
    hpdi.index = hpdi.index.droplevel(1)  # Only one credMass asked for
    # hdi doesn't know about logs, so tidy up here
    hpdi = hpdi[hpdi > 0].fillna(Logtol)
  
    # If max median is less than 10% of max max HPDI, use log axis
    logplot = hpdi.Median.Mean.max() < (hpdi.Upper.Mean+hpdi.Upper.SD).max()/10
    # And adjust this automatic response by arguments supplied
    logplot = (logplot and not Exlog) or Log
    ax.set_yscale('log' if logplot else 'linear')
  
    # HPDI region
    ax.fill_between(x,hpdi.Lower.Mean,hpdi.Upper.Mean,
                    color='lightgrey',ls='--',lw=0.5)
  
    # Mass of curves
    moc = cases.sample(ncurves,replace=True)
    for i,c in moc.iterrows(): ax.plot(x,c.Mean,color='m',ls='-',lw=0.5)
  
    # HPDI sd
    ax.fill_between(x,(hpdi.Lower.Mean - hpdi.Lower.SD),
                         (hpdi.Lower.Mean + hpdi.Lower.SD),
                    color=scheme['Lower'][1],ls='--',lw=0.5)
    ax.fill_between(x,(hpdi.Upper.Mean - hpdi.Upper.SD),
                         (hpdi.Upper.Mean + hpdi.Upper.SD),
                    color=scheme['Upper'][1],ls='--',lw=0.5)
  
    # Median sd
    ax.fill_between(x,(hpdi.Median.Mean - hpdi.Median.SD),
                         (hpdi.Median.Mean + hpdi.Median.SD),
                    color=scheme['Median'][1],ls='--',lw=0.5)
  
    # HPDI means
    ax.plot(x,hpdi.Lower.Mean,color=scheme['Lower'][0],ls="-")
    ax.plot(x,hpdi.Upper.Mean,color=scheme['Upper'][0],ls="-")
    # Median mean
    ax.plot(x,hpdi.Median.Mean,color=scheme['Median'][0],ls="-")

    # Prep legend elements
    hl = [pll.Line2D([],[],color=v[0],mfc=v[1],mec='none',ms=20,
                     marker=[(0,1),(1,1),(1,-1),(0,-1)],label=k+" $\pm$1 SD")
          for k,v in scheme.items() if len(v) == 2]
    hl = [pll.Line2D([],[],color='none',mfc='lightgrey',mec='none',ms=20,
                     marker=[(-1,0.5),(1,0.5),(1,-0.5),(-1,-0.5)],
                     label="{0:.0%} credival:".format(credMass[0])), *hl] + (
         [pll.Line2D([],[],color='m',lw=0.5,label="sample curve")]
           if ncurves > 0 else [])
  
  # Set overall title and subcaption
  fig.suptitle(title.format(**kw)+(" (log plot)" if logplot else "")+":")
  plt.figtext(x=0.5,y=0.02,size='small',ha='center',va='top',
              s=subtitle.format(**kw))

  # Add original data points
  colour = pd.Series([scheme['Data'],scheme['Imputed']],name='colour')
  xlim = ax.get_xlim()
  ##! ddsel = (dd.date >= xlim[0]) & (dd.date <= xlim[1])
  if dd is not None:
    ddp = pd.DataFrame(dict(date=dd.date,data=dd.data,colour=scheme['Data']))
    if 'missing' in list(dd):
      ddp['colour'] = colour[dd.missing.astype('i4')].values
    # NB dd.plot.scatter would be preferred but ignores facecolors='none'
    scatter = ax.scatter(ddp.date,ddp.data,marker='o',
                         facecolors='none',edgecolors=ddp.colour,s=50,
                         label=ddp.colour)

    # Add scatter elements to legend handles
    hl += [pll.Line2D([],[],color='none',mfc='none',mec=v,
                      marker='o',ms=8,label=k+" point")
           for k,v in scheme.items() if len(v) == 1 and v in ddp.colour.values]
  # Finally, add legend
  ax.legend(handles=hl)

  return (fig,ax)  # Did plot something

#----------------------------------------------------------------------------
# Argument parsing utilities
def parse_date(v):
  """
  argparse extender: parse a string as a date using a fixed set of formats.

  Parameters
  ----------
  v : str
    The date string or :const:`panstan.Auto`.

  Returns
  -------
  Either a datetime object interpreting the string or an empty list if\
  :const:`panstan.Auto` was passed.

  Raises
  ------
  argparse.ArgumentTypeError if string does not represent a date.

  Examples
  --------
  >>> import argparse
  >>> p = argparse.ArgumentParser(description="My Program")
  >>> p.add_argument("--date",metavar="date",nargs='+',default=Auto,
  ...                type=parse_date,help="Dates to use"),
  """
  fmtl = ["%Y-%m-%d","%Y/%m/%d","%d-%b-%Y","%d/%b/%Y",
          "%Y-%b-%d","%Y/%b/%d","%d-%b-%y","%d/%b/%y"]
  if v == Auto:
    rv = list()
  else:
    rv = None
    for fmt in fmtl:
      try:
        rv = datetime.strptime(v,fmt)
        break
      except ValueError:
        pass
    if rv is None:
      raise argparse.ArgumentTypeError("Invalid date in argument: " + v)
  return rv

def parse_non_negative_int(v):
  """
  argparse extender: parse a string as a non-negative integer.

  Parameters
  ----------
  v : str
    The integer string to parse.

  Returns
  -------
  A non-negative integer if the string represents one.

  Raises
  ------
  argparse.ArgumentTypeError if string does not represent a non-negative
  integer.

  Examples
  --------
  >>> import argparse
  >>> p = argparse.ArgumentParser(description="My Program")
  >>> p.add_argument("--number",metavar="number",nargs='+',
  ...                type=parse_non_negative_int,help="Numbers to use"),
  """
  error = True
  try:
    rv = int(v)
    if float(v) == rv and rv >= 0: error = False
  except:
    pass
  if error:
    raise argparse.ArgumentTypeError(
            "Invalid non-negative integer in argument: " + v)
  return rv

def parse_non_negative(v):
  """
  argparse extender: parse a string as a non-negative float.

  Parameters
  ----------
  v : str
    The float string to parse.

  Returns
  -------
  A non-negative float if the string represents one.

  Raises
  ------
  argparse.ArgumentTypeError if string does not represent a non-negative float.

  Examples
  --------
  >>> import argparse
  >>> p = argparse.ArgumentParser(description="My Program")
  >>> p.add_argument("--float",metavar="float",nargs='+',
  ...                type=parse_non_negative,help="Numbers to use"),
  """
  error = True
  try:
    rv = float(v)
    if rv >= 0: error = False
  except:
    pass
  if error:
    raise argparse.ArgumentTypeError(
            "Invalid non-negative float in argument: " + v)
  return rv

#----------------------------------------------------------------------------
# Key constants for documentation - these are used by the API
Auto = 'Automatic' #: Signifies automatic assignment of intervention dates.
CredMass = 0.9     #: Default credibility interval.
ModelCode = dict()
r"""
Model code dictionary

*The m-type model specification*

Whilst it is natural to fit to the data as-is, the wide range of values
will tend to make the MCMC fix upon limited aspects of the data.  Instead,
the log-data exhibit a simpler shape consistent at all scales.  The model
then fits Gaussians (constrained quadratic splines) to (log) epidemic data.
This could be improved by fitting epidemic curves directly.

For estimates of data variance, two models are covered, one for constant
variance and one for variable, i.e. a constant shape parameter multipled
by the fitted Gaussian value, ensuring larger variance for larger data
values.  Translating back to log-data requires a simple approximation.
For variable variance, one SD from mean (:math:`m`) is :math:`m+sm`,
where :math:`s` is the constant shape parameter.  In log world,
:math:`\mu=\log m`, giving \
:math:`e^\mu + s e^\mu = e^\mu (1+s) = e^\mu e^s + O(s^2) = 
e^{\mu+s} + O(s^2)` i.e. the variable Gaussian SD must approximately
correspond to a constant log-data SD provided :math:`s \ll 1`.
Replacing :math:`s` by :math:`\frac{s}{m}` recovers the constant SD case
and in the log-world leads to :math:`\mu + s e^{-\mu}`, provided
:math:`s e^{-\mu} \ll 1`, a more likely outcome.  This means that the
constant Gaussian SD is approximately modelled by the division of the
log-world SD by the Gaussian mean.  This is controlled by C in the model
below.  No warning of large :math:`s` is given.

Another feature of the model is the ability to estimate break-points where
parameters change.  This is useful for tracking changes in policy,
certainly for bounces in case numbers (successive Gaussian "waves"), but
possibly also in detecting the effects of lock-downs and other measures.
Allowing MCMC to search for the best breaks does work to an extent, but a
good estimate is available through ordinary linear modelling.  Hence, the
model only seeks to make minor adjustments to the linear model estimates.

The model avoids extremes by using logistic curves (see :func:`inv_logit`):

  1 Gaussians excessively time-shifted to the past are penalised.  This avoids
  solutions that assume a huge epidemic started before the first case.  The
  logistic is applied to the constant SD case only, to prevent the Gaussian SD
  becoming excessively large.

  2 Solutions implying more cases than humans are penalised.  A true epidemic
  model would have this limit built in as a parameter, but the logistic is used
  for each Gaussian spline to convert a limit to a near boolean real value.  An
  array containing 1.0 for each spline is then required to match a normal
  distribution with mean equal to the logistics and SD set to a fixed small
  value.  This makes solutions near or exceeding the limit much less likely,
  and illustrates a quite general technique for efficiently implementing such
  constraints within the model block, rather than as data or parameter
  constraints.

Data pre-processing takes care of the worst outliers and missing data.
The model supplements this analysis with imputed data points.
No date-error estimation is made, though this was considered (see model
comments).

*Parameter priors*

Expect duration of substantial case numbers of a month or so, with big SD.
This gives a~0.0006, b~0.033, making the standard normal very flat as a prior
for b;  a is constrained, and a typical half-cauchy is sufficiently flat.
For c, can only expect 1000's with a range of 100's of 1000's: log = 7-12.
For mud (break point estimates), an SD of 1 day is ample.
"""

#............................................................................
# Other constants needed by module
# Tolerances
Tol = 1e-5       # A small number used to deal with unwanted zeroes -- to stan
Ltol = 0.1       # Used to avoid log(0);  0.1 of a case is not a case
Logtol = np.log(Ltol)
Unlikely = -14   # Add to log for approx a millionth of smallest value

# ModelList names
Args = 'Args'
Code = 'Code'
Data = 'Data'    # Also used by argparse processing
Fit = 'Fit'
Index = 'Index'
Model = 'model'
Post = 'Post'
Pres = 'pres'
Summary = 'Summary'
Table = 'table'

# None of the remaining code will be documented
if __name__ == "__main__":
  #----------------------------------------------------------------------------
  # Futures...
  ## *** Print estimated time to run MCMC when Running based on last run
  ## Or add progress bar by diverting Pystan logs and tracking output
  ## See 'Future arguments as fixed values'
  ## Improve (add) Error handling...
  
  #............................................................................
  ## Future arguments as fixed values
  Worldpop = 7.5e9 # Constraint on peak cases  -- passed to stan
  ## Args for date range?
  
  #----------------------------------------------------------------------------
  # Globals
  endl = '\n'
  # Argument names (where not already defined in module)
  All = 'All'
  Column = 'Column'
  ConstantSD = 'C'
  Exlog = 'Exlog'
  File = 'File'
  Force = 'Force'
  Geometric = 'G'
  Interventions = 'I'
  Junctions = 'J'
  Log = 'Log'
  Manual = 'Manual'
  MCadaptdelta = 'mcadaptdelta'
  MCchains = 'mcchains'
  MCiter = 'mciter'
  MCtreedepth = 'mctreedepth'
  MCwarmup = 'mcwarmup'
  Ncurves = 'ncurves'
  Nosave = 'Nosave'
  Pdf = 'Pdf'
  Plot = "Plot"
  Review = 'Review'
  Samples = 'samples'
  SD1 = 'S'
  Totals = 'Totals'
  WAIC = 'Waic'
  # File types
  JGZ = '.json.gz'
  PDF = Pdf.lower()  # That'll confuse 'em!
  # Convenient values
  Day = timedelta(days=1)
  Hour = timedelta(hours=1)
  
  #----------------------------------------------------------------------------
  # Parse arguments
  # Set up command line parsing
  p = argparse.ArgumentParser(description="Estimate covid-19 progression",
       formatter_class=argparse.ArgumentDefaultsHelpFormatter,prefix_chars='-+')
  p.add_argument('-v', '--version',action='version',
                 version=Path(__file__).name+" "+ __version__)
  
  # Model specification arguments:  with only one model type (m) it is presumed
  # that any future model types will also use intervention and shape parameters
  # along the same lines, these being reasonably general notions.
  # ... multi-valued arguments
  p.add_argument("--interventions","-i",metavar="num",dest=Interventions,
      nargs='+',default=list(),type=parse_non_negative_int,
      help="Numbers of interventions (=intervals less one)")
  p.add_argument("--junctions","-j",metavar="date",dest=Junctions,nargs='+',
      default=Auto,type=parse_date,
      help="Dates for interventions (junctions between intervals)"),
  
  # ... flags
  ##! Add -G to apply geometric distribution...
  p.add_argument("--ConstantSD","-C",action='store_true',dest=ConstantSD,
        default=False,
        help="Run a constant SD model as well as one varying with case count")
  p.add_argument("--Directed","-D",action='store_true',dest=Geometric,
        default=False,
        help="Apply above/below geometric distribution as well as normal only")
  p.add_argument("--SD1","-S",action='store_true',dest=SD1,default=False,
        help="Run a single-valued SD model as well as one by interventions")
  
  # Data and display arguments
  p.add_argument("--column","-c",metavar="column",dest=Column,
        default="cases",help="Name of column to process")
  p.add_argument("--data","-d",metavar="string",dest=Data,default="UK",
        help="Filter for data to select")
  p.add_argument("--file","-f",metavar="path",dest=File,
        type=argparse.FileType('r'),help="ODS file to use, CSV with -M")
  p.add_argument("--ncurves","-n",metavar="num",dest=Ncurves,default=0,
        type=parse_non_negative_int,help="Number of sample curves to plot")
  p.add_argument("--samples","-s",metavar="num",dest=Samples,
        default=1000,type=parse_non_negative_int,
        help="Number of samples to use in posterior analysis")
  
  # MCMC arguments
  p.add_argument("--mcchains",metavar="num",dest=MCchains,
        default=cpu_count()-1,type=parse_non_negative_int,
        help="Number of MCMC chains")
  p.add_argument("--mciter",metavar="num",dest=MCiter,default=30000,
        type=parse_non_negative_int,help="Number of MCMC iterations")
  p.add_argument("--mcwarmup",metavar="num",dest=MCwarmup,
        default=10000,type=parse_non_negative_int,
        help="Number of MCMC iterations used in warmup")
  p.add_argument("--mctreedepth",metavar="num",dest=MCtreedepth,
        default=10,type=parse_non_negative_int,
        help="MCMC tree depth, >10 increases run time significantly"),
  p.add_argument("--mcadaptdelta",metavar="float",dest=MCadaptdelta,
        default=0.8,type=parse_non_negative,
        help="MCMC adapt_delta: increase to improve convergence"),
  
  # Operational flags
  p.add_argument("--All","-A",action='store_true',dest=All,default=False,
        help="Process all models already in the data file")
  p.add_argument("--Log","-L",action='store_true',dest=Exlog,default=False,
        help="Do not use log plots (even for extreme ranges)")
  p.add_argument("++Log","+L",action='store_true',dest=Log,default=False,
        help="Always use log plots")
  p.add_argument("--Manual","-M",action='store_true',dest=Manual,default=False,
        help="With CSV file, save manually corrected data; " +\
             "with no file, do not use saved corrections")
  p.add_argument("--Nosave","-N",action='store_true',dest=Nosave,default=False,
        help="Do not save MCMC data")
  p.add_argument("--Output","-O",action='store_true',dest=Pdf,default=False,
        help="Output plots to pages of '<column>.pdf'")
  p.add_argument("--Run","-R",action='store_true',dest=Review,default=False,
        help="Do not run MCMC")
  p.add_argument("++Run","+R",action='store_true',dest=Force,default=False,
        help="Rerun MCMC even if data not newer")
  p.add_argument("--Totals","-T",action='store_true',dest=Totals,default=False,
        help="Column data is totals not increments")
  p.add_argument("--Plot","-P",action='store_true',dest=WAIC,default=False,
        help="Skip all plots including HPDI, just compare models")
  p.add_argument("++Plot","+P",action='store_true',dest=Plot,default=False,
        help="Show all plots including trace and pairs plots")
  
  # Parse and qualify arguments
  argv = vars(p.parse_args())
  if (argv[Plot] and argv[WAIC]) or (argv[Log] and argv[Exlog]) or\
     (argv[Review] and argv[Force]):
    raise argparse.ArgumentTypeError("Contradictory arguments + and -")
    
  # Establish the model code dictionary
  fpath = Path(__file__).parent
  for stan in fpath.glob("*.stan"):
    with open(stan,'r') as mf: ModelCode[stan.stem] = mf.read()
  assert len(ModelCode) > 0,"No stan model code files found"
  print("Loaded code for model types " +", ".join(sorted(ModelCode.keys())))
  
  #----------------------------------------------------------------------------
  # Naming scheme for data frames:
  #   df*:  read from csv or ods file
  #   dd*:  data for analysis
  #   ds*:  shadow for another data frame
  #   dc*:  column analysis of another data frame
  #   dg*:  groups derived from data frame e.g. spread counts over missing data
  #   dm*:  model table
  #   dr*:  regression table
  
  #----------------------------------------------------------------------------
  # Primary data set
  # ECDC data direct from web
  ecdc = "data/ECDC" + JGZ  ## This needs tidying up and generalising
  cet = pytz.timezone('CET')   # ECDC works to CET
  now = datetime.now()
  lastRefresh = now - 3*Day    # Long enough ago to default to refresh
  df = None                    # No file read
  dmc = None                   # No manually corrected data
  if argv[File] is None:
    refresh = True
    try:
      with gzip.open(ecdc) as ed: df,lastRefresh,dmc = jsload(ed)
    except Exception as e:
      #print("ECDC exception: " + str(e))
      pass
    # Limit access to website
    # 1. Only try to refresh data once per hour
    #print(lastRefresh,df,dmc)
    if df is None:
      pass
    elif (now - lastRefresh)/Hour >= 1:
      ldt = max([datetime.strptime(dt,"%d/%m/%Y") for dt in df.dateRep])
      #print(ldt)
      # 2. Only refresh if latest data is old or now it's after 09:59CET
      cetnow = datetime.now(tz=cet)
      if ldt >= (now - Day) or\
         (ldt >= (now - 2*Day) and cetnow.hour < 10):
        refresh = False
    else:
      refresh = False
  
    if refresh:
      print("Refreshing ECDC data..." + endl)
      try:
        df = pd.read_csv(
                 "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv",
                 encoding='utf_8')  # but wanted codecs.BOM_UTF8
        lastRefresh = now
        Path(ecdc).parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(ecdc,'wt') as ed: jsdump([df,lastRefresh,dmc],ed)
      except URLError as e:
        print("Unable to refresh ECDC data (continuing): {0:s}".format(str(e)))
  else:
    ext = argv[File].name.split(sep=".")[-1]
    if ext == "csv":
      if argv[Manual]:
        try:
          with gzip.open(ecdc) as ed: df,lastRefresh,dmc = jsload(ed)
        except:
          pass  # No ECDC correction exists at this point, it will later
      # Read in manually corrected ECDC data
      dmc = pd.read_csv(argv[File],encoding='utf_8') # codecs.BOM_UTF8
      if df is None:
        df = dmc
      if argv[Manual]:
        Path(ecdc).parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(ecdc,'wt') as ed: jsdump([df,lastRefresh,dmc],ed)
    elif ext == "ods":
      # Manually compiled WHO data
      df = pd.read_excel(argv[File],engine="odf",skiprows=2)
    else:
      raise argparse.ArgumentTypeError("File type "+ext+" not supported")
  
  # Apply manual corrections
  if ((argv[File] is not None and argv[Manual]) or
      (argv[File] is None and not argv[Manual])) and dmc is not None:
    # Get rows with matching keys
    keys = ['geoId','dateRep']
    df.set_index(keys,drop=False,inplace=True)
    dmc.set_index(keys,drop=False,inplace=True)
    df = df.combine(dmc,lambda s1, s2: s1.where(s2.isnull(),s2))
  
  #............................................................................
  # Data transformations for model fitting:  find useful data - not nec ECDC now
  # Locate date column as the first one with the sequence 'date' in it
  re_d = re.compile('.*date.*',re.I)
  datecol = [cc.group() for c in list(df) for cc in re_d.finditer(c)]
  if len(datecol) == 0:  raise NameError("Unable to find dates")  ## Subclass
  print("Using "+datecol[0]+" column for dates")
  # Set up clean dataframe for sample
  dd = pd.DataFrame(dict(
           date=[datetime.strptime(d,"%d/%m/%Y") for d in df.loc[:,datecol[0]]],
           data=df.loc[:,argv[Column]]))
  # Filters to get data of interest
  dval = argv[Data]  # Convenience
  if isinstance(dval,str) and len(dval) > 0:
    # Build dataframe of df columns with useful info - like is this char data?
    dc = pd.DataFrame({'char': df.iloc[0,:].apply(lambda x: isinstance(x,str))})
    # Find rows which match supplied data argument
    # convert to str for elementwise comparison to avoid annoying warning
    ## if argv[data] == 'nan' then this could give wrong result
    ds = df.astype(str) == dval
    # Identify most frequent match with supplied data argument
    dc['data'] = (ds.sum(axis=0,skipna=True).idxmax() == dc.index)
    if dc.data.sum() == 0: raise NameError("Unable to locate data for"+dval)
    # In matched rows, find other string columns with unique values
    ds = ds.loc[:,dc.index[dc.data]]
    dc['uniq'] = (df.loc[ds.iloc[:,0]].nunique() == 1)
    # Get a list of these unique values
    dc['aka'] = df.apply(lambda c: c.loc[ds.iloc[:,0]].iat[0] \
                         if dc.uniq[c.name] and dc.char[c.name] else np.nan)
    # Get likely population data, if any
    pop = dc.loc[(dc.uniq) & (~dc.char) & (~dc.data)]\
                 .index.to_series().str.contains('pop',case=False)
    # Find out how frequent they are across the whole data set
    dc['num'] = df.apply(lambda c: sum(c.astype(str) == dc.aka.loc[c.name]))
    # Separate aka's into supersets and genuine aka
    dc['partof'] = dc.aka[dc.num.values > dc[dc.data].num.values]
    dc['aka'] = dc.aka[dc.partof.isnull()]
    # Announce the results of this analysis
    print("Matching {0:s} with {1:s} aka {2:s} in {3:s}".format(
          dc.index[dc.data].values[0],dc.aka[dc.data].values[0],
          ",".join(list(dc.aka[(dc.aka.notnull()) & (~dc.data)])),
          ",".join(list(dc.partof[dc.partof.notnull()]))))
    # Apply selection, if it uses a different column to the user-specified one
    if dc.index[dc.data].values[0] != argv[Column]:
      dd = dd.loc[ds.iloc[:,0],:]
    else:
      argv[Data] = ""
  # Set the population limit if specified in the data, else default to world
  if pop.sum() > 0:
    pop = df.loc[df[dc.index[dc.data].values[0]] ==
                    dc.aka[dc.data].values[0]][pop.index[0]].iloc[0]
    if pop is not None and pop > 1: Worldpop = int(pop)
    print("Population limited to "+str(Worldpop))
  
  #............................................................................
  # Filters and transforms for missing data
  # Eliminate missing data
  dd.dropna()
  # Fill in any date gaps with missing indicators
  alldates = pd.date_range(dd.date.min(),dd.date.max(),freq='D')
  dd = pd.concat([dd,pd.DataFrame(dict(date=
          [d for d in alldates if sum(dd.date == d) == 0]))],ignore_index=True)
  dd.data.fillna(0,inplace=True)
  # Sort by date ascending
  dd.sort_values(by=['date'],inplace=True)
  # Get cases increments and report full ranges
  if argv[Totals]:
    dd = pd.DataFrame(date=dd.date.iloc[1:],data=dd.data.diff())
  print("Full date range {0:%Y-%m-%d} - {1:%Y-%m-%d}".format(
        dd.date.min(),dd.date.max()))
  print("Full count range {0:.0f}-{1:.0f}".format(dd.data.min(),dd.data.max()))
  
  # Filtering out useless or misleading data...
  # Use seven-day moving average to set start date for analysis - before peak
  # Ignore data prior to when the seven-day moving average around each value <2.
  dd['ma7'] = (dd.data.rolling(7,min_periods=1,center=True).mean())
  dd = dd.loc[(dd.ma7 >= 2) | (dd.date >= dd.date.loc[dd.data.idxmax()])]
  md = dd.date.min()
  # Identify high and low outliers by ratios greater than 4.  NB first is NaN
  dd['incr'] = dd.data.pct_change()
  #dd['ratio'] = dd.data.rolling(2).apply(lambda x: (Tol+max(x))/(Tol+min(x)))
  #print((dd.ratio > 4))
  # Form zero values into groups along with the next non-zero value
  # ECDC put negative values in for corrections;  treated here as missing values
  dd['non0'] = (dd.data > 0)
  dd['group'] = dd.non0[::-1].cumsum()
  # Alas, groupby transform won't accept list or dict, so two calls needed:
  # Count the size of each group as this indicates missing data
  dd['glen'] = dd.groupby('group')['data'].transform('count')
  # Spread values after zeros across those zeros (estimate missing data)
  dd['val'] = dd.groupby('group')['data'].transform('mean')
  # Identify zeros and outliers as gaps
  dd['missing'] = (~dd.non0) | (dd.glen > 1) | (dd.incr > 3) | (dd.incr < -3/4)
  # Remove unwanted columns for remaining analysis
  dd.drop(['glen','group','incr','ma7','non0'],axis=1,inplace=True)
  missinglen = dd.missing.sum()
  # Use log data to avoid fascination with big values;  convert dates to numbers
  dd['ldata'] = dd.data.apply(lambda d: np.log(d) if d > 0 else Logtol)
  dd['lval'] = np.log(dd.val)  # No zeros by construction; no -ve by assumption
  dd['datenum'] = ((dd.date - md)/Day).astype('i4')
  print("Accepted date range {0:%Y-%m-%d} - {1:%Y-%m-%d}".format(
        dd.date.min(),dd.date.max()))
  print("Accepted count range {0:.0f}-{1:.0f}".format(
        dd.val.min(),dd.val.max()))
  
  # Cannot specify a junction outside data range!  Or even at the ends.
  junctions = np.array(argv[Junctions])  # np.array for elementwise ops
  if sum(junctions <= dd.date.min()) + sum(junctions >= dd.date.max()) > 0:
    raise ValueError("Some junctions are outside accepted date range")
  
  # Summary, relative to original R:
  # diffo -> data;  diffm -> val;  missing -> missing;  diffi -> val for missing
  
  # Store the data;  missing in pystan is a 1-based index into datenum array
  #print(dd.loc[dd.datenum.diff() > 1])
  mm_data = dict(N=dd.shape[0],lval=dd.lval,datenum=dd.datenum,M=missinglen,
    missing=pd.Series(dd.missing.reset_index().index[dd.missing].values+1,
                      name='missing',dtype='i4'))
  
  #............................................................................
  # Store the stan args to be used in all runs
  mm_args = dict(chains=argv[MCchains],iter=argv[MCiter],warmup=argv[MCwarmup])
  
  
  #----------------------------------------------------------------------------
  # Locate existing data
  # Set up file names for data storage and graphical output
  for cn in pd.concat([dc.aka[(dc.aka.notnull()) & ~dc.data],dc.aka[dc.data]]):
    jgz = "".join(["data/",cn,argv[Column],JGZ])
    if os.path.exists(jgz): break
  print("Data file: " + jgz)
  
  cpdf = None
  if argv[Pdf]:
    pdfmd = dict(Title="panstan data analysis",
                 Subject=argv[Data]+" "+argv[Column],
                 Keywords="panstan Stan MCMC "+argv[Data]+" "+argv[Column])
    cpdf = PdfPages("".join([argv[Data],argv[Column],PDF]),metadata=pdfmd)
  
  # Load existing data
  mml = ModelList()
  if os.path.exists(jgz):
    print("Loading MCMC data for {0:s} {1:s}".format(argv[Data],argv[Column]))
    with gzip.open(jgz) as jz: mml.from_json(jz)
    #print(str(mml))
  
  # To get an idea of treedepth distribution, enable this...
  if False:
    for k,v in mml.items():
      if Fit in v:
        #plot(hist(attr(mml[[i]]$mm_fit@sim$samples[[1]],
        #               "sampler_params")$treedepth__), main=names(mml)[i])
        pause("next model")
    print("Treedepth checked")
    #raise SystemExit("test")
  
  # Define parameters expected for each model type
  mtl = dict(m=['I','S','C','G'])
  
  # Ensure integrity of indexes vs listed models - all listed must be indexed
  # Interpret model names
  for mmn in mml:
    if mmn == Index: continue
    #print(mmn)
    #print(re.findall("([a-z]+)([0-9]+)\.(.*)",mmn))
    ma = re.findall("([a-z]+)([0-9]+)\.(.*)",mmn)[0]
    mt,ma = [ma[0],[ma[1]] + ma[2].split(".")]
    if mt not in mtl:
      raise ValueError("Model {0:s} has unknown type {1:s}".format(mmn,mt))
    mtv = mtl[mt]
    if len(mtv) > len(ma):
      ma = ma + [0]*(len(mtv)-len(ma))
    elif len(mtv) < len(ma):
      raise ValueError("Model {0:s} parameter mismatch: {1:s} = {2:s}".\
                       format(mmn,str(mtv),str(ma)))
    if mt not in mml[Index]:
      mml[Index][mt] = {Table:pd.DataFrame(columns=mtv)}
    if not Table in mml[Index][mt]:
      mml[Index][mt][Table] = pd.DataFrame(columns=mtv)
    if not mmn in mml[Index][mt][Table].index:
      mml[Index][mt][Table].append(pd.DataFrame(
        [dict([(mtv[i],ma[i]) for i in range(len(mtv))])],index=[mmn]))
  
  # Compile index of models to process
  # Interpret arguments
  csdv = (1,0) if argv[ConstantSD] else (0,)  # Just boolean as integer
  gsdv = (1,0) if argv[Geometric] else (0,)   # Just boolean as integer
  sdiv = (0,1) if argv[SD1] else (1,)         # Just 'not boolean' as integer
  # Just to confuse things, convert interventions to intervals as variable "iv"
  # And this is a vector, ivv, here;  also, need to manually force default
  ivv = [iv+1 for iv in argv[Interventions]]
  # The highest value of the number of interventions must be at least the
  # number of specified junctions, or else no interventions must be specified
  if len(ivv) == 0:
    ivv = [len(junctions) + 1]
  elif max(ivv) <= len(junctions):
    print("Adding an intervention count to cover all junctions." + endl)
    ivv.append(len(junctions) + 1)
  # Specified parameters for each model type (currently m only)
  # including additional column showing that model will be run
  dmd = dict(m=[("m{0:d}.{1:d}.{2:d}.{3:d}".format(iv,sdi,csd,gsd),
                dict(I=iv,S=sdi,C=csd,G=gsd,Process=True))
                for iv in ivv for sdi in sdiv for csd in csdv for gsd in gsdv])
  # Recast as tables of models
  dmd = dict([(mt,pd.DataFrame([d[1] for d in dml],index=[d[0] for d in dml]))
              for mt,dml in dmd.items()])
  
  # Announce the list to follow
  print(endl+"Models for {0:s} {1:s} with process indicator:".\
        format(argv[Data],argv[Column]))
  
  # Add in all known models (processing status will be determined later)
  for mt,mtv in mml[Index].items():
    # If there is no table in the index, nothing to add in
    if Table not in mtv: continue
    dmk = mtv[Table]
    # If there is no presentation list, assume no pres columns added yet
    pres = mtv[Pres] if Pres in mtv else list()
    # If there is no table for this model type, add one in anticipation
    if mt not in dmd: dmd[mt] = pd.DataFrame()
    # Add in missing columns, if any
    if not "G" in list(dmk):
      dmk["G"] = 0
      dmk.index = [dmki+".0" for dmki in dmk.index]
    # Append existing.  Repetition not suppressed by append(): Process is NaN
    dmd[mt] = dmd[mt].append(dmk.loc[:,~dmk.columns.isin(pres)])
    # Set Process:  if "all" specified, process all existing models, else no new
    dmd[mt].fillna(argv[All],inplace=True)
  
  # Build parameter and presentation table, showing processing status
  for mt,dm in dmd.items():
    # Sort by name for presentation and set up basic presentation columns
    dm.sort_index(inplace=True)
    pres = ["Process"]
    # Drop any duplicates based on the main model parameters
    dm.drop_duplicates(subset=mtl[mt],keep='first',inplace=True)
  
    # Model type specific processing
    if mt == "m":
      # For type m, suppress m1.1, m1.0 duplicates - they give the same models
      dm.loc[(dm.loc[:,["I","S"]] == [1,0]).sum(axis=1) == 2,["I","S"]] = [1,1]
      dm.drop_duplicates(subset=mtl[mt],keep='last',inplace=True)
  
      # Add presentation columns
      dm["Interventions"] = dm.I - 1
      dm["Shape parameters"] = dm.S*(dm.I-1)+1
      dm["Constant SD"] = (dm.C == 1)
      dm["Directed"] = (dm.G == 1)
      pres = pres+["Interventions","Shape parameters","Constant SD","Directed"]
  
    # Present the list of models of this type to be processed
    print(dm.loc[:,pres])
    # Save the updated model specifications in the index
    if mt not in mml[Index]: mml[Index][mt] = dict(model=dict())
    mml[Index][mt][Table] = dm
    mml[Index][mt][Pres] = pres
    dmd[mt] = dm
  
  # Process the models
  # This is the intended multi-model-type structure, but more work needed
  # to handle model parameterisation generically...
  for mt,dm in dmd.items():
    # Ideally, loop below will call a function and everything needed will
    # be in mml[mmn], avoiding extra arg passing
    for mmn,mp in dm.loc[dm.Process].iterrows():
      # Some common text about model parameters
      mpartext = "{0:d} {1:s} and {2:d} {3:s} {5:s} shape {4:s}".format(
                 mp.I-1,"intervention" if mp.I == 2 else "interventions",
                 mp.S*(mp.I-1)+1,"constant" if mp.C else "variable",
                 "parameter" if mp.S*(mp.I-1) == 0 else "parameters",
                 "directed" if mp.G else "undirected")
  
      # Decide on refreshes and saves
      mmrun = not argv[Review]
      post = None
      mmsave = not argv[Nosave] # double negative!
      hpdip = None       # No HDPI plot produced, yet
      if mmn in mml:     # Model has been processed before
        msto = mml[mmn]  # Stored model
        if not argv[Review]:
          # If there is no saved data, force sampling (backward compatibility)
          if Data not in msto:
            msto[Data] = dict(M=0,N=0,lval=[],datenum=[],missing=[])
          # If the code has changed, force recompilation as well as sampling
          if Fit not in msto or msto[Fit][Code] != ModelCode[mt]:
            msto[Fit][Code] = None
          if Args not in msto: msto[Args] = dict()  # Will force rerun
        # Rerun if forced or args uplifted or data changed or code changed
        # Alas, in this case, element-wise compare of Series is unhelpful...
        series = ('lval','datenum','missing')
        mmrun = mmrun and (argv[Force] or msto[Args].keys() != mm_args.keys() or
                 sum([msto[Args][k] < mm_args[k] for k in mm_args]) > 0 or
                 sum([mm_data[k] == msto[Data][k] for k in ('M','N')]) < 2 or
                 sum([mm_data[k].round(6).equals(msto[Data][k].round(6))
                     for k in series]) < 3 or msto[Fit][Code] is None)
      elif argv[Review]:
        print(endl+"Cannot review "+mmn+" because there is no data"+endl+endl)
        continue
  
      # If not running, extract post from saved data and don't save it again
      if not mmrun:
        print(endl+"Skipping MCMC for "+mmn+"...")
        post = mml[mmn][Fit][Post].copy(deep=False) # Stops invalid warnings
        mmsave = False
      # Else running the MCMC
      else:
        print(endl+"Running MCMC for "+mmn+"...")
  
        # Initial values - the better these are, the faster the MCMC run,
        # and the tighter the model can be made, making it even faster to run.
  
        # Interventions.  First use up all specified junctions, and then seek
        # to automatically identify new junctions.  Note that the earliest and
        # latest accepted dates are implicit junctions, and are treated as such.
        # Specified junctions are used in the order given, up to current I-1.
  
        # Begin with first automatic junction (when I > number of junctions +1).
        # Method: split sample at each date, fitting to data before and after
        # split;  record var(residuals).  Pick the lowest total variance, summed
        # proportionately by size of interval.  This identifies the best split.
        # Adverse bends:  lm fits to exp curve must meet shape constraint.  If
        # this is not achieved, use a linear fit as below and accept bigger var.
        # Fitting three parameters, so need absolute minimum of three points,
        # except where linear fit is forced by constraint on adverse bends.
        # Expect that it will always be possible to improve on total variance by
        # adding parameters (up to saturation), but WAIC identifies best model.
  
        # For larger I, repeat the procedure, looking to beat the existing
        # variances for the net fit.  It is necessary to ensure not just the
        # minimum of 3 points per interval needed to fit a quadratic uniquely,
        # but also to allow for variation during MCMC by a few days at each
        # junction.  MCMC exploration around a junction is constrained in 
        # the model:  expect the best split to be within a day or two of the
        # linear model answer, and require the same for manually specified
        # junctions.  This implies a minimum of 18 points are required in an
        # interval to be split.  This may miss some globally better splits,
        # but finding those would involve shifting the splits identified, and
        # doing that would greatly increase the complexity.
        minsplit = 9  # min length of an interval
  
        # Exclude infinite and missing values from data to be used
        ## if data inf, amend dd.missing definition - but shouldn't happen
        dr = dd.loc[~dd.missing,:]
        NJ = len(junctions) + 1  # +1 so NJ aligns to I in definition
        # Convert junctions to numbers...
        jn = [dr.datenum.min()] + [(j - md)/Day for j in junctions]  # NJ values
        # ...and then to data indices
        jn = np.array([sum(dr.datenum < j) for j in jn])
        # Restructure dr for use in simpler regression formulae
        dr = pd.DataFrame(dr[["datenum","ldata"]].values,columns=["x","y"])
        # Create top level model parameter array
        # TechNote: numpy can do all this, but pandas is neater overall
        # Level 1 index: existing interval and possible split
        lun = ['interval','split']
        # Level 2 index: split as lower/upper row number
        lu = ['lower','upper']
        # Columns: break (junction or intervention),var,3x coef,shape
        lrcols = ['Brk','Var','c','b','a','SD']  # Also referred to by 0:6
        # Length of dimensions needed
        n = dr.shape[0]
        ijmax = mp.I+1       # Even if NJ>I only use I junctions
        ijmin = min(mp.I,NJ)
        # Top level interval data frame
        drmi = pd.DataFrame(index=range(ijmax),columns=lrcols)
  
        # Prefill model with junctions up to I-1, using 0 and n for end points
        jnsel = [j < ijmin for j in range(0,NJ)]
        drmi['Brk'] = np.hstack((jn[jnsel],np.full(ijmax-sum(jnsel),n,'i4'))) 
  
        # For each lower break point (interval) to be found
        for i in range(ijmin,ijmax):            # for all auto interventions
          mix = pd.MultiIndex.from_product([range(i),lu],names=lun)
          drmj = pd.DataFrame(index=mix,columns=lrcols)
          # Find the best fit across all intervals
          for j in range(i):  # By construction, drmi has row j+1 at any time
            # Check this interval is viable and set up array for new breaks
            # Pandas/Python/numpy limitation: upcasting is uncontrollable!
            lrmr = (int(drmi.iloc[j,0]),int(drmi.iloc[j+1,0])) # Interval range
            n = lrmr[1] - lrmr[0]             # Number of points in interval
  
            # If interval is specified by a junction, do full fit and skip scan
            if n < 2*minsplit or i <= ijmin:  # i is 1 initially
              drmj.loc[(j,'upper'),:] = [lrmr[0],np.inf] + ([0,0,0,0] if n == 0\
                else lrfit(slice(*lrmr),dr,mp.C)[1:]) # Ensure fit not optimal
              continue
  
            # Scan for least variance in all possible splits of this interval
            lbnd = lrmr[0]+minsplit
            krng = range(lbnd,lrmr[1]-minsplit+1)
            mix = pd.MultiIndex.from_product([krng,lu],names=lun)
            drmk = pd.DataFrame(index=mix,columns=lrcols)
            for k in krng:
              # Fit and save the lower model
              mr = slice(lrmr[0],k)
              drmk.loc[(k-lbnd,'lower'),:] = [lrmr[0]] + lrfit(mr,dr,mp.C)
  
              # Fit and save the upper model
              mr = slice(k,lrmr[1])
              drmk.loc[(k-lbnd,'upper'),:] = [k] + lrfit(mr,dr,mp.C)
            # End loop: k interval index bounds
  
            # Find least total variance, summing proportionately (use sos)
            lrv = [sum(np.array([k+minsplit,n-k-minsplit])*drmk.xs(k)['Var'])
                   for k in range(n-2*minsplit+1)]
            lv = np.argmin(lrv)
            # Save the details for this interval and go to the next one
            drmj.loc[(j,lu),:] = drmk.xs(lv).values # No interval on xs
            drmj.loc[(j,'upper'),'Var'] = lrv[lv]  # Total variance for interval
          # End loop: j 1:i
  
          # For junctions (including 0), copy all results across
          if i <= ijmin:  # i is 1 initially
            drmi.iloc[0:i,:] = drmj.xs('upper',level=1).values
            continue
          # Find least total variance across intervals
          # Replace lower bound with optimal lower split, add one new for upper
          lv = np.argmin(drmj.xs('upper',level=1)['Var'])
          drmi.iloc[[lv,i],:] = drmj.xs(lv).values
          drmi.sort_values(by='Brk',inplace=True)
        # End loop: i 1:I-1
  
        # Required intervals are now available, but need to reinsert missings...
        # ... i.e. translate back to date numbers and remove ends, even if I=1
        drmi.drop(index=drmi.tail(1).index,inplace=True) # Lose out of range brk
        brk = pd.Series([0] + dr.x.loc[drmi.Brk[1:]].astype('i4').to_list() + \
                        [int(dd.datenum.max()+1)])
        drmi.set_index(pd.Index(brk.iloc[1:]),inplace=True)
        drmi.drop(columns=['Brk','Var'],inplace=True)
        # Set up signs used for model
        drmi.loc[:,['a','c']] = -drmi.loc[:,['a','c']]
        # Calculate optimal logdiffi estimates
        di = dd.lval.loc[dd.missing]  # Default, to be replaced
        for div,srmi in drmi[::-1].iterrows():
          dsel = (dd.missing) & (dd.datenum < div)
          x = dd.datenum.loc[dsel]
          di[dsel] = (-srmi.c + srmi.b*x - srmi.a*x*x)
  
        # Store initial values
        drmi.rename(columns=dict(SD='sigmam'),inplace=True) ## Arguably start as
        mm_start = drmi.to_dict('list')
        mm_start['logdiffi'] = di.to_list()
        mm_start['mud'] = brk.iloc[1:-1].to_list()
  
        # Generally good estimates for initial values will help convergence,
        # but some variation between chains is also desirable for exploration.
        # A crude estimate of a shape parameter is made to facilitate this.
        drmi = drmi.append(drmi.iloc[0]) # Dummy row to avoid nx1 mix-up
        sh = drmi.sigmam/dd.lval.mean()/(dd.data.mean() if mp.C == 1 else 1)
        mm_start = []
        for c in range(argv[MCchains]):
          drm = pd.DataFrame(norm.rvs(loc=drmi,scale=drmi.mul(sh,axis=0).abs()),
                             columns=drmi.columns).iloc[0:-1]
          # Doubly ensure a and sigmam are positive
          drm.loc[:,['a','sigmam']] = drm.loc[:,['a','sigmam']].abs()
          # Add variant to list of initial value sets
          mm_start +=[{**drm.to_dict('list'),'logdiffi':di.to_list(),
                       'mud':brk.iloc[1:-1].to_list()}]
          # Deal with requirement for a single shape parameter
          if mp.S == 0:
            ##mm_start[c]['sigmam'] = [np.mean(mm_start[c]['sigmam'])]
            mm_start[c]['sigmam'] = [np.sqrt(
              (drm.sigmam**2).mul(brk.diff().tail(-1).values,axis=0).sum() /
              brk.diff().sum()) ]
  
        # Report initial values, nicely tabulated
        print("Initial values:")
        for c in range(len(mm_start)):
          il = max([len(v) for v in mm_start[c].values()])
          drm = pd.DataFrame(dict([(k,v+[""]*(il-len(v))) for k,v
                                   in mm_start[c].items()]),index=range(1,il+1))
          drm.rename_axis("Chain "+str(c),axis=1,inplace=True)
          print(drm)
  
        # Store fixed values
        ## Fixed is now a bad term - non-change-determining?
        ## junctions should be checked to determine mmrun as mm_pars
        ## Or... they should be recorded and displayed with the model as mm_pars
        ## ... but that would require +R explicit if the junctions change
        mm_fixed = dict(d=np.array(brk.iloc[1:-1]),worldpop=Worldpop,tol=Tol)
  
        # Store  model parameters
        mm_pars = mp.loc[["I","S","C","G"]].to_dict()
  
        # Check if a recently compiled model is available (save MCMC time)
        m = None
        if mt in mml[Index] and Model in mml[Index][mt] and \
          'python' in mml[Index][mt][Model]:
          m = MyStanModel.load(mt,mml[Index][mt][Model]['python'])
        # Check this compiled version is up to date
        if m is not None and m.model_code != ModelCode[mt]: m = None
        if m is None:  # Code change or just never compiled
          m = MyStanModel(model_code=ModelCode[mt])
          mml[Index][mt][Model] = m.save(mt)  # avoid future recompilation
        # Use the model to fit the data
        fit = m.sampling(data={**mm_data,**mm_pars,**mm_fixed},
                         **mm_args,init=mm_start,
                         control=dict(max_treedepth=argv[MCtreedepth],
                                      adapt_delta=argv[MCadaptdelta]),
                         refresh=argv[MCwarmup])
        summary = fit.summary()
        # OrderdDict containing:
        #  summary   np.array pars x stats
        #  c_summary np.array pars x stats
        #  summary_rownames np.array str
        #  summary_colnames tuple str       includes n_eff and Rhat
        #  c_summary_rownames np.array str
        #  c_summary_colnames tuple str
        # Find n_eff and Rhat
        ds = pd.DataFrame(summary['summary'],index=summary['summary_rownames'],
                          columns=summary['summary_colnames'])
  
        # Attempt to extract samples for posterior distribution analysis later
        post = fit.to_dataframe(inc_warmup=True)
  
        # Pystan should have used a MultiIndex:  one-time transforms here
        ppars = ("a","b","c","mud","sigmam","logdiffi")
        gqs = ("mu","sigma","peak")
        ds = str2mi(ds,ppars,gqs,'Indices','Diagnostics',axis=0)
        post = str2mi(post,ppars,gqs,'Indices','Diagnostics',axis=1)
  
        # Even if save is not being done, need to store fit for access in mywaic
        mml[mmn] = {Fit:{Post:post,Code:ModelCode[mt],Summary:ds},
                    Data:mm_data,Args:mm_args}
        if mmsave:      # Optionally, save the model fit as well as model itself
          print("Saving MCMC data for "+argv[Data],argv[Column])
          Path(jgz).parent.mkdir(parents=True,exist_ok=True)
          with gzip.open(jgz,'wt') as jz: mml.to_json(jz)
  
      # post now contains an initial set of MCMC parameter columns:
      # chain (0:MCchains), draw (-MCwarmup:(MCiter-MCwarmup)) and warmup
      # (1.0=True,0.0=False) - if permute is true the last becomes
      # warmup permutation, chain_permutation, permutation_order,
      # and chain_permutation_order
      # This is followed by the model parameters as string names, e.g.:
      # a[1], b[1], c[1], sigmam[1], logdiffi[1], logdiffi[2],
      # and the generated values:  mu[1], sigma[1], peak[1]
      # Note that mud is missing when its dimension is 0
      # And then diagnostics:
      # lp__, accept_stat__, stepsize__, treedepth__, n_leapfrog__,
      # divergent__, energy__
      # Of these, treedepth__ should be less than MCtreedepth, and divergent__
      # should be 0.0 (as opposed to 1.0)
  
      # Primary MCMC diagnostics
      if mmrun or argv[Plot]:
        if post.shape[0] > 0 and not argv[WAIC]:
          # Components for plot rubric
          detail1 = " ".join(["Model",mmn,"for",argv[Data],argv[Column],
                        "with",mpartext])
          detail2 = " ".join([str(argv[MCchains]),"chains with",
                       str(argv[MCiter]-argv[MCwarmup]),"sampling iterations,",
                       str(argv[MCwarmup]),"warmup iterations on grey shading"])
    
          # stability traces
          ppars = ("a","b","c","mud","sigmam","logdiffi")
          pdfplot(prompt="trace plot",pdfile=cpdf,
                  fig=traceplot(post,ppars,(detail1,detail2))[0])
          # dashboard should still work - but is in Rethinking package
  
      # From now on, warmup is not wanted
      # NB copy needed to avoid warning about updating a view...
      post = post.loc[(post.Indices.warmup == 0.0),:].copy()
  
      # Always precis
      dp = precis(post,fitsum=mml[mmn][Fit][Summary])
      with pd.option_context('display.max_rows',None,'display.max_columns',None,
                             'display.float_format','{:.02f}'.format):
        #print(mml[mmn][Fit][Summary].round(2))  # Pystan summary
        print(dp) #.round(2))
  
      # Second wave diagnostics
      if mmrun or argv[Plot]:
        if post.shape[0] > 0 and not argv[WAIC]:
          # pairs plot
          samsize = min(2000,(argv[MCiter]-argv[MCwarmup])*argv[MCchains])
          pdfplot(prompt="pairs plot",pdfile=cpdf,
                  fig=pairs(post,ppars,detail1,argv[MCtreedepth],samsize)[0])
  
          # parameter plot - less useful, prone to lock-up: reenable if needed
          if False:
            #parplot <- plot(mm_fit) +
            #           labs(subtitle=paste("Parameters plot",detail1)) +
            #           theme(plot.subtitle=element_text(size=12))
            #pdfplot(prompt="parameters plot",args=list(parplot,pars=ppars))
            pass
        # End if: show MCMC diagnostics
      # End if: run the model
  
      # credMass values should match between tables and plots
      cm=[CredMass]
  
      # Process model output, if any
      cases = None
      if post.shape[0] > 0:
        # mud:  it is helpful to force its existence for index I
        #post[('mud',mp.I)] = np.inf  # Avoid saving to file
        # If needed, it can be removed:
        # post[post.columns[[m != ('mud',mp.I) for m in post.columns]]]
  
        # Determine reasonable date range for plot
        dq = hdi(np.hstack([(post.mu[1]-4*post.sigma[1]).values,
                         (post.mu[mp.I]+4*post.sigma[mp.I]).values]),
                 credMass=cm)
        dq.index = dq.index.droplevel(1)  # Only one credMass asked for
        drlim = np.arange(np.floor(min(dq.Lower.Mean.min(),dd.datenum.min())),
                          np.ceil(max(dq.Upper.Mean.max(),dd.datenum.max())))
        dr = pd.DataFrame(dict(datenum=drlim),dtype='i4')
    
        # Fix dates as dates
        dr['x'] = pd.TimedeltaIndex(dr.datenum,'D') + md
  
        # HPDI plots
        if not argv[WAIC]:
          # Get cases - no need for full set, first 1000 is fair sample
          cases = gencases(argv[Samples],dr,post,constantSD=mp.C)
  
        print("")
        with pd.option_context('display.max_rows',None,
                               'display.max_columns',None,
                               'display.float_format','{:.00f}'.format):
          #print(implications_quantile(post))  # Quantiles
          print(implications(post))       # HDI
      # End If: post exists
  
      # Produce shaded sample plot
      if not argv[WAIC]:
        # Define the colour and label scheme for HPDI plots
        tt = "New {Data} {Column} per day"
        st = "Model {mmn} - {mpartext} over {samples:d} samples."
        ylabel="number of new {Column} per day"
        hpdip = hpdiplot(dr.x,cases,dd,cm,title=tt,subtitle=st,ylabel=ylabel,
                         **argv,mmn=mmn,mpartext=mpartext)
        pdfplot(prompt="epidemic plot",pdfile=cpdf,fig=hpdip[0])
  
  # WAIC
  dw = mywaic(argv[Samples],dd,mml,argv[Review])
  print("")
  with pd.option_context('display.max_rows', None,
                         'display.max_columns', None): print(dw.round(2))
  
  # Indicate that now only waiting for figures to be dismissed
  nfig = len(plt.get_fignums())
  if nfig > 0:
    print("{0:d} figure{1:s} shown...".format(nfig,'s' if nfig > 1 else ''),
          end='',flush=True)
    plt.show()
  if cpdf is not None:  cpdf.close()
  print("done")
  
  rest="""
  # In stan call:
  n_jobs=max(cpu_count()-1,1)
  
  # For multithreading (multiprocessing) support:
  extra_compile_args = ['-pthread', '-DSTAN_THREADS']
  # And obviously use:  extra_compile_args=extra_compile_args
  n_jobs=-1  # May change to chains
  
  
  
  # Can also save fit samples as csv:
  df = pystan.misc.to_dataframe(fit)
  df.to_csv("path/to/fit.csv", index=False)
  
  To integrate logging:
  
  # Set up pystan logging before importing pystan
  import logging
  logger = logging.getLogger("pystan")
  
  # add root logger (logger Level always Warning)
  logger.addHandler(logging.NullHandler())
  
  # Set up logging as wanted
  logger_path = "pystan.log"
  fh = logging.FileHandler(logger_path, encoding="utf-8")
  fh.setLevel(logging.INFO)
  # optional step
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  
  import pystan
  
  
  """
