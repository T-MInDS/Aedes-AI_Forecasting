# autopep8: off

import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scipy.stats import nbinom, poisson

# autopep8: on

def poisson_quant(predictions, ci):
    u_quant = poisson.ppf(0.5 + ci / 2, mu=predictions)
    l_quant = poisson.ppf(0.5 - ci / 2, mu=predictions)
    return l_quant, u_quant

def negbin_quant(ns, mean_p, ci):
    u_quant = nbinom.ppf(0.5 + ci / 2, n=ns, p=mean_p)
    l_quant = nbinom.ppf(0.5 - ci / 2, n=ns, p=mean_p)
    return l_quant, u_quant