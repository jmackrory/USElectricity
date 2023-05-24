import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from IPython import get_ipython

ipython = get_ipython()
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

if "/tf" not in sys.path:
    sys.path.append("/tf/")
