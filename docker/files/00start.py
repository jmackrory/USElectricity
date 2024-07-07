import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython

print("running 00start")
ipython = get_ipython()
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

if "/home/root" not in sys.path:
    sys.path.append("/home/root/")
