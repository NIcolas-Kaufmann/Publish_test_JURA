import numpy as np 
import matplotlib.pyplot as plt 

from jura_example_code import rv_curve_class


rv = rv_curve_class(t0=0., p=10., e=0.5, w=np.pi/3, k=10., t_init=0., t_end=25.)
rv.plot()