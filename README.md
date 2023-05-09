# Publish_test


### Jura workshop example 


### Usage instructions
```
#import packages
import numpy as np 
import matplotlib.pyplot as plt 
#import the rv curves 
from rv_curve_jura.jura_example_code import rv_curve_class


#create an instance of the class
rv = rv_curve_class(t0=0., p=10., e=0.5, w=np.pi/3, k=10., t_init=0., t_end=25.)
# use the plot method to plot the rv curve
rv.plot()
```