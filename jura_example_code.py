#!/usr/bin/env python
# coding: utf-8

# # Example code for the JURA IV workshop
# ### Oscar Barragán, May 2023



import numpy as np
import matplotlib.pylab as plt

#Create a function that computes the Radial Velocity curve
#We know the orbital parameters of the planet and the Doppler semi-amplitude that the planet imprints in the star.
#Time of periastron passage, t0
#Orbital Period, p
#Orbital Eccentricity, e
#Angle of periastron passage, w
#Doppler semi-amplitude, k
#We also know two times t_init, t_end, in which our star will be observed.
t0 = 0. #days but typically Barycentre Julian Date (BJD) 
p  = 10. #days
e  = 0.5 
w  = np.pi/3
k  = 10 #m/s
t_init = 0 #days
t_end = 25 #days

#Create a function that computes the Radial Velocity curve
#input parameters: t_init(float), t_end(float), t0 (float), 
#    p (float), e(float), w(float), k(float), npts (integer,optional) 
#output: t_vector, rv (array) --> The time and RV of the curve we want to plot 
def rv_curve(t_init,t_end,t0,p,e,w,k,npts=1000):
    t_vector = np.linspace(t_init,t_end,npts)
    rv = k * np.cos(nu + w) + e * np.cos(w)
    return t_vector, rv

time, rv = rv_curve(t_init,t_end,t0,p,e,w,k)


# Code that appears on slide 24

# In[ ]:


#time to create the function that computes the true anomaly from the mean anomaly
#This function calculates the true anomaly
#input parameters: t (float), t0 (float), 
#    p (float), e(float)
#output: true (array) --> The true anomaly of the planetary orbit
def true_anomaly(t,t0,e,p):
    mean = 2.*np.pi * ( t - t0) / p                       #mean anomaly
    true = mean + e * np.sin(mean)                        #guess
    f = true - e * np.sin(true) - mean                    #first value of function f
    for i in range(len(t)):                               #iterate for all the values
        while np.abs(f[i]) > 1e-6:                        #Newton-Raphson condition
            f[i] = true[i] - e*np.sin(true[i]) - mean[i]  #calculate  f
            df   = 1. - e * np.cos(true[i])               #Calculate df
            true[i] = true[i] - f[i]/df                   #Update the eccentric anomaly
    eimag = np.sqrt(1. - e*e)*np.sin(true)                #Time to calculate true anomaly
    ereal = np.cos(true) - e
    true  = np.arctan2(eimag,ereal)                       #Get True anomaly from ecc anomaly
    return true

#Create a function that computes the Radial Velocity curve
#input parameters: t_init(float), t_end(float), t0 (float), 
#    p (float), e(float), w(float), k(float), npts (integer,optional) 
#output: t_vector, rv (array) --> The time and RV of the curve we want to plot 
def rv_curve(t_init,t_end,t0,p,e,w,k,npts=1000):
    #Get the time vector given t_init, t_end, and the number of points
    t_vector = np.linspace(t_init,t_end,npts)
    #Get the true a nomaly from the true_anomaly function
    nu =  true_anomaly(t_vector,t0,e,p)
    #Compute the RV curve
    rv = k * np.cos(nu + w) + e * np.cos(w)
    return t_vector, rv


# Code that appears on slide 25

# In[ ]:


import numpy as np
import matplotlib.pylab as plt

#Create a function that computes the Radial Velocity curve
#We know the orbital parameters of the planet and the Doppler semi-amplitude that the planet imprints in the star.
#Time of periastron passage, t0
#Orbital Period, p
#Orbital Eccentricity, e
#Angle of periastron passage, w
#Doppler semi-amplitude, k
#We also know two times t_init, t_end, in which our star will be observed.
t0 = 0. #days but typically Barycentre Julian Date (BJD) 
p  = 10. #days
e  = 0.5 
w  = np.pi/3
k  = 10 #m/s
t_init = 0 #days
t_end = 25 #days

#time to create the function that computes the true anomaly from the mean anomaly
#This function calculates the true anomaly
#input parameters: t (float), t0 (float), 
#    p (float), e(float)
#output: true (array) --> The true anomaly of the planetary orbit
def true_anomaly(t,t0,e,p):
    mean = 2.*np.pi * ( t - t0) / p                       #mean anomaly
    true = mean + e * np.sin(mean)                        #guess
    f = true - e * np.sin(true) - mean                    #first value of function f
    for i in range(len(t)):                               #iterate for all the values
        while np.abs(f[i]) > 1e-6:                        #Newton-Raphson condition
            f[i] = true[i] - e*np.sin(true[i]) - mean[i]  #calculate  f
            df   = 1. - e * np.cos(true[i])               #Calculate df
            true[i] = true[i] - f[i]/df                   #Update the eccentric anomaly
    eimag = np.sqrt(1. - e*e)*np.sin(true)                #Time to calculate true anomaly
    ereal = np.cos(true) - e
    true  = np.arctan2(eimag,ereal)                       #Get True anomaly from ecc anomaly
    return true

#Create a function that computes the Radial Velocity curve
#input parameters: t_init(float), t_end(float), t0 (float), 
#    p (float), e(float), w(float), k(float), npts (integer,optional) 
#output: t_vector, rv (array) --> The time and RV of the curve we want to plot 
def rv_curve(t_init,t_end,t0,p,e,w,k,npts=1000):
    #Get the time vector given t_init, t_end, and the number of points
    t_vector = np.linspace(t_init,t_end,npts)
    #Get the true a nomaly from the true_anomaly function
    nu =  true_anomaly(t_vector,t0,e,p)
    #Compute the RV curve
    rv = k * np.cos(nu + w) + e * np.cos(w)
    return t_vector, rv

t, rv = rv_curve(t_init,t_end,t0,p,e,w,k)

# plot the RV curve
plt.plot(t, rv)
plt.xlabel("Time [d]")
plt.ylabel("RV [m/s]")
plt.show()


# ### Python Classes

# Code that appears on slide 37

# In[ ]:


#Dog example
class dog:
    
    #Define the attributes by calling the __init__ function
    def __init__(self,name,age):
        #We can use the input variables as attributes
        self.name = name
        self.age = age
        #We can also create default attributes
        self.awake = True
        
        
    #Create the method that controls the sleep pattern of the dog
    def order(self,order):
        if order == 'sleep':
            if self.awake:
                print('Guau (OK, I will sleep, I am tired)')
                #We can modify attributes inside the instances
                self.awake = False
            else:
                pass #The dog is already sleeping, cannot talk
        if order == 'awake':
            if self.awake:
                print('Guau guau guau (I am awake, and hungry, feed me!)')
            else:
                print('Guaaaau (Good morning!)')
                #We can modify attributes inside the instances
                self.awake = True 


# Code that appears on slide 38

# In[ ]:


#Let us create an instance called timon, using the class dog
timon = dog(name='timon',age=17)
print(timon.name)
print(timon.age)
print(timon.awake)


# In[ ]:


#let us create an instance called lobo, using the class dog
lobo = dog(name='lobo',age=10)
print(lobo.name)
print(lobo.age)
print(lobo.awake)


# Code that appears on slide 39

# In[ ]:


#Dog example
class dog:
    
    #Define the attributes by calling the __init__ function
    def __init__(self,name,age):
        #We can use the input variables as attributes
        self.name = name
        self.age = age
        #We can also create default attributes
        self.awake = True
        
        
    #Create the method that controls the sleep pattern of the dog
    def order(self,order):
        if order == 'sleep':
            if self.awake:
                print('Guau (OK, I will sleep, I am tired)')
                #We can modify attributes inside the instances
                self.awake = False
            else:
                pass #The dog is already sleeping, cannot talk
        if order == 'awake':
            if self.awake:
                print('Guau guau guau (I am awake, and hungry, feed me!)')
            else:
                print('Guaaaau (Good morning!)')
                #We can modify attributes inside the instances
                self.awake = True 


# Code that appears on slide 40

# In[ ]:


timon.order('sleep')


# In[ ]:


print(timon.awake)


# In[ ]:


timon.order('sleep')


# In[ ]:


timon.order('awake')


# In[ ]:


print(timon.awake)


# In[ ]:


timon.order('awake')


# # Transforming the rv_code to  code to a python class 

# Code that appears on slide 42, 43, 44

# In[ ]:


import numpy as np
import matplotlib.pylab as plt

class rv_curve_class:
    
    """
    We know the orbital parameters of the planet and the Doppler 
    semi-amplitude that the planet imprints in the star.
    Time of periastron passage, t0
    Orbital Period, p
    Orbital Eccentricity, e
    Angle of periastron passage, w
    Doppler semi-amplitude, k
    We can also indicate two times t_init, t_end, in which our star will be observed,
    and the number of points npts to use to create our data.
    """
    
    def __init__(self, t0, p, e, w, k, t_init=0, t_end=25,npts=1000):
        #Asing all the input variables as attributes of the class
        self.t0 = t0
        self.p = p
        self.e = e
        self.w = w
        self.k = k
        self.t_init = t_init
        self.t_end = t_end
        self.npts = npts
        #We can create a new attribute using previous attributes
        self.t_vector = np.linspace(self.t_init, self.t_end, npts)    # get time vector

    
    # method to compute the true anomaly from the mean anomaly
    def true_anomaly(self):
        mean = 2. * np.pi * (self.t_vector - self.t0) / self.p   # mean anomaly
        true = mean + self.e * np.sin(mean)          # initial guess for true anomaly
        f = true - self.e * np.sin(true) - mean      # first value of function f
        for i in range(len(self.t_vector)):                     # iterate for all the values
            while np.abs(f[i]) > 1e-6:              # Newton-Raphson condition
                f[i] = true[i] - self.e * np.sin(true[i]) - mean[i]  # calculate f
                df = 1. - self.e * np.cos(true[i])  # calculate df
                true[i] = true[i] - f[i]/df         # update the eccentric anomaly
        eimag = np.sqrt(1. - self.e*self.e) * np.sin(true)    # calculate imaginary part of eccentric anomaly
        ereal = np.cos(true) - self.e                         # calculate real part of eccentric anomaly
        true = np.arctan2(eimag, ereal)                       # get true anomaly from eccentric anomaly
        return true
    
    # function to compute the radial velocity curve
    def rv_curve(self):
        #Note how when we call the true_anomaly method, we do not need to give any input parameter
        #All the parameters are attributes of the instance, so they are used in the method computationscre
        nu = self.true_anomaly()                         # get true anomaly from true_anomaly function
        rv = self.k * np.cos(nu + self.w) + self.e * np.cos(self.w)  # compute RV curve
        return self.t_vector, rv


# Code that appears on slide 45

# In[ ]:


# create instance of the class
rv = rv_curve_class(t0=0., p=10., e=0.5, w=np.pi/3, k=10., t_init=0., t_end=25.)

# compute the RV curve
t, rv = rv.rv_curve()

# plot the RV curve
plt.plot(t, rv)
plt.xlabel("Time [d]")
plt.ylabel("RV [m/s]")
plt.show()


# Code that appears on slide 47

# In[ ]:


import numpy as np
import matplotlib.pylab as plt

class rv_curve_class:
    
    """
    We know the orbital parameters of the planet and the Doppler 
    semi-amplitude that the planet imprints in the star.
    Time of periastron passage, t0
    Orbital Period, p
    Orbital Eccentricity, e
    Angle of periastron passage, w
    Doppler semi-amplitude, k
    We can also indicate two times t_init, t_end, in which our star will be observed,
    and the number of points npts to use to create our data.
    """
    
    def __init__(self, t0, p, e, w, k, t_init=0, t_end=25,npts=1000):
        #Asing all the input variables as attributes of the class
        self.t0 = t0
        self.p = p
        self.e = e
        self.w = w
        self.k = k
        self.t_init = t_init
        self.t_end = t_end
        self.npts = npts
        #We can create a new attribute using previous attributes
        self.t_vector = np.linspace(self.t_init, self.t_end, npts)    # get time vector

    
    # method to compute the true anomaly from the mean anomaly
    def true_anomaly(self):
        mean = 2. * np.pi * (self.t_vector - self.t0) / self.p   # mean anomaly
        true = mean + self.e * np.sin(mean)          # initial guess for true anomaly
        f = true - self.e * np.sin(true) - mean      # first value of function f
        for i in range(len(self.t_vector)):                     # iterate for all the values
            while np.abs(f[i]) > 1e-6:              # Newton-Raphson condition
                f[i] = true[i] - self.e * np.sin(true[i]) - mean[i]  # calculate f
                df = 1. - self.e * np.cos(true[i])  # calculate df
                true[i] = true[i] - f[i]/df         # update the eccentric anomaly
        eimag = np.sqrt(1. - self.e*self.e) * np.sin(true)    # calculate imaginary part of eccentric anomaly
        ereal = np.cos(true) - self.e                         # calculate real part of eccentric anomaly
        true = np.arctan2(eimag, ereal)                       # get true anomaly from eccentric anomaly
        return true
    
    # function to compute the radial velocity curve
    def rv_curve(self):
        #Note how when we call the true_anomaly method, we do not need to give any input parameter
        #All the parameters are attributes of the instance, so they are used in the method computationscre
        nu = self.true_anomaly()                         # get true anomaly from true_anomaly function
        #Note how we can add the rv attribute to the instance
        self.rv = self.k * np.cos(nu + self.w) + self.e * np.cos(self.w)  # compute RV curve

    #Let's create a new method to plot the light curve
    def plot(self):  
        #Let us compute the curve
        self.rv_curve()
        # plot the RV curve
        plt.plot(self.t_vector, self.rv)
        plt.xlabel("Time [d]")
        plt.ylabel("RV [m/s]")
        plt.show()


# In[ ]:


rv = rv_curve_class(t0=0., p=10., e=0.5, w=np.pi/3, k=10., t_init=0., t_end=25.)
rv.plot()

