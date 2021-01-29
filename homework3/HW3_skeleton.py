# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

#%% Q1




#%% Q2




#%% Q3




#%% Q4





#%% Q5





#%% Q6






#%% Q7





#%% Q8







#%% Q9

#%% load data
import pandas as pd 
measles=pd.read_csv('Measles.csv',header=None).values
mumps=pd.read_csv('Mumps.csv',header=None).values
chickenPox=pd.read_csv('chickenPox.csv',header=None).values

# close all existing floating figures
plt.close('all')

#%% Q9.a. plot annual total measles cases in each year

plt.figure()
plt.title('Fig 9.a: NYC measles cases')

# complete this part here

plt.show()

#%% Q9.b bar plot average mumps cases for each month of the year

plt.figure()
plt.title('Fig 9.b: Average monthly mumps cases')

# complete this part here

plt.show()


#%% Q9.c scatter plot monthly mumps cases against measles cases
mumpsCases = mumps[:, 1:].reshape(41*12)
measlesCases = measles[:, 1:].reshape(41*12)

plt.figure()
plt.title('Fig 9.c: Monthly mumps vs measles cases')

# complete this part here

plt.show()


#%% Q9.d plot monthly mumps cases against measles cases in log scale
plt.figure()
plt.title('Fig 9.d: Monthly mumps vs measles cases (log scale)')

# complete this part

plt.show()


#%% Answer to Q9.e

# complete this part here
answer = ''

print('\n\nAnswer to Q9.e: ' + answer)
