#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT 01
# ## Question 2
# 
# ### Computational Fluid Dynamics (ME543)     
# ### NIRMAL S.  [234103107]

# In[1]:


import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

import time


# In[2]:


# INPUTS

X = 1     # distance on X axis
Y = 1     # distance on Y axis

delta_x = 0.05     # node to node length in X-axis
delta_y = 0.05     # node to node length in Y-axis

M = int(X/delta_x)+1     # number of elements in X-axis (number of columns)
N = int(Y/delta_y)+1     # number of elements in Y-axis (number of rows)

beta = delta_x/delta_y
beta_sq = beta**2

epsilon = 10**(-6)

print("M = ", M, ",   N = ", N)


# In[3]:


# CREATION OF DOMAIN MATRIX

DOMAIN = np.zeros((M, N))     # creating array with initial values as zeros


# In[4]:


# Inputs for the Boundary Conditions

# Bottom BC:
T11 = 1      # Temp on bottom Boundary

# Left BC:
T21 = 1     # Temp on left Boundary

# Top BC:
T31 = 0     # Temp on top Boundary

# Right BC:
T41 = 1     # Temp on right Boundary


# In[5]:


# Implimenting the Boundary Values


for i in range(M):          # Bottom and Top Boundary Conditions
    
    DOMAIN[i,N-1] = T31     # TOP BC
    DOMAIN[i,0]   = T11     # BOTTOM BC
    

for j in range(N):          # Left and Right Boundary Conditions
    
    DOMAIN[0,j]   = T21       # LEFT BC
    DOMAIN[M-1,j] = T41       # RIGHT BC


# In[50]:


## Getting the Exact Solution from the formula:

Tc = T31    # Top Temp
Th = T11    # Bottom Temp

L = X    # X length
H = Y    # Y length

Exact_domain = DOMAIN.copy()

for i in range(1, M-1):
    for j in range(1, N-1):
        
        T = 0
        for n in range(1, 100):
            T += ((1-(-1)**n)/(n*math.pi)
                  *(math.sinh((n*math.pi*j*delta_y/L)))/(math.sinh((n*math.pi*H/L)))
                  *(math.sin((n*math.pi*i*delta_x/L))))
            
        Exact_domain[i, j] = Tc + (Th - Tc)*(1 - 2*T)
        
        
### Creating the .plt file ###
with open("Temp-Z_Exact.plt", "w") as f:
    f.write("VARIABLES = \"X\", \"Y\", \"Temp\"\n")
    f.write("ZONE T = \"BLOCK1\", I = %d, J = %d, F = POINT\n"%(M, N))
    
    for j in range(N):
        for i in range(M):
            f.write("%f\t%f\t%f\n"%(i*delta_x,  j*delta_y, Exact_domain[i,j]))


# In[45]:


# Creating DataFrame to store values of 
##  Iterations vs Errors of all Methods

IterError = pd.DataFrame(columns=['Iteration', 
                                  'Jacobian', 
                                  'Point Gauss-Seidel', 
                                  'PSOR', 
                                  'Line Gauss-Seidel', 
                                  'ADI'], 
                         dtype='float')

TimeToRun = pd.DataFrame(columns=['Method',
                                  'Iterations',
                                  'Time (ms)', 
                                  'Time/iter (ms)'], 
                         dtype='float')


# In[46]:


## Iteration loop of JACOBI Iterative Method

start_t = time.time()

iter_no = 0
error_arr = []

J_domain_prev = DOMAIN.copy()

while True:
    
    J_domain = J_domain_prev.copy()
    error_arr.append(0)
    
    for j in range(1, N-1):
        for i in range(1, M-1):
            
            J_domain[i,j] = 1/(2*(1+beta_sq))*( (beta_sq)*J_domain_prev[i,j-1] + J_domain_prev[i-1,j] 
                                       + J_domain_prev[i+1,j] + (beta_sq)*J_domain_prev[i,j+1] )
            
            error_arr[iter_no] += (J_domain[i,j] - J_domain_prev[i,j])**2
            
    error_arr[iter_no] = math.sqrt(error_arr[iter_no]/((M-2)*(N-2)))
        

     
    if iter_no>2000:
        break
    if error_arr[iter_no]<=epsilon:
        break

    iter_no += 1
    
    J_domain_prev = J_domain.copy()
    
    
end_t = time.time()


IterError['Jacobian'] = pd.Series(error_arr)


### Creating the .plt file ###
with open("Temp-Jacobian.plt", "w") as f:
    f.write("VARIABLES = \"X\", \"Y\", \"Temp\"\n")
    f.write("ZONE T = \"BLOCK1\", I = %d, J = %d, F = POINT\n"%(M, N))
    
    for j in range(N):
        for i in range(M):
            f.write("%f\t%f\t%f\n"%(i*delta_x,  j*delta_y, J_domain[i,j]))

            
TimeToRun.loc[0, 'Method'] = "Jacobian"
TimeToRun.loc[0, 'Iterations'] = iter_no
TimeToRun.loc[0, 'Time (ms)'] = ((end_t-start_t)*1000)
TimeToRun.loc[0, 'Time/iter (ms)'] = (end_t-start_t)*1000 / iter_no
print("Time taken to run Jacobian: %.3f ms"%((end_t-start_t)*1000))


# In[45]:


## Iteration loop of POINT GAUSS-SEIDEL Iterative Method

start_t = time.time()

iter_no = 0
error_arr = []

PGS_domain_prev = DOMAIN.copy()

while True:
    
    PGS_domain = PGS_domain_prev.copy()
    error_arr.append(0)
    
    for j in range(1, N-1):
        for i in range(1, M-1):
            
            PGS_domain[i,j] = 1/(2*(1+beta_sq))*( (beta_sq)*PGS_domain[i,j-1] + PGS_domain[i-1,j] 
                                       + PGS_domain_prev[i+1,j] + (beta_sq)*PGS_domain_prev[i,j+1] )
            
            error_arr[iter_no] += (PGS_domain[i,j] - PGS_domain_prev[i,j])**2
            
    error_arr[iter_no] = math.sqrt(error_arr[iter_no]/((M-2)*(N-2)))
        
        
     
    if iter_no>2000:
        break
    if error_arr[iter_no]<=epsilon:
        break

    iter_no += 1
    
    PGS_domain_prev = PGS_domain.copy()
    
    
end_t = time.time()


IterError['Point Gauss-Seidel'] = pd.Series(error_arr)


### Creating the .plt file ###
with open("Temp-Point Gauss-Seidel.plt", "w") as f:
    f.write("VARIABLES = \"X\", \"Y\", \"Temp\"\n")
    f.write("ZONE T = \"BLOCK1\", I = %d, J = %d, F = POINT\n"%(M, N))
    
    for j in range(N):
        for i in range(M):
            f.write("%f\t%f\t%f\n"%(i*delta_x,  j*delta_y, PGS_domain[i,j]))

            
TimeToRun.loc[1, 'Method'] = "Point Gauss-Seidel"
TimeToRun.loc[1, 'Iterations'] = iter_no
TimeToRun.loc[1, 'Time (ms)'] = ((end_t-start_t)*1000)
TimeToRun.loc[1, 'Time/iter (ms)'] = (end_t-start_t)*1000 / iter_no
print("Time taken to run Point Gauss-Seidel: %.3f ms"%((end_t-start_t)*1000))


# In[46]:


## Iteration loop of PSOR Iterative Method

start_t = time.time()

w = 1.85

iter_no = 0
error_arr = []

PSOR_domain_prev = DOMAIN.copy()

while True:
    
    PSOR_domain = PSOR_domain_prev.copy()
    error_arr.append(0)
    
    for j in range(1, N-1):
        for i in range(1, M-1):
            
            PSOR_domain[i,j] = ((1-w)*PSOR_domain_prev[i,j] 
                                + w/(2*(1+beta_sq))*( (beta_sq)*PSOR_domain[i,j-1] + PSOR_domain[i-1,j] 
                                       + PSOR_domain_prev[i+1,j] + (beta_sq)*PSOR_domain_prev[i,j+1] ))
            
            error_arr[iter_no] += (PSOR_domain[i,j] - PSOR_domain_prev[i,j])**2
            
    error_arr[iter_no] = math.sqrt(error_arr[iter_no]/((M-2)*(N-2)))
        
        
     
    if iter_no>1000:
        break
    if error_arr[iter_no]<=epsilon:
        break

    iter_no += 1
    
    PSOR_domain_prev = PSOR_domain.copy()
    
    
end_t = time.time()


IterError['PSOR'] = pd.Series(error_arr)


### Creating the .plt file ###
with open("Temp-PSOR.plt", "w") as f:
    f.write("VARIABLES = \"X\", \"Y\", \"Temp\"\n")
    f.write("ZONE T = \"BLOCK1\", I = %d, J = %d, F = POINT\n"%(M, N))
    
    for j in range(N):
        for i in range(M):
            f.write("%f\t%f\t%f\n"%(i*delta_x,  j*delta_y, PSOR_domain[i,j]))

            
TimeToRun.loc[2, 'Method'] = "PSOR"
TimeToRun.loc[2, 'Iterations'] = iter_no
TimeToRun.loc[2, 'Time (ms)'] = ((end_t-start_t)*1000)
TimeToRun.loc[2, 'Time/iter (ms)'] = (end_t-start_t)*1000 / iter_no
print("Time taken to run PSOR: %.3f ms"%((end_t-start_t)*1000))


# In[47]:


def TDMA(a,b,c,d):
    n = len(d)
    P = np.zeros(n-1, float)
    Q = np.zeros(n, float)
    updated_val = np.zeros(n, float)
    
    P[0] = -c[0]/b[0]
    Q[0] = d[0]/b[0]

    ### Calculation of P and Q
    for i in range(1,n-1):
        P[i] = -c[i]/(b[i] + a[i-1]*P[i-1])
    for i in range(1,n):
        Q[i] = (d[i] - a[i-1]*Q[i-1])/(b[i] + a[i-1]*P[i-1])
        
    ### Back Substitution of Phi
    updated_val[n-1] = Q[n-1]
    for i in reversed(range(n-1)):
        updated_val[i] = P[i]*updated_val[i+1] + Q[i]
        
    return updated_val


# In[48]:


## Iteration loop of LINE GAUSS-SEIDEL Iterative Method

start_t = time.time()

iter_no = 0
error_arr = []

LGS_domain = DOMAIN.copy()

while True:
    
    error_arr.append(0)
    
    #################### X - SWEEP ####################
    a = [1]*(M-2)
    b = [-2*(1+beta_sq)]*(M-2)
    c = [1]*(M-2)
    d = [0]*(M-2)

    ### Since there are values to the right and left of the updation area, we can't take a[0] and c[N] as zero
#     a[0] = 0
#     c[-1] = 0

    for j in range(1, N-1):
        
        ### Calculating values of d ###
        for i in range(1, M-1):     
            d[i-1] = (-beta_sq)*(LGS_domain[i, j-1] + LGS_domain[i, j+1])
        # Considering the right and left boundary cases
        d[0]  = (-beta_sq)*(LGS_domain[0, j-1] + LGS_domain[0, j+1]) - a[0]*LGS_domain[0, j]
        d[-1] = (-beta_sq)*(LGS_domain[M-2, j-1] + LGS_domain[M-2, j+1]) - c[-1]*LGS_domain[M-1, j]  
        
        ### Applying TDMA to update row ###
        updated_row = TDMA(a, b, c, d)
     
        for i in range(1, M-1):
            error_arr[iter_no] += (updated_row[i-1] - LGS_domain[i,j])**2
        
        LGS_domain[1:-1, j] = updated_row.copy()
            
    error_arr[iter_no] = math.sqrt(error_arr[iter_no]/((M-2)*(N-2)))
        
     
    
    if iter_no>1000:
        break
    if error_arr[iter_no]<=epsilon:
        break

    iter_no += 1
    
end_t = time.time()


IterError['Line Gauss-Seidel'] = pd.Series(error_arr)

### Creating the .plt file ###
with open("Temp-Line Gauss-Seidel.plt", "w") as f:
    f.write("VARIABLES = \"X\", \"Y\", \"Temp\"\n")
    f.write("ZONE T = \"BLOCK1\", I = %d, J = %d, F = POINT\n"%(M, N))
    
    for j in range(N):
        for i in range(M):
            f.write("%f\t%f\t%f\n"%(i*delta_x,  j*delta_y, LGS_domain[i,j]))

            
TimeToRun.loc[3, 'Method'] = "Line Gauss-Seidel"
TimeToRun.loc[3, 'Iterations'] = iter_no
TimeToRun.loc[3, 'Time (ms)'] = ((end_t-start_t)*1000)
TimeToRun.loc[3, 'Time/iter (ms)'] = (end_t-start_t)*1000 / iter_no
print("Time taken to run Line Gauss-Seidel: %.3f ms"%((end_t-start_t)*1000))


# In[49]:


## Iteration loop of Alternating Direction Implicit (ADI) Method

start_t = time.time()

iter_no = 0
error_arr = []

ADI_domain = DOMAIN.copy()

while True:
    
    ADI_domain_prev = ADI_domain.copy()    ## (used only to calculate the error after each iteration)
    
    error_arr.append(0)
            
    #################### X - SWEEP ####################
    a = [1]*(M-2)
    b = [-2*(1+beta_sq)]*(M-2)
    c = [1]*(M-2)
    d = [0]*(M-2)

    ### Since there are values to the right and left of the updation area, we can't take a[0] and c[N] as zero
#     a[0] = 0
#     c[-1] = 0

    for j in range(1, N-1):
        
        ### Calculating values of d ###
        for i in range(1, M-1):     
            d[i-1] = (-beta_sq)*(ADI_domain[i, j-1] + ADI_domain[i, j+1])
        # Considering the right and left boundary cases
        d[0]  = (-beta_sq)*(ADI_domain[0, j-1] + ADI_domain[0, j+1]) - a[0]*ADI_domain[0, j]
        d[-1] = (-beta_sq)*(ADI_domain[M-2, j-1] + ADI_domain[M-2, j+1]) - c[-1]*ADI_domain[M-1, j]       
        
        ### Applying TDMA to update row ###
        updated_row = TDMA(a, b, c, d)
        
        ADI_domain[1:-1, j] = updated_row.copy()
        
        
            
    #################### Y - SWEEP ####################
    a = [beta_sq]*(N-2)
    b = [-2*(1+beta_sq)]*(N-2)
    c = [beta_sq]*(N-2)
    d = [0]*(N-2)

    ### Since there are values to the right and left of the updation area, we can't take a[0] and c[N] as zero
#     a[0] = 0
#     c[N-1] = 0
    
    for i in range(1, M-1):
#     for i in range(1, 2):

        ### Calculating values of d ###
        for j in range(1, N-1):
            d[j-1] = (-1)*(ADI_domain[i-1, j] + ADI_domain[i+1, j])
        # Considering the top and bottom boundary cases
        d[0]  = (-1)*(ADI_domain[i-1, 0] + ADI_domain[i+1, 0]) - a[0]*ADI_domain[i, 0]
        d[-1] = (-1)*(ADI_domain[i-1, N-2] + ADI_domain[i+1, N-2]) - c[-1]*ADI_domain[i, N-1]

        ### Applying TDMA to update col ###
        updated_col = TDMA(a, b, c, d)
        
        ### Replace the row with the updated row
        ADI_domain[i, 1:-1] = updated_col.copy()
        
        
        
    #################### Error Calculation ####################
    for i in range(1, M-1):
        for j in range(1, N-1):
            error_arr[iter_no] += (ADI_domain[i,j] - ADI_domain_prev[i,j])**2
            
    error_arr[iter_no] = math.sqrt(error_arr[iter_no]/((M-2)*(N-2)))
        

    if iter_no>1000:
#     if iter_no>=0:
        break
    if error_arr[iter_no]<=epsilon:
        break

    iter_no += 1
    
end_t = time.time()


IterError['ADI'] = pd.Series(error_arr)

### Creating the .plt file ###
with open("Temp-ADI.plt", "w") as f:
    f.write("VARIABLES = \"X\", \"Y\", \"Temp\"\n")
    f.write("ZONE T = \"BLOCK1\", I = %d, J = %d, F = POINT\n"%(M, N))
    
    for j in range(N):
        for i in range(M):
            f.write("%f\t%f\t%f\n"%(i*delta_x,  j*delta_y, ADI_domain[i,j]))

            
TimeToRun.loc[4, 'Method'] = "ADI"
TimeToRun.loc[4, 'Iterations'] = iter_no
TimeToRun.loc[4, 'Time (ms)'] = ((end_t-start_t)*1000)
TimeToRun.loc[4, 'Time/iter (ms)'] = (end_t-start_t)*1000 / iter_no
print("Time taken to run ADI: %.3f ms"%((end_t-start_t)*1000))


# In[50]:


# sns.heatmap(pd.DataFrame(J_domain.transpose()[::-1]), cmap="Blues")
# sns.heatmap(pd.DataFrame(PGS_domain.transpose()[::-1]), cmap="Blues")
# sns.heatmap(pd.DataFrame(PSOR_domain.transpose()[::-1]), cmap="Blues")
# sns.heatmap(pd.DataFrame(LGS_domain.transpose()[::-1]), cmap="Blues")
sns.heatmap(pd.DataFrame(ADI_domain.transpose()[::-1]), cmap="Blues")


# plt.show()


# In[51]:


# Weight vs Iteration dataframe

IterWeight = pd.DataFrame(columns=['Weight', 
                                   'Iterations'], 
                          dtype='float')


# In[52]:


## PSOR Finding Optimum weight

w = 1
index = 0

step = 0.05

while True:
    
    iter_no = 0
    error = 0.0

    PSOR_domain_prev = DOMAIN.copy()

    while True:
        
        error = 0.0
        
        PSOR_domain = PSOR_domain_prev.copy()

        for j in range(1, N-1):
            for i in range(1, M-1):

                PSOR_domain[i,j] = (1-w)*PSOR_domain_prev[i,j] + w/(2*(1+beta_sq))*( (beta_sq)*PSOR_domain[i,j-1] + PSOR_domain[i-1,j] 
                                           + PSOR_domain_prev[i+1,j] + (beta_sq)*PSOR_domain_prev[i,j+1] )

                error += (PSOR_domain[i,j] - PSOR_domain_prev[i,j])**2

        error = math.sqrt(error/((M-2)*(N-2)))  ## ##


        # RIGHT Boundary Condition (Back Difference second order df/dx=0)
        for j in range(N):
            PSOR_domain[M-1,j] = (1/3)*(4*PSOR_domain[M-2,j] - PSOR_domain[M-3,j])
               
        if iter_no>2000:
            break
        if error<=epsilon:
            break

        iter_no += 1

        PSOR_domain_prev = PSOR_domain.copy() 
    
    IterWeight.loc[index,'Iterations'] = iter_no
    IterWeight.loc[index,'Weight'] = w
    
    w += step
    index += 1
    
    if w>=2:
        break


# In[53]:


## Plotting Weights vs Iterations in PSOR Iterative Method

plt.figure(figsize=(10,6))

plt.plot(IterWeight['Iterations'], IterWeight['Weight'], '-o')

plt.title("Weights vs Iterations")

plt.xlabel("Iterations")
plt.ylabel("Weight")

x_iter_w_plot = IterWeight['Iterations'].min()
y_iter_w_plot = IterWeight[IterWeight['Iterations']==x_iter_w_plot]['Weight']
plt.text(x_iter_w_plot, y_iter_w_plot, "  <----- w_opt = %.2f" %y_iter_w_plot)

plt.grid()

plt.savefig("Temp-Weights_Iter.jpg")


# In[54]:


## Calculating the optimum weight using formula

a_temp = (math.cos(math.pi/(M-1)) + beta_sq*math.cos(math.pi/(N-1)))/(1+beta_sq)

w_opt = 2*(1-math.sqrt(1-a_temp))/a_temp

print("w_opt = %.3f" %w_opt)


# In[55]:


## Plotting the log10(error) vs log10(iterations)

plt.figure(figsize=(10,6))


IterError['Iteration'] = pd.Series(IterError.index+1)
logIter = IterError['Iteration'].apply(lambda x: math.log(x))

logJacobian = IterError['Jacobian'].apply(lambda x: math.log10(x))
logPGS = IterError['Point Gauss-Seidel'].apply(lambda x: math.log10(x))
logLGS = IterError['Line Gauss-Seidel'].apply(lambda x: math.log10(x))
logADI = IterError['ADI'].apply(lambda x: math.log10(x))
logPSOR = IterError['PSOR'].apply(lambda x: math.log10(x))


plt.plot(logIter, logJacobian, color='red', label='Jacobian')
plt.plot(logIter, logPGS, color='orange', label='Point GS')
plt.plot(logIter, logLGS, color='yellow', label='Line GS')
plt.plot(logIter, logADI, color='green', label='ADI')
plt.plot(logIter, logPSOR, color='blue', label='PSOR')

plt.title("Log(Error) vs Log(Iterations)")

plt.xlabel("Log(Iterations)")
plt.ylabel("Log(Error)")

plt.legend()

plt.savefig("Temp-Error_vs_Iter.jpg")

plt.show()


# In[56]:


TimeToRun.to_csv("Temp-Time_to_run.csv", index=False)


# In[ ]:




