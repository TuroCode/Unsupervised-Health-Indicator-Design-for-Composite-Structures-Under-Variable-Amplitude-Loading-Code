import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from numba import jit,njit
#Author: Arturo Rull Nomen
warnings.filterwarnings('ignore')


###In case of questions regarding this and implementation, ask Arturo
"""TO use this copy ono your progam, note that njit requires your funcntion to be on the file"""


def PMT_report(feature,window_time,coefficients, Fit=False,):
    """
    Returns an array [prognosability,Monotonocity,Trendability] of features if Fit=False, and returns
    the Fit factor with defined coefficients if set to True.
    
    The feature will be a (n_samples,number of feautures recorded per column) array
    """
    #print("Starting PMT analysis")
    #print(coefficients)
    a=coefficients[0]
    b=coefficients[1]
    c=coefficients[2]
    #print(coefficients)
    #Ensures the time arrray refernced will always be within range
    length_data=np.zeros(feature.shape[0])
    for i,run in enumerate(feature):
        length_data[i]=len(run)
    max_len=np.max(length_data)
    time=np.arange(0,max_len*window_time,window_time)



    ###Attain monotonicity of a single distribution of features and cycle through different samples, take then the average of all samples.
    #print("Computing monotonicity...")
    n_samples=np.shape(feature)[0]

    Monotonicity_arr=np.zeros(n_samples)
    #print(time)
    
    for k in range(n_samples): #Loop through every sample
        Num=0
        Den=0
        feature_k=feature[k,:]
        time_k=time[feature_k!=0] #Sizes the time accordingly to the trimmed feature
        feature_k=feature_k[feature_k!=0] #Takes zeros out

        n_measurements=len(feature_k)
        

        #Attains the time and measuremenents arrays into 1d for this part


        for i in range(n_measurements):

            #Do not forget j=0 always (see definition)
            if i!=0:
                Num+=(time_k[0]-time_k[i])*np.sign(feature_k[0]-feature_k[i])

            for j in range(i+1,n_measurements):
                Num+=(time_k[j]-time_k[i])*np.sign(feature_k[j]-feature_k[i])
                Den+=time_k[j]-time_k[i]
        #Once the sums have been performed, will compute the value of monotonicity
        Monotonicity_arr[k]=(Num/Den)


    Monotonicity=np.sum(Monotonicity_arr)/n_samples #Computes the average of all values of Monotonicity

    ###Prognosability by means computing the standard deviation of the final values of the feature and apply the definition of trensability

    #print("Computing prognosability...")
    initial_vals=np.zeros(n_samples)
    final_vals=np.zeros(n_samples) 
    for i in range(n_samples):
        feature_i=feature[i,:]
        feature_i=feature_i[feature_i!=0]
        initial_vals[i]=feature_i[0]
        final_vals[i]=feature_i[-1] #Attains a list of the final values


    Prognosability= np.e**(-np.std(final_vals)/np.mean(np.abs(initial_vals-final_vals))) #Accounting for the definition from the lit.review

    ###Attain trendability

    #print("Computing trendability...")
    z_val=np.zeros(n_samples)

    for k in range(n_samples): #Loop through every sample 

        ###Attain an individual 1D array with the required length for the given sample 
        feature_k=feature[k,:]
        time_k=time[feature_k!= 0]
        feature_k=feature_k[feature_k!= 0]
        

        n_measurements=len(feature_k)


        #Compute first and deconde derivatives of the feature
        dydt=(feature_k[1:]-feature_k[:-1])/(time_k[1:]-time_k[:-1]) #first order approxiamtion of the first derivative of the dataset... rough approximations
        dy2dt2=(feature_k[2:]-2*feature_k[1:-1]+feature_k[:-2])/((time_k[2:]-time_k[1:-1]+time_k[:-2]-time_k[1:-1])/2)**2 #Second order approxiamtion of the second derivative of the feature (see https://en.wikipedia.org/wiki/Finite_difference)
        
        #Count values where greater than zero

        n1d=len(np.where(dydt>0)[0])
        n2d=len(np.where(dy2dt2>0)[0])
        D=n_measurements
        val=n1d/(D-1)+n2d/(D-2) #Definition taken from (DOI: 10.4233/uuid:538558fb-ac9a-414d-8a59-4b523d8ff74c)
        z_val[k]=val

    #Once the values of Z have been defined for every sample, the std of this variable is computed and trendability is attained
    Trendability=1-np.std(z_val)

    if Fit:
        return  a*Monotonicity+b*Prognosability+c*Trendability
    else:
        return [Prognosability,Monotonicity,Trendability]