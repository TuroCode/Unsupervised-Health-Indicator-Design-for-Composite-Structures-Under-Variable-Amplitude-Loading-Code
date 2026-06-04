
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import random
from Feature_Extraction import feature_extraction
from Data_Standarization import Filter_Scale_File_Def #Note that the data used will be standarised!
from numba import jit,njit
from Entropy import run_entropy_pipeline,entropy_config
from sklearn.preprocessing import StandardScaler

#(Main) Responsible: Arturo Rull Nomen

"""
This file includes the ML algorythm proposed based on gradient ascent.
Please note that a convention for Fitness of 0.6*Monotonicity+0.4*Prognosability+0.00001*Trendability
"""




###----------------------------------SETUP OF OVERARCHING PARAMETERS--------------------------------


plotting_results=True #Shows the reslting HI derives for al samples fed onto the programme


######################################
#Feature Selection
######################################



Column = np.array(["CSS","CCNT","R","D","RMS"])
Features=np.array(["sums","maxima","maxima","maxima","sums"])
readlocally=False #Set to true if already have the matrix loaded locally (decreases computational time)

######################################
#Machine learning algorithm parameters
######################################


fit_coeff=np.array([0.599999,0.4,0.00000001]) # a*Monotonicity+b*Prognosability+c*Trendability, definition of the fitness used


###Learning hyperparameters set up

window_seconds=125
lr_ref=.01 #Learning rate reference value, check out definition for more info
adaptative_lr=False #If set to true, adaptative learning rate will be considered, this will selct larger learning rates for features with a low order of magnitude values, allowing for a better optimisation requireing less iterations


back_prop_step=0.01 #Step taken for approximating the partial derivative of the HI versus different coefficients
coeffs=np.array([0.1,0.1,0.1,0.1,0.1]) #Coefficients for initialisation

#Miscellanious
fitness=0 #Used for startup of the gradient descent (recommend to leave @ zero)
iter_max=1000





###----------------------------------------------------PMT EVAL DEFS---------------------------------------


#Imported MPT report function, to evaluate loss function for backpropagation
@njit
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
    #Ensures tAhe time arrray refernced will always be within range
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


    return  a*Monotonicity+b*Prognosability+c*Trendability

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



###---------------------------------------ML PART: DEFINITIONS AND ITERATIONS-----------------------------



@njit
def HI_fun(matrixes,coeff):
    val=np.zeros((matrixes[0].shape[0],matrixes[0].shape[1]))

    "Definiton of the HI, number of features and defintion are to be changed"
    for i in range(coeff.shape[0]):
        val=val+matrixes[i]*coeff[i]
    
    return val


###In order to use numba, will define a function for iteration, initial coeficients and data array and returns the new coefficients
@njit
def HIEVAL(matrixes,coeff,HI_fun,fit_coeff):
    "Evaluates the current performance of the HI, based on the coefficients and definition given"
    #Compute the matrix of HI values based on the definition given thus returns a SINGLE matrix of size (n_files x n_windows)
    HI=HI_fun(matrixes,coeff) ##Returns a sinlge matrix
    fitness=PMT_report(HI,window_seconds,fit_coeff,True) #Returns the fit factor from the def HI
    return fitness

@njit
def BACKPROP(EVAL_fun,coeff,step,matrixes,fitness,lr_array,HI_fun,fit_coeff):
    """Loops through the coefficients and perfroms gradient ascent, takes the evaluation function, current coefficients, iteration step"""

    grad_array=np.zeros(len(Features)) #Define an array that computes the gradient in each direction of all variables' coefficients

    #Iterate through the different coefficients, will then find a vectoc that point in the direction of the gradient, then will step it that direction right at the end of the loop
    for i in range(len(coeff)):
        
        coeff_search=coeff.copy()
        coeff_search[i]=coeff[i]+step #Coeff search is solely used for backpropagattion purposes
        
        fitness_bpg=EVAL_fun(matrixes,coeff_search,HI_fun,fit_coeff)
        grad_array[i]=(fitness_bpg-fitness)/step #Gradient with respect to that coefficient at the original point 

    return coeff+grad_array*lr_array #Update all and returns the result




"Actual begining of computations, will first attain matrices of the same shape for each feature (they should be the same size for every different feature, as the window time was constant)"


Reference="Reference.parquet"

if not readlocally:

    #Setup for feature
    Scaling_dict={}
    df_ref=pd.read_parquet(Reference)

    #Will not be using all of the columns for feature extraction, as sutch we will not use for the dictionary def
    #df_ref= df_ref["CSS","CCNT","R","D","RMS"]

    time=df_ref["t"].values #Returns array from the t table, required for feature evaluation
    
    featurename=["sums", "window_times", "averages", "standard_deviations", "skewnesses", "maxima","kurtosi","entropy"]


    #Define scaling values to each column: Creates a dictionary that relates feature to mean, std
    
    for i,column in enumerate(Column):
        
        column_chosen=Column[i] #Just the name of the column
        column_dat=df_ref[column_chosen] #Column dat actually 
        
        #Compute mean and std of every single feature extractable for THAT column 

        sums, window_times, averages, standard_deviations, skewnesses, maxima, kurtosi = feature_extraction(column_dat, time, window_seconds=window_seconds)
        entropy=run_entropy_pipeline(Reference,column_chosen,config=entropy_config)[0]
        allfeatures=np.array([sums, window_times, averages, standard_deviations, skewnesses, maxima, kurtosi,entropy])
        
    
        for j,feat in enumerate(allfeatures):
    
            min_feat=np.min(feat)
            max_feat=np.max(feat)
            name=featurename[j] + Column[i]
            Scaling_dict.update({name:(min_feat,max_feat)})

    


    matrixes=[]

    ###Features from data are extracted below
    for i in Features:
        matrixes.append([])#Will store n_features arrays of (n_samples,n_measurements)

    print("______________Reading data______________")

    for file in Path(os.path.join(BASE_DIR, "clean_data/Training")).iterdir():

        print(f"reading {file}")
        df = pd.read_parquet(file)
        time = df['t'].values #Returns array from the t table, required for feature evaluation
        
        for i in range(len(Features)):
            
            #As loop across the features, will index the column in question and scale it based on the reference file given (part of the testing set)
            column_chosen = Column[i] #For analysis of the feature developed

            features_dat=df[column_chosen].to_numpy()

            sums, window_times, averages, standard_deviations, skewnesses, maxima, kurtosi = feature_extraction(features_dat, time, window_seconds=window_seconds)
            
            #Appends the corresponding feature to the corresponsing spot
            name=Features[i]+Column[i]
            min_scale=Scaling_dict[name][0]
            max_scale=Scaling_dict[name][1]
            range_scale=max_scale-min_scale


            if Features[i]=="sums": (matrixes[i].append(list((np.array(sums)-min_scale)/range_scale)))
            elif Features[i]=="window_times":matrixes[i].append(list((np.array(window_times)-min_scale)/range_scale))
            elif Features[i]=="averages":matrixes[i].append(list((np.array(averages)-min_scale)/range_scale))
            elif Features[i]=="standard_deviations":matrixes[i].append(list((np.array(standard_deviations)-min_scale)/range_scale))
            elif Features[i]=="skewnesses":matrixes[i].append(list((np.array(skewnesses)-min_scale)/range_scale))
            elif Features[i]=="maxima":matrixes[i].append(list((np.array(maxima-min_scale)/range_scale)))
            elif Features[i]=="kurtosi":matrixes[i].append(list((np.array(kurtosi)-min_scale)/range_scale))
            elif Features[i]=="entropy":
                entropy = run_entropy_pipeline(file,column_chosen,config=entropy_config)[0] #Computes entropy array for that file then and stores it in the respective matrix
                matrixes[i].append(list((entropy-min_scale)/range_scale)) #Append in the end

            ###Sums will be an array of (n_files x n_windows in that file) note that the matrix is not square!

    # Pad all rows to the same length to handle files with different durations(todo)
    matrixes_padded=[]
    
    for i,feature in enumerate(matrixes):
        max_len = max(len(row) for row in feature)
        matrix_padded = np.array([row + [0] * (max_len - len(row)) for row in feature])
        matrixes_padded.append(matrix_padded) #Saves this matrix to the overall one

    matrixes=np.array(matrixes_padded) #Now we have set of padded matrices
    #print(matrixes)
    np.save("Feature_matrix",matrixes)

else:
    matrixes=np.load("Feature_matrix.npy") #Read localy if already have, to avoid long computational times




#======================Determination of the lr coefficients following constant delta of mean of the features========

if adaptative_lr:
    mean_value=np.zeros(len(matrixes))
    for i,feature in enumerate(matrixes):
        mean_value[i]=abs(np.mean(feature)) #Takes the average values of the feature for all samples... considerable, so the formula derived for adaptative learning rates ,may be used for serveral samples

    #print(mean_value)
    max_mean=np.max(mean_value)
    lr_array=lr_ref*max_mean/mean_value #returns an array for the learning rate for each of the feeatures (i.e., a b and c), avoid scaling of the features which is quite nice...
    #print(lr_array)
else:
    lr_array=np.ones(len(Features))*lr_ref



#==================================Start of the ML loop===============================


print("\n______________Start of ML loop______________\n")

fitness_arr=np.zeros([iter_max])
coeff_hist=[]
coeff_search=[]
Fit_max=np.sum(fit_coeff)

j=0


while j<iter_max:


    j=j+1
    

    if j%25==0:
        coeff_hist.append([coeffs,j]) #Stores data for that iteration and number of iteration
        print(f"_______iteration__{j}____________\n Fitness = {fitness}, Coeffs= {coeffs}")

    fitness=HIEVAL(matrixes,coeffs,HI_fun,fit_coeff)
    fitness_arr[j-1]=fitness

    ###Backpropagation

    #Perfrom gradient ascent on the fit index as fitness function
    coeffs=BACKPROP(HIEVAL,coeffs,back_prop_step,matrixes,fitness,lr_array,HI_fun,fit_coeff)

#Note that for this methow will converge to local minima, not ensuring convergence to absolute minima



print("______________Optimisation overview______________")

print(f'Initial fitness: {fitness_arr[0]}\n Final fitness: {fitness_arr[-1]}')
print(f'Array of coefficients for the final values used: {coeffs}')

#Plotting
if plotting_results:

    plt.subplot(211)
    HI=HI_fun(matrixes,coeffs)

    for row in HI:
        row_drop_zeros=row[row!=0]
        plt.plot(range(len(row_drop_zeros)),row_drop_zeros)

    plt.xlabel("Window")
    plt.ylabel("HI values")
    plt.subplot(212)
    plt.plot(range(len(fitness_arr)),fitness_arr)
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.show()
