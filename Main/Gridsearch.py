import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from Data_Standarization import Filter_Scale_File_Def
from Feature_Extraction import feature_extraction
from Entropy import run_entropy_pipeline
from numba import njit

###Learning hyperparameters set up
fit_coeff=np.array([0.599999,0.4,0.00000001]) # a*Monotonicity+b*Prognosability+c*Trendability, definition of the fitness used
window_seconds=125
lr_ref=.1 #Learning rate reference value, check out definition for more info
adaptative_lr=False #If set to true, adaptative learning rate will be considered, this will selct larger learning rates for features with a low order of magnitude values, allowing for a better optimisation requireing less iterations


back_prop_step=.1 #Step taken for approximating the partial derivative of the HI versus different coefficients
init_coeffs=np.array([0.1,0.1,0.1,0.1,0.1]) #Coefficients for initialisation

#Miscellanious
fitness=0 #Used for startup of the gradient descent (recommend to leave @ zero)
iter_max=100


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

    return  a*Monotonicity+b*Prognosability+c*Trendability
    
entropy_config = {
    "window_seconds": 125,
    "step_seconds": 125,
    "bins": 15,
    "strategy": "uniform",
    "encode": "ordinal",
    "min_samples": 10,
    "base": 2
}


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# min_fitness_threshold = 0.4

# featuresPMT = []

# data_folder = Path("Main/clean_data/Training") 
# data_columns = ["R", "CSS", "CCNT", "RMS", "D"]


# for column_chosen in data_columns:
#     print('Column', column_chosen)

#     # Collect features across all files first
#     all_averages, all_stds, all_maxima, all_entropy = [], [], [], []
    
#     for file in data_folder.iterdir():
#         print('File', file.name)
#         df = pd.read_parquet(file)
#         time = df['t'].values
#         column = Filter_Scale_File_Def(df, [column_chosen])[0][column_chosen].values
#         sums, window_centers, averages, standard_deviations, skewnesses, maxima, kurtosi = feature_extraction(column, time, window_seconds=125)
#         entropy_values = run_entropy_pipeline(file, column_chosen, entropy_config)[0]

#         all_averages.append(averages)
#         all_stds.append(standard_deviations)
#         all_maxima.append(maxima)
#         all_entropy.append(entropy_values)

#     def pad_to_2d(arrays):
#         max_len = max(len(a) for a in arrays)
#         return np.array([np.pad(a, (0, max_len - len(a))) for a in arrays])

#     features_list = [
#         pad_to_2d(all_averages), 
#         pad_to_2d(all_stds), 
#         pad_to_2d(all_maxima), 
#         pad_to_2d(all_entropy)
#     ]

#     for i, feature_2d in enumerate(features_list):
#         fitness = PMT_report(feature_2d, 125, np.array([0.6, 0.4, 0.00000000000001]))

#         featuresPMT.append({
#             "column": column_chosen,
#             "feature": i,
#             "fitness": fitness
#         })

# max_entry = max(featuresPMT, key=lambda x: x["fitness"])
# print("Max entry:", max_entry)

# # print(featuresPMT)

# high_fitness = [x for x in featuresPMT if x["fitness"] > min_fitness_threshold]
# for entry in high_fitness:
#     print(entry)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model(Column, Features):
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




    if not readlocally:


        Scaling_dict={}
        df_ref=pd.read_parquet("Reference.parquet")

        #Define scaling values to each column: Dictionary that relates to each column used for features the mean and standard deviation for the respective colum <
        for i,column in enumerate(Column):
        
            column_chosen=Column[i]
            #Compute statistical values for the respective column
            std=np.std(df_ref[column_chosen])
            mean=np.mean(df_ref[column_chosen])

            Scaling_dict.update({column_chosen:(mean,std)})


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
                mean_scaling=Scaling_dict[column_chosen][0]
                std_scaling=Scaling_dict[column_chosen][1]
            
                features_dat=((df[column_chosen]-mean_scaling)/std_scaling)
                features_dat=features_dat.to_numpy()

                sums, window_times, averages, standard_deviations, skewnesses, maxima, kurtosi = feature_extraction(features_dat, time, window_seconds=window_seconds)
                
                #Appends the corresponding feature to the corresponsing spot

                if Features[i]=="sums":matrixes[i].append(sums)
                elif Features[i]=="window_times":matrixes[i].append(window_times)
                elif Features[i]=="averages":matrixes[i].append(averages)
                elif Features[i]=="standard_deviations":matrixes[i].append(standard_deviations)
                elif Features[i]=="skewnesses":matrixes[i].append(skewnesses)
                elif Features[i]=="maxima":matrixes[i].append(maxima)
                elif Features[i]=="kurtosi":matrixes[i].append(kurtosi)
                elif Features[i]=="entropy":
                    entropy = run_entropy_pipeline(file,column_chosen,config=entropy_config)[0] #Computes entropy array for that file then and stores it in the respective matrix
                    matrixes[i].append(entropy.tolist()) #Append in the end

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
        full_matrixes=np.load("Feature_matrix.npy") #Read localy if already have, to avoid long computational times
        # print(np.shape(full_matrixes))
        matrixes=full_matrixes[Features]




    #======================Determination of the lr coefficients following constant delta of mean of the features========

    if adaptative_lr:
        mean_value=np.zeros(len(matrixes))
        for i,feature in enumerate(matrixes):
            mean_value[i]=np.abs(np.mean(feature)) #Takes the average values of the feature for all samples... considerable, so the formula derived for adaptative learning rates ,may be used for serveral samples

        #print(mean_value)
        max_mean=np.max(mean_value)
        lr_array=lr_ref*max_mean/mean_value #returns an array for the learning rate for each of the feeatures (i.e., a b and c), avoid scaling of the features which is quite nice...
        # print(lr_array)
    else:
        lr_array=np.ones(len(Features))*lr_ref



    #==================================Start of the ML loop===============================


    # print("\n______________Start of ML loop______________\n")

    fitness_arr=np.zeros([iter_max])
    coeff_hist=[]
    coeff_search=[]
    coeffs = init_coeffs
    Fit_max=np.sum(fit_coeff)

    j=0


    while j<iter_max:


        j=j+1
        

        fitness=HIEVAL(matrixes,coeffs,HI_fun,fit_coeff)
        fitness_arr[j-1]=fitness

        ###Backpropagation

        #Perfrom gradient ascent on the fit index as fitness function
        coeffs=BACKPROP(HIEVAL,coeffs,back_prop_step,matrixes,fitness,lr_array,HI_fun,fit_coeff)
    
    return coeffs, fitness, fitness_arr, lr_array




readlocally=True #Set to true if already have the matrix loaded locally (decreases computational time)
features = np.array(["sums", "sums", "sums","sums", "sums", "standard_deviations","standard_deviations","standard_deviations","standard_deviations","standard_deviations","maxima","maxima","maxima","maxima","maxima", "entropy", "entropy", "entropy", "entropy", "entropy"]) #missing q-index and maxima
columns = np.array(["CSS","CCNT","R","D","RMS","CSS","CCNT","R","D","RMS","CSS","CCNT","R","D","RMS", "CSS","CCNT","R","D","RMS"])
models = [] #only saves the number of the model and ijkos (location of model)
fitness_plot = []
lr_plot = []

counter = 0
feature_number = 4
for i in range(feature_number):
    for j in range(feature_number):
        for k in range(feature_number):
            for o in range(feature_number):
                for s in range(feature_number):
                    counter += 1 
                    iter_features = np.array([0+5*i, 1+5*j, 2+5*k, 3+5*o, 4+5*s])
                    if readlocally == True:
                        coeffs, fitness, fitness_arr, lr_array =  train_model(columns, iter_features)
                        fitness_plot = fitness_arr
                        lr_plot = lr_array
                    else: 
                        coeffs, fitness =  train_model(columns, features) # gives error after generating matrix file, set readlocally to true after this errors out
                    models.append([i, j, k, o, s, fitness]) 
                    print(counter, fitness, iter_features)



models = np.array(models)


best_idx = np.argmax(models[:, 5])   # index of best model (saves the row in which the maximum value of PM appears)
best_model = models[best_idx]         # full row [i, j, k, PM_value]

print("Best PM value:", best_model[5])
print("Best i,j,k:", best_model[:5])

# #extract best model location
# i = best_model[0]
# j = best_model[1]
# k = best_model[2]
# o = best_model[3]
# s = best_model[4]
# best_features = np.array([int(best_model[0])*5+0, int(best_model[1])*5+1, int(best_model[2])*5+2, int(best_model[3])*5+3, int(best_model[4])*5+4])
# coeff, fitness = train_model(columns, best_features)

# plt.plot(range(len(lr_plot)),lr_plot)
# plt.xlabel("iteration")
# plt.ylabel("fitness")
# plt.show()