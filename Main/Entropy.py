# ENtropy approach for feature extraction

# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
PLOTS_DIR = os.path.join(BASE_DIR, "Entropy Plots")

#Test
#print("CLEAN_DIR:", CLEAN_DIR)
#print("Exists:", os.path.exists(CLEAN_DIR))
#print("Dirs inside:", os.listdir(CLEAN_DIR))

#MAKING SURE OUTPUT EXISTS ON UR LAPTOP
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


#TO PLOT OR NOT TO PLOT
DO_PLOTS = False
#===PARAMETERS AND STUFF =====
# A
# R
# THR
# E
# NoTRAI
# TRAI
# CSS
# CCNT
# RMS
# PCTA
# PCTD
# D
# CHIT
# SS
# 
#NEW CODE
#REMOVED PCTA AND TRAI AND THR AND PCTD AND NOTRAI
# removed D, E, R, SS, 

#add again
# ,'CSS','CCNT',
#     'RMS','CHIT'
# BEST ONES BELOW
    # 'A','CSS','CCNT',,'CHIT'
feature_names = [
'RMS','CHIT','A']
# features = df['A'].values  # replace with the column you want to extract entropy from

# Parameters
window_seconds = 125  # seconds per window
step_seconds = 125    # overlap step
bins = 15 # 'auto' #ASK IF VALId
strategy = 'uniform' #options r uniform quantile and kmeans
encode = 'ordinal' #I aint touching this shit
min_samples = 10 #ask if valid
base = 2 #TO GET BITS AS INSTRUCTED BY BENJAMIN

#START WITH DISCRETISATIONS
#THEN ENTIRE ENTROPY WHICH IS JUST FOR CHECKING
#FINALLY THE USEFUL ENTROPY OVER TIME

# === FUNCTION: Pre-compute discretized data for entire feature ===
def discretize_feature(feature, bins=bins, strategy=strategy, encode=encode):
    """
    Discretize entire feature once and return bin indices and discretizer
    """
    # Remove NaN
    feature_clean = feature[~np.isnan(feature)]
    n_bins = bins
    if np.unique(feature_clean).size < bins:
        n_bins = np.unique(feature_clean).size
    if len(feature_clean) == 0:
        return None, None, np.array([])  # No valid data
    
    # Reshape for KBinsDiscretizer
    X = feature_clean.reshape(-1, 1).astype(float)
    
    # Create and fit discretizer
    discretizer = KBinsDiscretizer(n_bins=bins, encode=encode, strategy=strategy)
    
    try:
        # Transform to bin indices
        binned = discretizer.fit_transform(X)
        
        if encode == 'ordinal':
            bin_indices = binned.flatten()
        else:
            bin_indices = np.argmax(binned, axis=1)
        
        # Return bin indices AND the fitted discretizer
        return discretizer, bin_indices, feature_clean
        
    except Exception as e:
        print(f"Warning: Discretization failed for feature: {e}")
        return None, None, np.array([])

# === FUNCTION: Compute entropy from bin indices ===
def entropy_from_bins(bin_indices, base=2):
    """
    Calculate Shannon entropy from pre-computed bin indices
    """
    if len(bin_indices) == 0:
        return np.nan
    
    unique, counts = np.unique(bin_indices, return_counts=True)
    p = counts / len(bin_indices)
    
    if base == 2:
        H = -np.sum(p * np.log2(p))
    else:
        H = -np.sum(p * np.log(p) / np.log(base))
    
    return H

# === FUNCTION: Windowed entropy from pre-computed bin indices ===
def windowed_entropy_from_bins(bin_indices, original_feature, time, 
                               window_seconds=window_seconds, step_seconds=step_seconds,
                               min_samples=min_samples, base=base):
    """
    Compute windowed entropy using pre-computed bin indices
    """
    entropies = []
    window_centers = []
    
    # We need to map bin_indices back to original time indices
    # Create a mask of non-NaN positions
    non_nan_mask = ~np.isnan(original_feature)
    non_nan_time = time[non_nan_mask]
    
    start = 0
    end_time = time[-1]
    
    while start + window_seconds <= end_time:
        # Find indices in the NON-NAN time array that fall within window
        start_idx = np.searchsorted(non_nan_time, start, side='left')
        end_idx = np.searchsorted(non_nan_time, start + window_seconds, side='right')
        
        # Get bin indices for this window
        window_bins = bin_indices[start_idx:end_idx]
        
        if len(window_bins) < min_samples:
            entropies.append(np.nan)
        else:
            H = entropy_from_bins(window_bins, base=base)
            entropies.append(H)
        
        window_centers.append(start + window_seconds / 2)
        start += step_seconds
    
    return np.array(entropies), np.array(window_centers)

# =========================
# CONFIG
# =========================
entropy_config = {
    "window_seconds": 125,
    "step_seconds": 125,
    "bins": 15,
    "strategy": "uniform",
    "encode": "ordinal",
    "min_samples": 10,
    "base": 2
}


# =========================
# ENTROPY PIPELINE WRAPPER
# =========================
def run_entropy_pipeline(file_path, feature_name, config):


    df = pd.read_parquet(file_path)

    time = df["t"].values
    feature = df[feature_name].values

    # unpack config
    window_seconds = config["window_seconds"]
    step_seconds = config["step_seconds"]
    bins = config["bins"]
    strategy = config["strategy"]
    encode = config["encode"]
    min_samples = config["min_samples"]
    base = config["base"]

    # -------------------------
    # 1. Discretize feature
    # -------------------------
    _, bin_indices, _ = discretize_feature(
        feature,
        bins=bins,
        strategy=strategy,
        encode=encode
    )

    if bin_indices is None or len(bin_indices) == 0:
        return np.array([]), np.array([])

    # -------------------------
    # 2. Windowed entropy
    # -------------------------
    entropy_values, window_centers = windowed_entropy_from_bins(
        bin_indices,
        feature,
        time,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_samples=min_samples,
        base=base
    )

    return entropy_values, window_centers


# =========================
# EXAMPLE USAGE
# =========================
# entropies, times = run_entropy_pipeline(
#     file_path="INSERT FILE NAME",
#     feature_name="RMS",
#     config=entropy_config
# )


#function to padd the outputs
def pad_entropy_list(entropies):
    max_len = max(len(arr) for arr in entropies)
    padded = np.zeros((len(entropies), max_len))

    for i, arr in enumerate(entropies):
        padded[i, :len(arr)] = arr

    return padded


files_to_plot = []
for root, dirs, files in os.walk(CLEAN_DIR):
    for file in files:
        if file.endswith(".parquet"):
            full_path = os.path.join(root, file)
            files_to_plot.append(full_path)

# print("Files found:", len(files_to_plot))

#TESTING
# entropies, times = run_entropy_pipeline(
#     file_path=files_to_plot[0],
#     feature_name="RMS",
#     config=entropy_config
# )
# print(entropies, np.shape(entropies))
# test = np.shape(entropies)
# print("Entropy length:", len(entropies))
# print("First values:", entropies[:10])
# print("Times shape:", np.shape(times))


for file_name in files_to_plot:
    #informing what it's currently plotting.
    # print(f"Plotting: {file_name}")
    #Input stuff
    relative_path = os.path.relpath(file_name, CLEAN_DIR)
    subfolder = relative_path.split(os.sep)[0]

    output_dir = os.path.join(PLOTS_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    parquet_file = file_name
    base_name = os.path.splitext(os.path.basename(file_name))[0]

    save_path = os.path.join(output_dir, f"{base_name}_entropy.png")
    save_path_full = os.path.join(output_dir, f"{base_name}_full_entropy.png")
    df = pd.read_parquet(parquet_file)

    # Get the time and feature column
    time = df['t'].values
    
    # DICTIONARY STORAGE
    entropy_results = {}  # for time series entropy
    full_results = {}     # for bar plot entropy
    window_centers = None  # will be set after first feature
    entropies_array = []
    for feature_name in feature_names:
        # print(f"  Processing: {feature_name}")
        
        #take values of dat shit
        feature = df[feature_name].values
        
        # STEP 1: Discretize THIS feature (move inside the loop!)
        discretizer, bin_indices, feature_clean = discretize_feature(
            feature, bins=bins, strategy=strategy, encode=encode
        )
        
        if discretizer is None:
            # print(f"     Skipping {feature_name} - discretization failed")
            full_results[feature_name] = np.nan
            entropy_results[feature_name] = np.array([np.nan])
            continue
        
        # STEP 2: Calculate bar plot entropy from bin indices
        H_bar = entropy_from_bins(bin_indices, base=base)  # Use this, not full_shannon_entropy
        full_results[feature_name] = H_bar
        
        # print(f"    Full entropy: {H_bar:.4f}")
        
        # STEP 3: Calculate windowed entropy
        entropy_values, window_centers = windowed_entropy_from_bins(
            bin_indices, feature, time, 
            window_seconds=window_seconds, 
            step_seconds=step_seconds,
            min_samples=min_samples, 
            base=base
        )
        
        # STEP 4: STORE the results!
        entropy_results[feature_name] = entropy_values
        #numpy my balls
        entropies_array.append(entropy_values)
        # print(f"    Windowed entropy: {len(entropy_values)} windows")

#padding ma shit:
# max_len = max(len(arr) for arr in entropies_array)

# padded = np.zeros((len(entropies_array), max_len))

# for i, arr in enumerate(entropies_array):
#     padded[i, :len(arr)] = arr

# entropies_array = padded

# print(padded)
# print(np.shape(padded))

    #=-=-==-=-=-==PMT TESTING SECTION-=-=-=-=-=-=-=-=
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import warnings
# from numba import jit,njit
# #Author: Arturo Rull Nomen
# warnings.filterwarnings('ignore')


# @njit
# def PMT_report(feature,window_time,coefficients, Fit=False,):
#     """
#     Returns an array [prognosability,Monotonocity,Trendability] of features if Fit=False, and returns
#     the Fit factor with defined coefficients if set to True.
    
#     The feature will be a (n_samples,number of feautures recorded per column) array
#     """
#     #print("Starting PMT analysis")
#     #print(coefficients)
#     a=coefficients[0]
#     b=coefficients[1]
#     c=coefficients[2]
#     #print(coefficients)
#     #Ensures the time arrray refernced will always be within range
#     length_data=np.zeros(feature.shape[0])
#     for i,run in enumerate(feature):
#         length_data[i]=len(run)
#     max_len=np.max(length_data)
#     time=np.arange(0,max_len*window_time,window_time)



#     ###Attain monotonicity of a single distribution of features and cycle through different samples, take then the average of all samples.
#     #print("Computing monotonicity...")
#     n_samples=np.shape(feature)[0]

#     Monotonicity_arr=np.zeros(n_samples)
#     #print(time)
    
#     for k in range(n_samples): #Loop through every sample
#         Num=0
#         Den=0
#         feature_k=feature[k,:]
#         time_k=time[feature_k!=0] #Sizes the time accordingly to the trimmed feature
#         feature_k=feature_k[feature_k!=0] #Takes zeros out

#         n_measurements=len(feature_k)
        

#         #Attains the time and measuremenents arrays into 1d for this part


#         for i in range(n_measurements):

#             #Do not forget j=0 always (see definition)
#             if i!=0:
#                 Num+=(time_k[0]-time_k[i])*np.sign(feature_k[0]-feature_k[i])

#             for j in range(i+1,n_measurements):
#                 Num+=(time_k[j]-time_k[i])*np.sign(feature_k[j]-feature_k[i])
#                 Den+=time_k[j]-time_k[i]
#         #Once the sums have been performed, will compute the value of monotonicity
#         Monotonicity_arr[k]=(Num/Den)


#     Monotonicity=np.sum(Monotonicity_arr)/n_samples #Computes the average of all values of Monotonicity

#     ###Prognosability by means computing the standard deviation of the final values of the feature and apply the definition of trensability

#     #print("Computing prognosability...")
#     initial_vals=np.zeros(n_samples)
#     final_vals=np.zeros(n_samples) 
#     for i in range(n_samples):
#         feature_i=feature[i,:]
#         feature_i=feature_i[feature_i!=0]
#         initial_vals[i]=feature_i[0]
#         final_vals[i]=feature_i[-1] #Attains a list of the final values


#     Prognosability= np.e**(-np.std(final_vals)/np.mean(np.abs(initial_vals-final_vals))) #Accounting for the definition from the lit.review

#     ###Attain trendability

#     #print("Computing trendability...")
#     z_val=np.zeros(n_samples)

#     for k in range(n_samples): #Loop through every sample 

#         ###Attain an individual 1D array with the required length for the given sample 
#         feature_k=feature[k,:]
#         time_k=time[feature_k!= 0]
#         feature_k=feature_k[feature_k!= 0]
        

#         n_measurements=len(feature_k)


#         #Compute first and deconde derivatives of the feature
#         dydt=(feature_k[1:]-feature_k[:-1])/(time_k[1:]-time_k[:-1]) #first order approxiamtion of the first derivative of the dataset... rough approximations
#         dy2dt2=(feature_k[2:]-2*feature_k[1:-1]+feature_k[:-2])/((time_k[2:]-time_k[1:-1]+time_k[:-2]-time_k[1:-1])/2)**2 #Second order approxiamtion of the second derivative of the feature (see https://en.wikipedia.org/wiki/Finite_difference)
        
#         #Count values where greater than zero

#         n1d=len(np.where(dydt>0)[0])
#         n2d=len(np.where(dy2dt2>0)[0])
#         D=n_measurements
#         val=n1d/(D-1)+n2d/(D-2) #Definition taken from (DOI: 10.4233/uuid:538558fb-ac9a-414d-8a59-4b523d8ff74c)
#         z_val[k]=val

#     #Once the values of Z have been defined for every sample, the std of this variable is computed and trendability is attained
#     Trendability=1-np.std(z_val)


#     return  a*Monotonicity+b*Prognosability+c*Trendability
    
# print('Full values =', type(full_results))
# print('entropy values =', type(entropy_values), type(entropy_values[0]))
# coefficients = np.array([1,2,3])
# # PMT_report(full_results,window_time=window_seconds,coefficients=coefficients, Fit=False,)
# a = PMT_report(entropies_array,window_time=window_seconds,coefficients = coefficients, Fit=True,) 
# print(a)  
# =-=-==-=-=-== PLOTTING SECTION (DISABLED BY DEFAULT) -=-=-=-=-=-=-=-=

if DO_PLOTS:

    # ================= TIME SERIES ENTROPY PLOT =================
    if window_centers is not None:

        plt.figure(figsize=(12, 8))
        plotted_any = False

        for feature_name, entropy_values in entropy_results.items():

            if not np.all(np.isnan(entropy_values)):
                plt.plot(
                    window_centers,
                    entropy_values,
                    label=feature_name,
                    linewidth=1
                )
                plotted_any = True

        if plotted_any:
            plt.xlabel("Time (s)")
            plt.ylabel("Shannon Entropy (bits)")
            plt.title(f"Shannon Entropy Features - {base_name}")
            plt.legend(loc='best', ncol=2, fontsize='small')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Time series entropy plot saved to: {save_path}")

        else:
            print(f"Warning: No valid time series data to plot for {base_name}")
            plt.close()

    else:
        print(f"Warning: No window centers generated for {base_name}")


    # ================= FULL ENTROPY BAR PLOT =================

    plt.figure(figsize=(12, 8))

    valid_features = []
    valid_entropies = []

    for feature_name, entropy_value in full_results.items():
        if not np.isnan(entropy_value):
            valid_features.append(feature_name)
            valid_entropies.append(entropy_value)

    if len(valid_features) > 0:

        plt.bar(valid_features, valid_entropies)
        plt.ylabel("Shannon Entropy (bits)")
        plt.title(f"Shannon Entropy for each feature - {base_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Full entropy bar plot saved to: {save_path_full}")

    else:
        print("Warning: All full entropy results are NaN")
        plt.close()
    # #ATTEMPT TO CLEAN THE SHAESH by not plotting when nan exists maybe it cleans it?
    # valid_results = {i: j for i, j in entropy_results.items() if not np.isnan(j)}
    # if valid_results:
    #     plt.figure(figsize=(12, 8))
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Shannon Entropy (bits)")
    #     plt.title("Shannon Entropy for each feature over time")
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight') #CHANGE DPI IF ITS UGLY
    #     plt.close()
    #     print(f"Done generating plots for {base_name}")
    # else:
    #     print("Warning: All full entropy results are NaN")
    # # OPTIONAL: plot the entropy feature over time
    # # plt.figure(figsize=(12,8))
    # # for feature_name, entropy_values in entropy_results.items():
    # #     plt.plot(window_times, entropy_values, label=feature_name)

    # # # plt.xlabel("Time (s)")
    # # plt.ylabel("Shannon Entropy (bits)")
    # # plt.title("Shannon Entropy Features")
    # # plt.legend()

    # # Save figure
    # save_path = os.path.join(PLOTS_DIR, f"{base_name}_entropy.png")

    # # plt.savefig(save_path, dpi=300, bbox_inches='tight') #CHANGE DPI IF ITS UGLY
    # # plt.close()
    # # print(f"Done generating plots for {base_name}")

    # #ATTEMPT TO CLEAN THE SHAESH by not plotting when nan exists maybe it cleans it?
    # valid_full_results = {i: j for i, j in full_results.items() if not np.isnan(j)}
    # if valid_results:
    #     plt.figure(figsize=(12, 8))
    #     plt.bar(valid_full_results.keys(), valid_full_results.values())
    #     plt.ylabel("Shannon Entropy (bits)")
    #     plt.title("Shannon Entropy for each feature")
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #     plt.savefig(save_path_full, dpi=300, bbox_inches='tight') #CHANGE DPI IF ITS UGLY
    #     plt.close()
    #     print(f"Done generating plots for {base_name}")
    # else:
    #     print("Warning: All full entropy results are NaN")
    # #NEW PLOTS WITHOUT TIME DEPENDENCY
    # plt.figure(figsize=(12, 8))
    # plt.bar(full_results.keys(), full_results.values())
    # plt.ylabel("Shannon Entropy (bits)")
    # plt.title("Shannon Entropy for each feature")
    # plt.xticks(rotation = 45)
    # plt.tight_layout()
    # plt.savefig(save_path_full, dpi=300)
    # plt.close()    
# # Create plots folder if it doesn't exist
# plots_dir = os.path.join(BASE_DIR, "Entropy Plots")
# os.makedirs(plots_dir, exist_ok=True)


    # print(f"Plot saved to: {save_path}")
    # print(f"plot saved to: {save_path_full}")

    # === FUNCTION: Compute entropy from bin indices ===
        #USES THE DISCRETISER OUTPUT
        # def entropy_from_bins(bin_indices, base=2):

        #     if len(bin_indices) == 0:
        #         return np.nan #SHOULDNT NEED TO BE USED CUZ I CLEANED IT BUT JUST IN CASE
            
        #     unique, counts = np.unique(bin_indices, return_counts=True)
        #     p = counts / len(bin_indices)
        #     #ONLY BASE 2 GONNA BE USED TBH 
        #     if base == 2:
        #         H = -np.sum(p * np.log2(p))
        #     else:
        #         H = -np.sum(p * np.log(p) / np.log(base))
            
        #     return H
        # # FUNCTION: compute Shannon entropy per time window
        
        # def shannon_entropy_feature(feature, time, 
        #                             window_seconds=window_seconds,
        #                             step_seconds=step_seconds,
        #                             min_samples = min_samples,
        #                             bins=bins , #WORK IN PROGRESS
        #                             strategy = strategy, #options r uniform quantile and kmeans
        #                             encode = encode,
        #                             base=2):
        #     #to store entropy
        #     entropies = []
        #     #storing center for historgrams
        #     window_centers = []

        #     start = 0
        #     end_time = time[-1]

        #     while start + window_seconds <= end_time:
        #         # Get indices within current window OLD CODE
        #         # idx = (time >= start) & (time < start + window_seconds)
        #         # window_data = feature[idx]
        #         #NEW CODE IMPLEMENTING NP.SEARCHSORTED
        #         start_idx = np.searchsorted(time, start, side='left')
        #         end_idx = np.searchsorted(time, start + window_seconds, side='right')
        #         window_data = feature[start_idx:end_idx]
        #         #REMOVE NAN
        #         window_data = window_data[~np.isnan(window_data)]
        #         #ACCOUNT FOR NAN
        #         if len(window_data) < min_samples:
        #             entropies.append(np.nan)
        #         else:
        #             #USING SKLEARN DISCRETISEeee
        #             X = window_data.reshape(-1, 1).astype(float)
                
        #             # Apply KBinsDiscretizer
        #             discretizer = KBinsDiscretizer(n_bins=bins, encode=encode, strategy=strategy)
                
        #             try:
        #                 # Transform data to bin indices
        #                 binned = discretizer.fit_transform(X)
                        
        #                 if encode == 'ordinal':
        #                     # For ordinal encoding, binned contains bin indices directly
        #                     bin_indices = binned.flatten()
        #                 else:
        #                     # For one-hot encoding, we need to convert to bin indices
        #                     bin_indices = np.argmax(binned, axis=1)
                        
        #                 # Calculate probability distribution from bin counts
        #                 unique, counts = np.unique(bin_indices, return_counts=True)
        #                 p = counts / len(bin_indices)
                        
        #                 # Shannon entropy
        #                 if base == 2:
        #                     H = -np.sum(p * np.log2(p))
        #                 # elif base == np.e:
        #                 #     H = -np.sum(p * np.log(p))  WE PROBS NEVER GONNA USE BUT JUST IN CASE LOL
        #                 # else:
        #                 #     H = -np.sum(p * np.log(p) / np.log(base))
                        
        #                 entropies.append(H)
                    
        #             except Exception as e:
        #                 # Fallback in case discretization fails
        #                 print(f"Warning: Discretization failed for window at {start}: {e}")
        #                 entropies.append(np.nan)
            
        
        #         # Window center for plotting
        #         window_centers.append(start + window_seconds / 2)

        #         # Move to next window
        #         start += step_seconds

        #     return np.array(entropies), np.array(window_centers)
        


        # #COMPUTE ENTROPY FOR EVERY FEATURE WITHOUT TIME WINDOW
        # def full_shannon_entropy(feature, bins=bins, base=base, strategy = strategy, encode = encode):
        #     feature_clean = feature[~np.isnan(feature)]

        #     if len(feature_clean) ==0: #IF ALL IS NAN THEN RETURN ALL IS NAN
        #         return np.nan
        #     #reshape for kbins
        #     X = feature_clean.reshape(-1, 1).astype(float)
        #     #Discretise
        #     discretiser = KBinsDiscretizer(n_bins = bins, encode = encode, strategy=strategy)

        #     try:
        #         #transofirming the data to bindicies
        #         binned = discretiser.fit_transform(X)

        #         if encode == 'ordinal':
        #             bin_indices = binned.flatten()
        #         else:
        #             bin_indices = np.argmax(binned, axis = 1)

        #         unique, counts = np.unique(bin_indices, return_counts=True)
        #         p = counts / len(bin_indices)

        #         #shannon entropy
        #         if base == 2:
        #             H = -np.sum(p * np.log2(p))
        #         #else:
        #             #H = -np.sum(p * np.log(p) / np.log(base))
        #     # hist, _ = np.histogram(feature, bins=bins)
        #     # p = hist/ hist.sum()
        #     # p = p[p>0]
        #         return H

        #     #EXCEPTION
        #     except Exception as e:
        #         print(f"Warning: discretisation failed for full feature: {e}")
        #         return np.nan
        # Compute the entropy feature for 1 feature OLD CODE
        # entropy_values, window_times = shannon_entropy_feature(features, time,
        #                                                        window_seconds=window_seconds,
        #                                                        step_seconds=step_seconds,
        #                                                        bins=bins,
        #                                                        base=base)