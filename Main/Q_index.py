import numpy as np
import glob
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 


# Load data
'''
main_file = pd.read_parquet(glob.glob("Main/**/*.parquet"))



feature_names = ['A','R','E','CSS','CCNT','CNTS','SS','D','RMS','CHIT']

cols_to_drop = ['t','THR','TRAI','NoTRAI','PTCA','PTCD']
FINAL_main_file = main_file.drop(columns=cols_to_drop, errors='ignore')

# transform to numpy  
#dfnumpy = FINAL_main_file.to_numpy(dtype='float')
# 
# 
# fatigue = main_file['t'].to_numpy()

'''

def compute_q_index(X, window_size=None):

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    fatigue = np.arange(1, len(X)+1)

    # Reference window
    ref_size = int(len(X)*0.1)
    X_ref = X[:ref_size, :]

    # normalize the reference data 
    scaler = StandardScaler()
    # X_ref_scaled = scaler.fit_transform(X_ref)
    X_ref_scaled = X_ref

    # normalize the full data 
    # X_full_scaled = scaler.transform(dfnumpy)
    X_full_scaled = X

    # perform the PCA
    pca = PCA(n_components=0.90)
    pca.fit(X_ref_scaled)

    # Formulate the residual projection matrix
    P_r = pca.components_.T 

    # transform full data set to PC space 
    T = X_full_scaled @ P_r

    # compute the reconstruction matrix 
    X_r = T @ P_r.T 

    # Compute the Q-index
    residuals = X_full_scaled - X_r
    q = np.sum(np.square(residuals), axis=1)

    # convert the index to dataframe
    q_series = pd.DataFrame(q)
    q_exp = q_series.expanding().mean().to_numpy()

    q_final = q_exp[::window_size]
    fatigue_plot = fatigue[::window_size]


    return q_final, fatigue_plot



'''
dfnumpy = FINAL_main_file.to_numpy(dtype='float')
q_final, fatigue_plot = compute_q_index(dfnumpy, window_size=2)

# Plot Q-index vs time
plt.figure(figsize=(10,9)) 
plt.plot(fatigue_plot, q_final,color='blue', linewidth='2' )
plt.xlabel('time')
plt.ylabel('Q-index')
plt.show()

print(q_final)


main_file = pd.read_parquet(glob.glob("Main/**/*.parquet"))



feature_names = ['A','R','E','CSS','CCNT','CNTS','SS','D','RMS','CHIT']

cols_to_drop = ['t','THR','TRAI','NoTRAI','PTCA','PTCD']
FINAL_main_file = main_file.drop(columns=cols_to_drop, errors='ignore')

# transform to numpy  
dfnumpy = FINAL_main_file.to_numpy(dtype='float')
fatigue = main_file['t'].to_numpy()

X = np.asarray(dfnumpy, dtype=float)

ref_size = int(len(dfnumpy)*0.1)
X_ref = dfnumpy[:ref_size, :]

# normalize the reference data 
scaler = StandardScaler()
# X_ref_scaled = scaler.fit_transform(X_ref)
X_ref_scaled = X_ref

# normalize the full data 
# X_full_scaled = scaler.transform(df_numpy)
X_full_scaled = dfnumpy 

# perform the PCA
pca = PCA(n_components=0.90)
pca.fit(X_ref_scaled)

# Formulate the residual projection matrix
P_r = pca.components_.T 

# transform full data set to PC space 
T = X_full_scaled @ P_r

# compute the reconstruction matrix 
X_r = T @ P_r.T 

# Compute the Q-index
residuals = X_full_scaled - X_r
q = np.sum(np.square(residuals), axis=1)

# convert the index to dataframe
q_series = pd.DataFrame(q)

# implement the cumulative moving average 
q_final = q_series.expanding().mean().to_numpy()
   
'''



