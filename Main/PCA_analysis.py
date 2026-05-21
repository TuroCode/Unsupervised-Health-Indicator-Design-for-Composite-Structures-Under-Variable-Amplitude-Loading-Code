import glob
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------
#OPEN AND EDIT FILE

#main_file will be the only and main file we perform PCA on
main_file = pd.read_parquet(glob.glob("Main/**/*.parquet"))
 

feature_names = [
        'A','R','E','CSS','CCNT', 'CNTS', 'SS','D',
        'RMS','CHIT'
    ]
#select columns coming after 't'
cols_to_drop = ["t", "THR", "TRAI", "NoTRAI", "PCTA", "PCTD"] 
FINAL_main_file = main_file.drop(columns=cols_to_drop, errors='ignore')

#transform to numpy array
df_numpy = FINAL_main_file.to_numpy(dtype='float32')


timeline = main_file["t"].values
D_data = main_file["D"].values

#--------------------------------------------------------
#PERFORM PCA

#data scaling
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_numpy)


pca = PCA(n_components=10)
Final_PCA = pca.fit_transform(scaled_array)

# Extract the explained variance ratio
variance_ratios = pca.explained_variance_ratio_

# Compute cumulative explained variance
var_ratio_cumulative = np.cumsum(variance_ratios)

PCs = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10",]

#PLOT PC values
plt.figure(figsize=(10,9))
plt.plot(PCs, var_ratio_cumulative * 100, marker='o', linestyle='-', color='blue') 
for i, (pc, val) in enumerate(zip(PCs, var_ratio_cumulative)):
    plt.text(pc, val * 100, f'{val * 100:.1f}%', ha='center', va='bottom')  # Label each point with its percentage
plt.xlabel('Principal Component', fontsize=14)
plt.ylabel('Cumulative sum of explained variance after each PC (%)', fontsize=14)
plt.show()



#----------------------------------------------------------------
#PLOTTING 

fig, (sub1, sub2) = plt.subplots(1, 2, figsize = (15,6))


# Get the loadings (components)
loadings = pca.components_  # Shape: (n_components, n_features)

# Create a DataFrame for easier interpretation
loadings_df = pd.DataFrame(
    loadings,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)


# Show only the top 6 PCs that explain 95% variance
top_6_loadings = loadings_df.iloc[:6]  # PC1-PC6 only
print(top_6_loadings)

# Sum the absolute contributions across PC1-PC6
feature_importance_in_95pct = top_6_loadings.abs().sum(axis=0).sort_values(ascending=False)
print("\nFeature importance (cumulative across PC1-PC6):")
print(feature_importance_in_95pct)






def plot1(sub1):
    ##
    # define loadings

    ##
    sub1.scatter(loadings[:, 0], loadings[:, 1])

    # Add labels to each point
    for i, label in enumerate(feature_names):
        sub1.text(loadings[i, 0], loadings[i, 1], label)

    sub1.axhline(0, color='black')  # y=0 axis
    sub1.axvline(0, color='black')  # x=0 axis

    sub1.set_xlabel("Weight on Principal Component 1 (Damage Progression)")
    sub1.set_ylabel("Weight on Principal Component 2")
    sub1.set_title("PCA result")
    plt.show()
    return None
