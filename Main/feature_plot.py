# PLOT ALL FEATURES VS TIME
# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")
PLOTS_DIR = os.path.join(BASE_DIR, "Feature_Time_Plots")

# Make sure output directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Feature names (same as your entropy script)
feature_names = [
    'A', 'R', 'E', 'CSS', 'CCNT',
    'RMS', 'D', 'CHIT', 'SS', 'CNTS'
]

max_points = 100_000

def downsample_data(time, values, max_points=max_points):
    """Downsample data for faster plotting"""
    if len(time) > max_points:
        indices = np.linspace(0, len(time)-1, max_points, dtype=int)
        return time[indices], values[indices]
    return time, values

# Get all parquet files
files_to_plot = [
    f for f in os.listdir(CLEAN_DIR)
    if f.endswith(".parquet") and not f.endswith(".csv")
]

for file_name in files_to_plot:
    print(f"\nProcessing: {file_name}")
    
    # Load data
    parquet_file = os.path.join(CLEAN_DIR, file_name)
    base_name = os.path.splitext(file_name)[0]
    df = pd.read_parquet(parquet_file)
    
    # Get time array
    time = df['t'].values
    print(f"  Original points: {len(time):,}")
    
    # Create figure with subplots
    n_features = len(feature_names)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    fig.suptitle(f"Features vs Time - {base_name}", fontsize=16, y=1.02)
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Plot each feature
    for idx, feature_name in enumerate(feature_names):
        if feature_name in df.columns:
            feature_values = df[feature_name].values
            
            # Remove NaN for cleaner plotting
            mask = ~np.isnan(feature_values)
            if np.any(mask):
                t_clean = time[mask]
                f_clean = feature_values[mask]
                
                # DOWNSAMPLE for plotting
                t_plot, f_plot = downsample_data(t_clean, f_clean, max_points=max_points)
                
                # Plot on the appropriate subplot
                ax = axes[idx]
                ax.plot(t_plot, f_plot, linewidth=0.5, alpha=0.7)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(feature_name)
                ax.set_title(f'{feature_name}')
                ax.grid(True, alpha=0.3)
                
                print(f"    {feature_name}: plotted {len(t_plot):,} points (downsampled from {len(t_clean):,})")
            else:
                ax = axes[idx]
                ax.text(0.5, 0.5, 'All NaN', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{feature_name}')
        else:
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Missing', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feature_name}')
    
    # Hide any unused subplots
    for idx in range(len(feature_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PLOTS_DIR, f"{base_name}_all_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")
    
    # Create normalized plot with downsampling
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    fig2.suptitle(f"Normalized Features vs Time - {base_name}", fontsize=14)
    
    plotted_count = 0
    for feature_name in feature_names:
        if feature_name in df.columns:
            feature_values = df[feature_name].values
            mask = ~np.isnan(feature_values)
            
            if np.any(mask):
                t_clean = time[mask]
                f_clean = feature_values[mask]
                min_val = np.min(f_clean)
                max_val = np.max(f_clean)
                
                if max_val > min_val:  # Avoid division by zero
                    normalized = (f_clean - min_val) / (max_val - min_val)
                    
                    # DOWNSAMPLE for normalized plot
                    t_plot, norm_plot = downsample_data(t_clean, normalized, max_points=max_points)
                    
                    ax2.plot(t_plot, norm_plot, label=feature_name, linewidth=0.8, alpha=0.7)
                    plotted_count += 1
    
    if plotted_count > 0:
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Normalized Value (0-1)')
        ax2.set_title(f'All Features Normalized - {base_name}')
        ax2.legend(loc='best', ncol=2, fontsize='small')
        ax2.grid(True, alpha=0.3)
        
        # Save normalized plot
        save_path_norm = os.path.join(PLOTS_DIR, f"{base_name}_all_features_normalized.png")
        plt.savefig(save_path_norm, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path_norm}")
    else:
        print(f"  No valid data for normalized plot")
    
    plt.close()

print("\nDone! All plots saved to:", PLOTS_DIR)