import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#DIRECTORIES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "clean_data")

#STARTING PROMPT
print("⚠️  WARNING: This will OVERWRITE your original files!")
confirm = input("Enter AUTO_TRIM to run: ")

if confirm != "AUTO_TRIM":
    print("Exiting...")
    exit()

parquet_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.parquet')]
print(f"Found {len(parquet_files)} parquet files")

# Fixed window removal function
def window_removal(df, file_name, time_col='t'):
    """Manually remove a time window from the data"""
    print(f'\n✂️  Manual window removal for {file_name}:')
    print(f"Current time range: {df[time_col].min():.1f}s - {df[time_col].max():.1f}s")
    
    # Show the data to help decide
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[time_col], df['CCNT'], 'b-', alpha=0.5, linewidth=0.8, label='CCNT')
    if 'D' in df.columns:
        ax.plot(df[time_col], df['D'], 'r-', alpha=0.5, linewidth=0.8, label='D')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title(f'{file_name} - Select window to remove')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    
    # Get user input
    try:
        begin = float(input("Enter the start time to cut (in seconds): "))
        end = float(input("Enter the end time to cut (in seconds): "))
        
        # Validate input
        if begin >= end:
            print("❌ Error: Start time must be less than end time")
            return df, None
        
        if begin < df[time_col].min() or end > df[time_col].max():
            print(f"⚠️  Warning: Cut window extends beyond data range")
            print(f"   Data range: {df[time_col].min():.1f}s - {df[time_col].max():.1f}s")
            confirm_cut = input("Continue anyway? (y/n): ").lower()
            if confirm_cut != 'y':
                return df, None
        
        # Create mask for data outside the cut window
        mask = (df[time_col] < begin) | (df[time_col] > end)
        df_clean = df[mask].copy()
        
        # Calculate time shift
        gap = end - begin
        
        # Adjust times after the cut window
        df_clean.loc[df_clean[time_col] > end, time_col] -= gap
        
        # Reset time to start at 0 if needed
        if len(df_clean) > 0:
            df_clean[time_col] = df_clean[time_col] - df_clean[time_col].iloc[0]
        
        removed_rows = len(df) - len(df_clean)
        removed_pct = (removed_rows / len(df)) * 100 if len(df) > 0 else 0
        
        print(f"✅ Removed {removed_rows:,} rows ({removed_pct:.1f}%)")
        print(f"   New time range: {df_clean[time_col].min():.1f}s - {df_clean[time_col].max():.1f}s")
        
        # Show before/after comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Before
        axes[0].plot(df[time_col], df['CCNT'], 'b-', alpha=0.5, linewidth=0.8)
        axes[0].axvspan(begin, end, alpha=0.3, color='red', label='Removed window')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('CCNT')
        axes[0].set_title(f'BEFORE - {file_name}\nRemoved: {begin:.1f}s - {end:.1f}s')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # After
        axes[1].plot(df_clean[time_col], df_clean['CCNT'], 'b-', alpha=0.5, linewidth=0.8)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('CCNT')
        axes[1].set_title(f'AFTER - Manual Cut\n({len(df_clean):,} rows, {df_clean[time_col].max():.1f}s)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df_clean, (begin, end)
        
    except ValueError:
        print("❌ Error: Please enter valid numbers")
        return df, None

# Define feature extraction directly in this file
def feature_extraction(feature, time, window_seconds=125, threshold=16):
    """Compute features per time window"""
    window_centers = []
    start = 0
    end_time = time[-1]

    sums = []
    averages = []
    standard_deviations = []
    skewnesses = []
    maxima = []
    kurtosi = []
    low_fractions = []
    while start + window_seconds <= end_time:
        idx = (time >= start) & (time < start + window_seconds)
        window_data = feature[idx]
        window_data = np.array(window_data)
        
        n = len(window_data)
        sum_window_data = np.sum(window_data)
        sums.append(sum_window_data)
        low_fraction = np.mean(window_data <= threshold)
        low_fractions.append(low_fraction)
        if n > 1:
            average = sum_window_data / n
            averages.append(average)

            std_dev_help = window_data - average
            std_dev = np.sqrt(1 / (n - 1) * np.sum(std_dev_help**2))
            standard_deviations.append(std_dev)

            max_val = np.max(window_data)
            if len(maxima) == 0 or max_val > maxima[-1]:
                maxima.append(max_val)
            else:
                maxima.append(np.nan)

            if std_dev == 0:
                skewnesses.append(0)
                kurtosi.append(0)
            else:
                skewness = 1/n * np.sum((window_data - average)**3) / std_dev**3
                skewnesses.append(skewness)
                kurtosis = 1/n * np.sum((window_data - average)**4) / std_dev**4 - 3
                kurtosi.append(kurtosis)
        else:
            averages.append(np.nan)
            standard_deviations.append(np.nan)
            skewnesses.append(np.nan)
            maxima.append(np.nan)
            kurtosi.append(np.nan)

        window_centers.append(start + window_seconds / 2)
        start += window_seconds

    return sums, window_centers, averages, standard_deviations, skewnesses, maxima, kurtosi, low_fractions

# To determine thresholds
def analyze_ccnt_distribution(df, file_name):
    """Analyze CCNT distribution to help choose threshold"""
    ccnt = df['CCNT'].values
    
    print(f"\n📊 CCNT Analysis for {file_name}:")
    print(f"  Min: {ccnt.min():.4f}")
    print(f"  Max: {ccnt.max():.4f}")
    print(f"  Mean: {ccnt.mean():.4f}")
    print(f"  Median: {np.median(ccnt):.4f}")
    print(f"  Std: {ccnt.std():.4f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(ccnt, p)
        print(f"    {p}%: {val:.4f}")
    
    # Count of zeros
    zeros = np.sum(ccnt == 0)
    print(f"\n  Zero values: {zeros:,} ({zeros/len(ccnt)*100:.1f}%)")
    
    # Suggest thresholds
    p95 = np.percentile(ccnt, 95)
    p90 = np.percentile(ccnt, 90)
    p75 = np.percentile(ccnt, 75)
    
    print(f"\n  Suggested thresholds:")
    print(f"    Conservative (high activity only): {p95:.4f}")
    print(f"    Moderate: {p90:.4f}")
    print(f"    Sensitive (low activity): {p75:.4f}")
    
    return p90

# End of experiment function
def detect_end_of_experiment(df, time_col='t', count_col='CCNT', window_seconds=125, threshold=0.1):
    """Detect where the experiment ends by analyzing counts per window."""
    
    # Use feature extraction
    sums, window_centers, averages, std_devs, skews, maxs, kurtosi, low_fractions = feature_extraction(
        df[count_col].values, 
        df[time_col].values, 
        window_seconds
    )
    
    if len(averages) == 0:
        print("  No windows created - data too short?")
        return None
    
    # Find where activity drops below threshold
    active_windows = [avg > threshold for avg in averages]
    
    if not any(active_windows):
        print(f"  ⚠️  No active windows found! All below threshold {threshold}")
        return None
    
    # Find the last active window
    last_active_idx = max([i for i, active in enumerate(active_windows) if active])
    last_active_time = window_centers[last_active_idx]
    
    # Check subsequent windows for consistent inactivity
    inactive_count = 0
    for i in range(last_active_idx + 1, len(active_windows)):
        if not active_windows[i]:
            inactive_count += 1
        else:
            # Found another active window - reset
            inactive_count = 0
            last_active_idx = i
            last_active_time = window_centers[i]
    
    # If we have at least 3 consecutive inactive windows after last activity
    if inactive_count >= 3:
        cut_time = last_active_time + window_seconds
        print(f"  Detected end at ~{cut_time:.1f}s (last activity at {last_active_time:.1f}s)")
        return cut_time
    
    return None

# Trim the end
def trim_experiment_end(df, time_col='t', count_col='CCNT', window_seconds=125, threshold=0.1):
    """Trim the end of the experiment where counts are below threshold."""
    cut_time = detect_end_of_experiment(df, time_col, count_col, window_seconds, threshold)
    
    if cut_time is None:
        print("  No trimming needed - experiment appears active throughout")
        return df, None  # Always return 2 values: df and None
    
    # Apply trim
    df_trimmed = df[df[time_col] <= cut_time].copy()
    
    # Reset time to start at 0
    if len(df_trimmed) > 0:
        df_trimmed[time_col] = df_trimmed[time_col] - df_trimmed[time_col].iloc[0]
    
    removed = len(df) - len(df_trimmed)
    removed_pct = (removed / len(df)) * 100 if len(df) > 0 else 0
    
    print(f"  Trimmed from {len(df):,} to {len(df_trimmed):,} rows")
    print(f"  Removed {removed:,} rows ({removed_pct:.1f}%)")
    
    return df_trimmed, cut_time  # Always return 2 values

#NEW ATTEMPT FOR GAP REMOVAL
def is_inactive_window(window_data, threshold, frac=0.9):
    return np.mean(window_data <= threshold) > frac

#Gap removal
def remove_middle_gaps(df, time_col='t', count_col='CCNT', window_seconds=125, threshold=0.1, min_gap_windows=3):  

    # Use feature extraction
    sums, window_centers, averages, std_devs, skews, maxs, kurtosi, low_fractions = feature_extraction(
        df[count_col].values, 
        df[time_col].values, 
        window_seconds
    )
    
    # Calculate window averages
    if len(averages) == 0:
        print("  No windows created - data too short?")
        return df, []  # Always return 2 values
    
    # Identify inactive windows
    inactive_windows = [lf > 0.9 for lf in low_fractions]
    
    # Find segments of consecutive inactive windows
    gaps = []
    in_gap = False
    gap_start_idx = None
    
    for i, is_inactive in enumerate(inactive_windows):
        if is_inactive and not in_gap:
            # Start of a potential gap
            in_gap = True
            gap_start_idx = i
        elif not is_inactive and in_gap:
            # End of gap
            in_gap = False
            gap_length = i - gap_start_idx
            
            if gap_length >= min_gap_windows:
                # Calculate actual time boundaries
                gap_start_time = window_centers[gap_start_idx] - window_seconds/2
                gap_end_time = window_centers[i-1] + window_seconds/2
                
                gaps.append((gap_start_time, gap_end_time))
                print(f"  Found gap: {gap_start_time:.1f}s - {gap_end_time:.1f}s ({gap_length} windows, {gap_end_time-gap_start_time:.1f}s)")
    
    # Check if there's a gap at the end
    if in_gap and (len(inactive_windows) - gap_start_idx) >= min_gap_windows:
        gap_start_time = window_centers[gap_start_idx] - window_seconds/2
        gap_end_time = df[time_col].max()
        gaps.append((gap_start_time, gap_end_time))
        print(f"  Found trailing gap: {gap_start_time:.1f}s - {gap_end_time:.1f}s")
    
    # Remove gaps if any found
    if gaps:
        # Create mask to keep data not in any gap
        mask = np.ones(len(df), dtype=bool)
        for gap_start, gap_end in gaps:
            mask &= ~((df[time_col] >= gap_start) & (df[time_col] <= gap_end))
        
        df_clean = df[mask].copy()
        
        # Reset time to start at 0
        if len(df_clean) > 0:
            df_clean[time_col] = df_clean[time_col] - df_clean[time_col].iloc[0]
        
        removed_rows = len(df) - len(df_clean)
        print(f"  Removed {removed_rows:,} rows across {len(gaps)} gaps")
        return df_clean, gaps  # Always return 2 values
    else:
        print("  No significant gaps found")
        return df, []  # Always return 2 values

#Trimming with gap removal
def trim_ae_data_comprehensive(df, time_col='t', count_col='CCNT', window_seconds=125, 
                                activity_threshold=0.1, min_gap_windows=3):
    """
    Trim both ends and gaps based on count activity.
    
    This function:
    1. Removes gaps in the middle
    2. Trims the end where activity stops
    
    Returns:
    --------
    df_final : DataFrame with gaps and end trimmed
    cut_time : float, time where end was cut (None if no end cut)
    gaps : list of tuples, gaps that were removed
    """
    original_rows = len(df)
    
    # Step 1: Remove middle gaps
    print("\n  Detecting and removing middle gaps...")
    df_no_gaps, gaps = remove_middle_gaps(df, time_col, count_col, window_seconds, 
                                          activity_threshold, min_gap_windows)
    
    if gaps:
        print(f"  Removed {len(gaps)} gaps")
    
    # Step 2: Trim the end
    print("\n  Trimming the end...")
    df_final, cut_time = trim_experiment_end(df_no_gaps, time_col, count_col, 
                                             window_seconds, activity_threshold)
    
    total_removed = original_rows - len(df_final)
    
    if total_removed > 0:
        print(f"\n  Total: Removed {total_removed:,} rows ({(total_removed/original_rows)*100:.1f}%)")
    else:
        print(f"\n  Total: No rows removed")
    
    # Always return 3 values
    return df_final, cut_time, gaps


# Show the trimming with before/after comparison
def show_trimming_comparison(df_before, df_after, file_name, cut_time=None, activity_threshold=None):
    """Show side-by-side comparison before and after trimming."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Before trimming
    axes[0].plot(df_before['t'], df_before['CCNT'], 'b-', alpha=0.5, linewidth=0.8, label='CCNT')
    if 'D' in df_before.columns:
        axes[0].plot(df_before['t'], df_before['D'], 'r-', alpha=0.5, linewidth=0.8, label='D')
    if cut_time:
        axes[0].axvline(x=cut_time, color='g', linestyle='--', linewidth=2, label=f'Cut at {cut_time:.0f}s')
    
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'BEFORE - {file_name}\n({len(df_before):,} rows, {df_before["t"].max():.1f}s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: After trimming
    axes[1].plot(df_after['t'], df_after['CCNT'], 'b-', alpha=0.5, linewidth=0.8, label='CCNT')
    if 'D' in df_after.columns:
        axes[1].plot(df_after['t'], df_after['D'], 'r-', alpha=0.5, linewidth=0.8, label='D')
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Value')
    axes[1].set_title(f'AFTER - Trimmed\n({len(df_after):,} rows, {df_after["t"].max():.1f}s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add summary text
    removed_rows = len(df_before) - len(df_after)
    removed_percent = (removed_rows / len(df_before)) * 100 if len(df_before) > 0 else 0
    
    summary_text = f'Trim Summary: Removed {removed_rows:,} rows ({removed_percent:.1f}%)'
    if activity_threshold:
        summary_text += f'\nThreshold: {activity_threshold}'
    if cut_time:
        summary_text += f'\nCut time: {cut_time:.1f}s'
    
    fig.suptitle(summary_text, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

#visualisation for gap trimming 
def show_trimming_comparison_with_gaps(df_before, df_after, file_name, gaps=None, cut_time=None, activity_threshold=None):
    """Show side-by-side comparison with gap indicators."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Before trimming with gaps highlighted
    axes[0].plot(df_before['t'], df_before['CCNT'], 'b-', alpha=0.5, linewidth=0.8, label='CCNT')
    if 'D' in df_before.columns:
        axes[0].plot(df_before['t'], df_before['D'], 'r-', alpha=0.5, linewidth=0.8, label='D')
    
    # Highlight gaps in red
    if gaps:
        for gap_start, gap_end in gaps:
            axes[0].axvspan(gap_start, gap_end, alpha=0.3, color='red', label='Gap' if gap_start == gaps[0][0] else '')
    
    if cut_time:
        axes[0].axvline(x=cut_time, color='g', linestyle='--', linewidth=2, label=f'Cut at {cut_time:.0f}s')
    
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'BEFORE - {file_name}\n({len(df_before):,} rows, {df_before["t"].max():.1f}s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: After trimming
    axes[1].plot(df_after['t'], df_after['CCNT'], 'b-', alpha=0.5, linewidth=0.8, label='CCNT')
    if 'D' in df_after.columns:
        axes[1].plot(df_after['t'], df_after['D'], 'r-', alpha=0.5, linewidth=0.8, label='D')
    
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Value')
    axes[1].set_title(f'AFTER - Trimmed\n({len(df_after):,} rows, {df_after["t"].max():.1f}s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add summary text
    removed_rows = len(df_before) - len(df_after)
    removed_percent = (removed_rows / len(df_before)) * 100 if len(df_before) > 0 else 0
    
    summary_text = f'Trim Summary: Removed {removed_rows:,} rows ({removed_percent:.1f}%)'
    if activity_threshold:
        summary_text += f'\nThreshold: {activity_threshold}'
    if gaps:
        summary_text += f'\nGaps removed: {len(gaps)}'
    if cut_time:
        summary_text += f'\nEnd cut at: {cut_time:.1f}s'
    
    fig.suptitle(summary_text, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ========== SETUP PARAMETERS BEFORE MAIN LOOP ==========
print("\n" + "="*60)
print("📝 Set trimming parameters (these will apply to all files):")
print("="*60)

window_seconds = float(input("Window size in seconds (default=125): ") or "125")

# Analyze first file to suggest threshold
first_file_path = os.path.join(CLEAN_DIR, parquet_files[0])
first_df = pd.read_parquet(first_file_path)
suggested_threshold = analyze_ccnt_distribution(first_df, parquet_files[0])

print(f"\nSuggested threshold based on data: {suggested_threshold:.4f}")

# Let user choose threshold method
print("\nHow would you like to set the threshold?")
print("  1. Use suggested threshold (based on 90th percentile)")
print("  2. Enter custom value")

choice = input("Choice (1-2, default=1): ") or "1"

if choice == "1":
    activity_threshold = suggested_threshold
    print(f"Using suggested threshold: {activity_threshold:.4f}")
else:  # choice == "2"
    activity_threshold = float(input(f"Enter threshold (suggested: {suggested_threshold:.4f}): "))

remove_gaps = input("Remove middle gaps? (y/n, default=n): ").lower() == 'y'

print(f"\n✅ Using: window={window_seconds}s, threshold={activity_threshold}, remove_gaps={remove_gaps}")
print("="*60)

# ========== MAIN PROCESSING LOOP WITH GAP REMOVAL AND MANUAL CUT OPTION ==========
for i, file_name in enumerate(parquet_files, 1):
    print(f"\n{'='*60}")
    print(f"File {i}/{len(parquet_files)}: {file_name}")
    print(f"{'='*60}")
    
    try:
        # Load
        file_path = os.path.join(CLEAN_DIR, file_name)
        df = pd.read_parquet(file_path)
        
        print(f"Original: {len(df):,} rows, {df['t'].min():.1f}s - {df['t'].max():.1f}s")
        
        # Initialize variables for this file
        current_threshold = activity_threshold
        file_processed = False
        retry = True
        
        # Allow retry loop for this file
        while retry:
            # Ask if user wants to customize threshold for this file
            if i > 1 or retry is not True:  # Show customize option after first file or on retry
                print("\nOptions for this file:")
                print("  1. Use default threshold")
                print("  2. Customize threshold")
                print("  3. ✂️  Manual window cut")
                print("  4. Skip this file entirely")
                print("  5. Exit program")
                
                file_option = input("Choice (1-5, default=1): ") or "1"
                
                if file_option == "4":
                    print(f"⏭️  Skipping {file_name}")
                    file_processed = True
                    break
                elif file_option == "5":
                    print("Exiting program...")
                    exit()
                elif file_option == "3":
                    # Manual window cut
                    df_manual_cut, cut_window = window_removal(df, file_name)
                    if cut_window:
                        # Ask if they want to save the manually cut version
                        print("\nWhat would you like to do with the manually cut version?")
                        print("  1. ✅ Save this version")
                        print("  2. ❌ Discard and continue with original")
                        print("  3. 🔄 Try another manual cut")
                        
                        manual_action = input("Choice (1-3, default=1): ") or "1"
                        
                        if manual_action == "1":
                            df_manual_cut.to_parquet(file_path, compression='snappy')
                            print(f"✅ Saved manually cut version! {len(df_manual_cut):,} rows")
                            file_processed = True
                            break
                        elif manual_action == "3":
                            # Continue loop to try another manual cut
                            continue
                        else:
                            # Discard and continue with original
                            pass
                    else:
                        print("Manual cut cancelled or failed, continuing...")
                    continue
                elif file_option == "2":
                    analyze_ccnt_distribution(df, file_name)
                    current_threshold = float(input(f"Enter threshold for this file (default={activity_threshold}): ") or str(activity_threshold))
                    print(f"Using custom threshold: {current_threshold}")
                # else option 1 - use default, continue
            
            # MAIN TRIMMING SECTION
            if remove_gaps:
                print("\n  Performing comprehensive trim (gaps + end)...")
                df_trimmed, cut_time, gaps = trim_ae_data_comprehensive(df, 't', 'CCNT', window_seconds, current_threshold, min_gap_windows=3)
            else:
                df_trimmed, cut_time = trim_experiment_end(df, 't', 'CCNT', window_seconds, current_threshold)
                gaps = []
            
            # Handle case where no end detected but gaps might have been removed
            if cut_time is None and not gaps:
                print("  ⚠️  No end detected and no gaps found - file appears active throughout")
                
                # Show current state
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                ax.plot(df['t'], df['CCNT'], 'b-', alpha=0.5, linewidth=0.8)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('CCNT')
                ax.set_title(f'{file_name} - No trimming needed')
                ax.grid(True, alpha=0.3)
                plt.show()
                
                print("\nWhat would you like to do?")
                print("  1. Save file as-is")
                print("  2. Skip file (keep original)")
                print("  3. Try different threshold")
                print("  4. ✂️  Try manual cut")   
                print("  5. Exit program")
                
                no_cut_option = input("Choice (1-5, default=2): ") or "2"
                
                if no_cut_option == "1":
                    df.to_parquet(file_path, compression='snappy')
                    print(f"✅ Saved (unchanged)")
                    file_processed = True
                    break
                elif no_cut_option == "3":
                    print("Retrying with different threshold...")
                    continue
                elif no_cut_option == "4":
                    # Manual cut option
                    df_manual_cut, cut_window = window_removal(df, file_name)
                    if cut_window:
                        df_manual_cut.to_parquet(file_path, compression='snappy')
                        print(f"✅ Saved manually cut version! {len(df_manual_cut):,} rows")
                        file_processed = True
                        break
                    else:
                        print("Manual cut cancelled, continuing...")
                        continue
                elif no_cut_option == '5':
                    print("Exiting program...")
                    exit()
                else:
                    print(f"⏭️  Skipping {file_name}")
                    file_processed = True
                    break

            # Handle case where gaps were removed but no end cut
            elif cut_time is None and gaps:
                print("  ✅ Gaps were removed from the middle, but no end trimming needed.")
                # Show comparison with gaps
                print("\nShowing comparison plot...")
                show_trimming_comparison_with_gaps(df, df_trimmed, file_name, gaps, cut_time, current_threshold)
                
                print("\nWhat would you like to do?")
                print("  1. ✅ Save this trimmed version (gaps removed)")
                print("  2. ❌ Skip file (keep original)")
                print("  3. 🔄 Try different threshold")
                print("  4. ✂️  Try manual cut instead")
                print("  5. 🚪 Exit program")
                
                action = input("Choice (1-5, default=1): ") or "1"
                
                if action == "1":
                    df_trimmed.to_parquet(file_path, compression='snappy')
                    print(f"✅ Saved! {len(df_trimmed):,} rows")
                    file_processed = True
                    break
                elif action == "3":
                    print("Retrying with different threshold...")
                    continue
                elif action == "4":
                    # Manual cut option
                    df_manual_cut, cut_window = window_removal(df, file_name)
                    if cut_window:
                        df_manual_cut.to_parquet(file_path, compression='snappy')
                        print(f"✅ Saved manually cut version! {len(df_manual_cut):,} rows")
                        file_processed = True
                        break
                    else:
                        print("Manual cut cancelled, continuing...")
                        continue
                elif action == "5":
                    print("Exiting program...")
                    exit()
                else:
                    print(f"⏭️  Skipped - original file preserved")
                    file_processed = True
                    break
            
            # Normal case: end detected (with or without gaps)
            else:
                # Show comparison
                print("\nShowing comparison plot...")
                if remove_gaps and gaps:
                    show_trimming_comparison_with_gaps(df, df_trimmed, file_name, gaps, cut_time, current_threshold)
                else:
                    show_trimming_comparison(df, df_trimmed, file_name, cut_time, current_threshold)
                
                # Ask what to do with the result
                print("\nWhat would you like to do?")
                print("  1. ✅ Save this trimmed version")
                print("  2. ❌ Skip this file (keep original)")
                print("  3. 🔄 Try different threshold")
                print("  4. ✂️  Try manual cut instead")
                print("  5. 📊 Show more details (window averages)")
                print("  6. 🚪 Exit program")
                
                action = input("Choice (1-6, default=1): ") or "1"
                
                if action == "1":
                    df_trimmed.to_parquet(file_path, compression='snappy')
                    print(f"✅ Saved! {len(df_trimmed):,} rows")
                    file_processed = True
                    break
                elif action == "2":
                    print(f"⏭️  Skipped - original file preserved")
                    file_processed = True
                    break
                elif action == "3":
                    print("\n🔄 Retrying with different threshold...")
                    # Show analysis to help choose new threshold
                    analyze_ccnt_distribution(df, file_name)
                    current_threshold = float(input(f"Enter new threshold (previous was {current_threshold}): "))
                    continue  # Go back to trimming with new threshold
                elif action == "4":
                    # Manual cut option
                    df_manual_cut, cut_window = window_removal(df, file_name)
                    if cut_window:
                        df_manual_cut.to_parquet(file_path, compression='snappy')
                        print(f"✅ Saved manually cut version! {len(df_manual_cut):,} rows")
                        file_processed = True
                        break
                    else:
                        print("Manual cut cancelled, continuing...")
                        continue
                elif action == "5":
                    # Show more detailed window analysis
                    sums, window_centers, averages, std_devs, skews, maxs, kurtosi = feature_extraction(
                        df['CCNT'].values, 
                        df['t'].values, 
                        window_seconds
                    )
                    
                    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                    
                    # Window averages
                    axes[0].plot(window_centers, averages, 'b-', linewidth=1.5)
                    axes[0].axhline(y=current_threshold, color='r', linestyle='--', label=f'Threshold={current_threshold}')
                    axes[0].fill_between(window_centers, 0, averages, where=np.array(averages) > current_threshold,
                                         color='green', alpha=0.3, label='Active')
                    axes[0].set_ylabel('Average CCNT')
                    axes[0].set_title(f'Window Averages ({window_seconds}s windows)')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Standard deviations
                    axes[1].plot(window_centers, std_devs, 'g-', linewidth=1.5)
                    axes[1].set_xlabel('Time (s)')
                    axes[1].set_ylabel('Standard Deviation')
                    axes[1].set_title('Window Standard Deviations')
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Continue to next iteration to ask again
                    continue
                elif action == "6":
                    print("Exiting program...")
                    exit()
        
        # If file was processed, move to next
        if file_processed:
            # Option to stop after this file
            if i < len(parquet_files):
                print("\n" + "-"*40)
                continue_choice = input("Continue to next file? (y/n, default=y): ").lower()
                if continue_choice != 'y' and continue_choice != '':
                    print("Stopping early...")
                    break
            continue
                
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Ask what to do on error
        print("\nWhat would you like to do?")
        print("  1. Skip this file and continue")
        print("  2. Exit program")
        error_choice = input("Choice (1-2, default=1): ") or "1"
        
        if error_choice == "2":
            print("Exiting...")
            break
        else:
            continue

print("\n" + "="*60)
print("All files processed!")
print("="*60)