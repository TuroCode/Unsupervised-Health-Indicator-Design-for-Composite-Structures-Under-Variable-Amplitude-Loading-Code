import matplotlib.pyplot as plt
import numpy as np

# Data
features = ['A', 'R', 'THR', 'E', 'NoTRAI', 'TRAI', 'CSS', 'CCNT', 'RMS', 'PCTA', 'PCTD', 'D', 'CHIT', 'SS', 'CNTS']

prognosability = [0.3003, 0.3366, 0.0000, 0.1898,    None, 0.2512, 0.4869, 0.4232,  0.9089, 0.3205,    None, 0.6378,   None, 0.4869, 0.4232]
monotonicity   = [0.1413, 0.1619, 0.9997, 0.1563, 0.9997, 1.0000, 0.2112, 0.2141, -0.1892, 1.0000, 0.9997, 0.1820, 0.2472, 0.1423, 0.1452]
trendability   = [0.9929, 0.9926, 1.0000, 0.9899, 1.0000, 0.9478, 0.9674, 0.9717,  0.9573, 0.9599, 1.0000, 0.9916, 0.8475, 0.9880, 0.9765]
fitness        = [0.2049, 0.2318, 0.5998, 0.1697,    None, 0.7005, 0.3215, 0.2978,  0.2501, 0.7282,    None, 0.3643,   None, 0.2802, 0.2564]

# Replace None with NaN for plotting
prognosability = [v if v is not None else np.nan for v in prognosability]
monotonicity   = [v if v is not None else np.nan for v in monotonicity]
trendability   = [v if v is not None else np.nan for v in trendability]
fitness        = [v if v is not None else np.nan for v in fitness]

x = np.arange(len(features))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 6))

bars1 = ax.bar(x - 1.5*width, prognosability, width, label='Prognosability', color='darkorange')
bars2 = ax.bar(x - 0.5*width, monotonicity,   width, label='Monotonicity',   color='steelblue')
bars3 = ax.bar(x + 0.5*width, trendability,   width, label='Trendability',   color='seagreen')
bars4 = ax.bar(x + 1.5*width, fitness,        width, label='Fitness',        color='grey', edgecolor='black', linewidth=0.8)

ax.set_xlabel('Column Name', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
# ax.set_title('Baseline Fitness Scores for Raw Data Features', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=10)
ax.set_ylim(-0.3, 1.15)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('baseline_fitness_scores.png', dpi=300)
plt.show()