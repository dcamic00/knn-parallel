import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Gráfica de Speedup ---
df_scale = pd.read_csv('scaling_results.csv')
df_scale['Threads'] = df_scale['Threads'].astype(str).str.extract(r'(\d+)').astype(int)

plt.figure(figsize=(9, 5))
plt.plot(df_scale['Threads'], df_scale['Speedup'], marker='o', linestyle='-', linewidth=2.5, markersize=9, color='#1f77b4')
plt.title('Strong Scaling: Speedup vs Threads', fontsize=14, pad=15)
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Speedup (over 1-thread SIMD)', fontsize=12)
plt.xticks([1, 2, 4, 8, 16])
plt.grid(True, linestyle='--', alpha=0.7)
for i, row in df_scale.iterrows():
    plt.annotate(f"{row['Speedup']:.2f}x", (row['Threads'], row['Speedup']), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('speedup_plot.png', dpi=300)
plt.close()

# --- 2. Gráfica de Precisión (Accuracy) ---
k_values = ['K=1', 'K=5', 'K=10']
euclidean_acc = [96.91, 96.88, 96.65]
cosine_acc = [97.23, 97.30, 97.08]

x = np.arange(len(k_values))
width = 0.35

plt.figure(figsize=(9, 5))
fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x - width/2, euclidean_acc, width, label='Euclidean', color='#ff7f0e')
rects2 = ax.bar(x + width/2, cosine_acc, width, label='Cosine', color='#2ca02c')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Classification Accuracy: Euclidean vs Cosine Distance', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(k_values, fontsize=12)
ax.set_ylim(96.0, 97.5)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)
plt.close()

# --- 3. Gráfica de Scheduling ---
df_sched = pd.read_csv('scheduling_results.csv')
plt.figure(figsize=(8, 5))
bars = plt.bar(df_sched['Schedule_Type'], df_sched['Time_Seconds'], color=['#1f77b4', '#ff7f0e', '#d62728'], width=0.6)
plt.title('OpenMP Scheduling Performance (8 Threads)', fontsize=14, pad=15)
plt.xlabel('Schedule Strategy', fontsize=12)
plt.ylabel('Execution Time (Seconds)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 3, f"{yval:.2f}s", ha='center', va='top', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('scheduling_plot.png', dpi=300)
plt.close()

print("¡Generadas con éxito: speedup_plot.png, accuracy_plot.png y scheduling_plot.png!")