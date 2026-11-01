import pandas as pd
import matplotlib.pyplot as plt

# 1. Leer el archivo CSV que generó tu código C++
df = pd.read_csv('scaling_results.csv')

# Limpiar la columna de hilos (convierte "1(Seq)" a 1 numérico)
df['Threads'] = df['Threads'].astype(str).str.extract(r'(\d+)').astype(int)

# 2. Configurar la gráfica
plt.figure(figsize=(10, 6))
plt.plot(df['Threads'], df['Speedup'], marker='o', linestyle='-', linewidth=2.5, markersize=9, color='#1f77b4')

# 3. Darle formato y etiquetas
plt.title('Aceleración (Speedup) vs Número de Hilos\nClasificador k-NN Exacto en MNIST', fontsize=14, pad=15)
plt.xlabel('Número de Hilos (OpenMP)', fontsize=12)
plt.ylabel('Speedup (respecto a secuencial puro)', fontsize=12)
plt.xticks([1, 2, 4, 8, 16])
plt.grid(True, linestyle='--', alpha=0.7)

# Añadir el número exacto encima de cada punto
for i, row in df.iterrows():
    plt.annotate(f"{row['Speedup']:.2f}x", 
                 (row['Threads'], row['Speedup']),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 fontsize=10,
                 fontweight='bold')

# 4. Guardar la imagen en tu carpeta
plt.tight_layout()
plt.savefig('speedup_plot.png', dpi=300, bbox_inches='tight')
print("¡Éxito! Gráfica generada y guardada como 'speedup_plot.png' en tu carpeta.")