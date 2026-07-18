import matplotlib.pyplot as plt
import numpy as np

labels = ['FP32 Model', 'INT8 Model']
sizes = [25.4,6.8]
latencies = [45.2,38.5]

x = np.arange(len(labels))
width = 0.35

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_ylabel('Size (MB)', color=color)
rects1 = ax1.bar(x - width/2, sizes, width, label='Size (MB)', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Latency (ms)', color=color)
rects2 = ax2.bar(x + width/2, latencies, width, label='Latency (ms)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.xticks(x, labels)
plt.title('Performance Comparison: Model Size vs. Latency')
fig.tight_layout()
plt.savefig('benchmark_comparison.png')
print("Chart saved as benchmark_comparison.png")