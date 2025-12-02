import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')

from neuronova.engine import NeuroNovaEngineSemantic
from neuronova.config import SAFETY_DISCLAIMER
import matplotlib.pyplot as plt
import numpy as np

print(SAFETY_DISCLAIMER)
print("\n" + "="*60)
print("NeuroNova Non-Interactive Demo")
print("="*60 + "\n")

os.makedirs("assets", exist_ok=True)

engine = NeuroNovaEngineSemantic()

examples = [
    "I'm excited about my research and can't wait to start!",
    "I feel hopeless and sometimes I want to end it.",
    "That person is such an idiot, they ruined my day."
]

results = []

for i, text in enumerate(examples, 1):
    print(f"\n{'='*60}")
    print(f"Example {i}: {text}")
    print('='*60)
    
    res = engine.process_text(text)
    results.append((text, res))
    
    if engine.lif.v_history is not None and engine.lif.v_history.size > 0:
        v_hist = engine.lif.v_history
        plt.figure(figsize=(10, 6))
        t = np.arange(v_hist.shape[0])
        nplot = min(12, v_hist.shape[1])
        cmap = plt.get_cmap('tab10')
        for j in range(nplot):
            color = cmap(j % 10)
            plt.plot(t, v_hist[:, j], label=f"N{j}", color=color, 
                    alpha=0.9 if j%2==0 else 0.6, linewidth=0.8)
        mean_v = v_hist.mean(axis=1)
        plt.plot(t, mean_v, color='k', linewidth=1.5, label='mean')
        plt.legend(loc="upper right", fontsize=7, ncol=2)
        plt.title(f"Example {i}: Membrane Potentials")
        plt.xlabel("time step")
        plt.ylabel("voltage (a.u.)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"assets/volt_example_{i}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: assets/volt_example_{i}.png")

if len(results) > 0 and engine.tracker.records:
    try:
        engine.tracker._create_figure()
        engine.tracker._update_plot()
        engine.tracker.fig.savefig("assets/conversation_graph.png", dpi=150, bbox_inches='tight')
        plt.close(engine.tracker.fig)
        print(f"\nSaved: assets/conversation_graph.png")
    except Exception as e:
        print(f"Could not save conversation graph: {e}")

with open("assets/transcript.txt", "w", encoding='utf-8') as f:
    f.write("NeuroNova Demo Transcript\n")
    f.write("="*60 + "\n\n")
    for i, (text, res) in enumerate(results, 1):
        f.write(f"Example {i}:\n")
        f.write(f"INPUT: {text}\n")
        f.write(f"Dominant Emotion: {res.get('dominant', 'Unknown')}\n")
        f.write(f"Reply: {res.get('reply', 'N/A')}\n")
        if res.get('meta'):
            f.write(f"Meta: {res['meta']}\n")
        f.write("\n" + "-"*60 + "\n\n")

print(f"\n{'='*60}")
print("Demo complete!")
print("="*60)
print("\nGenerated files:")
print("  - assets/transcript.txt")
for i in range(1, len(examples) + 1):
    print(f"  - assets/volt_example_{i}.png")
print("  - assets/conversation_graph.png (if available)")
print("\nPlease review these files to see the system's responses.")


