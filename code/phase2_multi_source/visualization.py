"""Draw the heat network stored in results/phase2_solution.txt and save it as a PNG."""
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import ast

from utils import DATA_FILE, RESULTS_DIR, SOLUTION_FILE

# ── Read the solution file ───────────────────────────────────────────────────
with open(SOLUTION_FILE, "r", encoding="utf-8") as sol:
    content = sol.read()

source_id      = None
built          = {}
incoming_heat  = {}
outcoming_heat = {}

for line in content.split("\n"):
    if line.startswith("source_id"):
        source_id = int(line.split("=")[1].strip())
        break

sections = content.split("\n\n")
for section in sections:
    section = section.strip()
    if section.startswith("built"):
        built = ast.literal_eval(section.split(" = ", 1)[1])
    elif section.startswith("incoming_heat"):
        incoming_heat = ast.literal_eval(section.split(" = ", 1)[1])
    elif section.startswith("outcoming_heat"):
        outcoming_heat = ast.literal_eval(section.split(" = ", 1)[1])

# ── Read the Excel data (same file as the MILP) ──────────────────────────────
def read_excel_data(filename, sheet_name):
    data = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    values = data.values
    if min(values.shape) == 1:
        if values.shape[0] == 1:
            values = values.tolist()
        else:
            values = values.transpose()
            values = values.tolist()
        return values[0]
    else:
        data_dict = {}
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                data_dict[(i, j)] = values[i][j]
        return data_dict

f = DATA_FILE
number_of_nodes = read_excel_data(f, "Nodes")[0]
cords           = read_excel_data(f, "NodesCord")
D               = read_excel_data(f, "EdgesDemandPeak(i)")
betta           = read_excel_data(f, "Betta")[0]
lambda_         = read_excel_data(f, "Lambda")[0]

source_node = number_of_nodes + source_id  # e.g. 20 for source_id=0

# ── Compute P (net power at each node) and V (flow on each pipe) ─────────────
all_nodes_in_sol = list(range(number_of_nodes)) + [source_node]
Pin = {node: 0.0 for node in all_nodes_in_sol}
Pout = {node: 0.0 for node in all_nodes_in_sol}

for j in range(number_of_nodes):
    for i in all_nodes_in_sol:
        if i != j and built.get(i, {}).get(j, 0) == 1:
            Pin[j] += incoming_heat.get(i, {}).get(j, 0.0)
            Pout[j] += outcoming_heat.get(i, {}).get(j, 0.0)

Pin[source_node] = sum(
    incoming_heat.get(source_node, {}).get(j, 0.0)
    for j in range(number_of_nodes)
    if built.get(source_node, {}).get(j, 0) == 1
)
Pout[source_node] = Pin[source_node]

V = {}
for i in all_nodes_in_sol:
    V[i] = {}
    for j in all_nodes_in_sol:
        V[i][j] = incoming_heat.get(i, {}).get(j, 0.0)

# ── Build the graph ───────────────────────────────────────────────────────────
g = nx.DiGraph()

# All consumer nodes + the source
for i in range(number_of_nodes):
    g.add_node(i)
g.add_node(source_node)

# Built pipes (between consumers + source → consumers)
edges_built = []
for i in all_nodes_in_sol:
    for j in range(number_of_nodes):
        if i != j and built.get(i, {}).get(j, 0) == 1:
            g.add_edge(i, j)
            edges_built.append((i, j))

connected_nodes = set()
for (i, j) in edges_built:
    connected_nodes.add(i)
    connected_nodes.add(j)

# ── GPS positions ─────────────────────────────────────────────────────────────
pos = {}
for i in range(number_of_nodes):
    pos[i] = (cords[(i, 0)], cords[(i, 1)])
pos[source_node] = (cords[(source_node, 0)], cords[(source_node, 1)])

# ── Labels ────────────────────────────────────────────────────────────────────
labels = {}
for i in range(number_of_nodes):
    excel_num = i + 1
    if i in connected_nodes:
        labels[i] = f"{excel_num}\nPout={Pout[i]:.1f}"
    else:
        labels[i] = f"{excel_num}"
labels[source_node] = f"S{source_id}\n(node {source_node})"

# ── Colors and sizes 
pin_values = [Pin[n] for n in g.nodes() if Pin.get(n, 0.0) > 0]
pin_min = min(pin_values) if pin_values else 0.0
pin_max = max(pin_values) if pin_values else 0.0

def size_from_pin(pin):
    if pin <= 0:
        return 500
    if pin_max == pin_min:
        return 1800
    return 700 + ((pin - pin_min) / (pin_max - pin_min))**0.5 * 2600

node_colors = []
node_sizes  = []
for n in g.nodes():
    if n == source_node:
        node_colors.append('#e74c3c')   # red = source
        node_sizes.append(size_from_pin(Pin[n]))
    elif n in connected_nodes:
        node_colors.append('#2ecc71')   # green = connected
        node_sizes.append(size_from_pin(Pin[n]))
    else:
        node_colors.append('#bdc3c7')   # grey = not connected
        node_sizes.append(500)

# ── Pipe colors (proportional to V) ──────────────────────────────────────────
edge_widths = []
edge_colors = []
for (u, v) in edges_built:
    v_val = V[u][v]
    edge_widths.append(max(1.0, 0.005 * v_val))
    edge_colors.append(v_val)

cmap = plt.cm.plasma

#  Drawing 
plt.figure(figsize=(14, 10))

nx.draw_networkx_nodes(
    g, pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors='black',
    linewidths=0.8
)

nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, font_weight='bold')

if edges_built:
    nx.draw_networkx_edges(
        g, pos,
        edgelist=edges_built,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=edge_widths,
        connectionstyle='arc3,rad=0.1',
    )

    for (u, v) in edges_built:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        v_val = V[u][v]
        if v_val > 0:
            plt.text(mx, my, f"P={v_val:.1f}", fontsize=9, color='#2980b9',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor='#2980b9', alpha=0.8))

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
    )
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="Power (kW)")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black',
                   label=f'Source (node {source_node}, source_id={source_id})'),
    mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Connected'),
    mpatches.Patch(facecolor='#bdc3c7', edgecolor='black', label='Not connected'),
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax = plt.gca()
ax.set_axis_off()
n_connected = len([n for n in connected_nodes if n != source_node])
plt.title(f"Heat network — {n_connected} / {number_of_nodes} nodes connected  |  source_id={source_id}",
          fontsize=14, fontweight='bold')
plt.tight_layout()
output_png = RESULTS_DIR / "phase2_heat_network.png"
plt.savefig(output_png, dpi=150)
plt.show()
print(f"\n[Saved to {output_png}]")
