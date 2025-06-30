import networkx as nx
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
from datasets import util
from pyvis.network import Network

def visualize_dgl_graph(g):

    # Convert to NetworkX graph
    nx_g = g.to_networkx()

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(nx_g, k=0.5, iterations=100)
    nx.draw(nx_g, pos, with_labels=True, font_size=8, node_color='skyblue',
            node_size=10, edge_color='gray', width=0.5, arrows=False)
    plt.show()


if __name__ == "__main__":
    cad_file = "./data/FABWave/Tag_Holder/bin/643a6c34-4ac6-4d3a-84d1-a152ea320fac.bin"
    graph = load_graphs(cad_file)[0][0]


    # Visualize
    visualize_dgl_graph(graph)
