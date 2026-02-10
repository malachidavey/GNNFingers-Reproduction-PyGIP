from pygip.datasets.dgl_tu_loader import load_tu_as_dgl

for name in ["ENZYMES", "PROTEINS", "AIDS"]:
    print(f"\n=== Loading {name} ===")
    graphs, ds = load_tu_as_dgl(name)

    print(f"Loaded {len(graphs)} graphs.")
    g = graphs[0]

    print("Graph type:", type(g))
    print("Nodes:", g.num_nodes(), "Edges:", g.num_edges())
    print("Features:", g.ndata["feat"].shape)

    # Graph-level label:
    print("Graph label:", g.graph_data["label"])
