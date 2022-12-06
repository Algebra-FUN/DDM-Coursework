

import networkx as nx
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from itertools import takewhile
from functools import partial as part
import pickle


g = nx.read_gml("../data/netscience.gml")


plt.figure(figsize=(16, 16))
nx.draw_kamada_kawai(g)
plt.savefig("../fig/network.pdf")
plt.show()


def deg_dist_plot(g: nx.Graph, *args, plotter=plt, fit=False, start_from=None, **kwargs):
    degs = [k for n, k in g.degree if k > 0]
    km, kM = min(degs), max(degs)
    degx = np.arange(km, kM+1)
    disty = (np.bincount(degs)/len(degs))[km:kM+1]
    plotter.scatter(degx, disty, *args, **kwargs, label="degree distribution")
    if fit:
        if start_from is not None:
            disty = disty[degx > start_from]
            degx = degx[degx > start_from]
        degx = degx[disty > 0]
        disty = disty[disty > 0]
        logX = sm.add_constant(np.log10(degx))
        logy = np.log10(disty)

        mod = sm.OLS(logy, logX)
        results = mod.fit()
        a, b = results.params
        fity = degx**b*10**a
        plotter.plot(degx, fity, label="fit", c="orange")


plt.figure(figsize=(12, 4))
plt.subplot(121)
deg_dist_plot(g)
plt.xlabel("$k$")
plt.ylabel("$P_k$")
plt.grid(True)
plt.title("degree distribution(lin-lin)")
plt.subplot(122)
deg_dist_plot(g, fit=True, start_from=3)
plt.xlabel("$\log(k)$")
plt.ylabel("$\log(P_k)$")
plt.xscale("log")
plt.yscale("log")
plt.ylim([10**-4, 1])
plt.grid(True)
plt.title("degree distribution(log-log)")
plt.legend()
plt.savefig("../fig/deg-dist.pdf")
plt.show()


giant_subgraph = g.subgraph(max(nx.connected_components(g), key=len))


giant_subgraph.number_of_nodes(), np.log(giant_subgraph.number_of_nodes())


nx.average_shortest_path_length(giant_subgraph)


nx.average_clustering(giant_subgraph)


maxNComm = 410
gncomm = nx.community.girvan_newman(g)
comm_series = list(takewhile(lambda c: len(c) <= maxNComm, gncomm))

with open("../out/gn-comm-finding.pkl", "wb") as output:
    pickle.dump(comm_series, output)


with open("../out/gn-comm-finding.pkl", "rb") as inputs:
    comm_series = pickle.load(inputs)


def filter_comm(g: nx.Graph, communities, minsize=30):
    communities = list(filter(lambda c: len(c) >= minsize, communities))
    nodes = set().union(*communities)
    return communities, g.subgraph(nodes)


default_color_map = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan'
]


def community_plot(g: nx.Graph, communities, color_map=default_color_map, minsize=30, **kwargs):
    communities, g = filter_comm(g, communities, minsize)
    node_list = []
    node_color = []
    for i, community in enumerate(communities):
        for node in community:
            node_list.append(node)
            node_color.append(color_map[i])
    nx.draw_kamada_kawai(g, nodelist=node_list,
                         node_color=node_color, **kwargs)
    return communities


fig, axes = plt.subplots(3, 2, figsize=(49, 68))
for i, ax in enumerate(axes.flatten()):
    communities = community_plot(g, comm_series[i], ax=ax)
    C = len(comm_series[i])
    ax.set_title(f"C={C}", fontsize=50)
    ax.set_axis_off()
plt.savefig("../fig/communities.png")
plt.show()


def random_attack(g: nx.Graph, p):
    N = g.number_of_nodes()
    C = np.floor(N*(1-p)).astype(int)
    choices = np.random.choice(g.nodes, size=C, replace=False)
    return g.subgraph(choices)


def target_attack(g: nx.Graph, p):
    N = g.number_of_nodes()
    C = np.floor(N*(1-p)).astype(int)
    choices = [node for node, k in sorted(
        g.degree, key=lambda item:item[1])[:C]]
    return g.subgraph(choices)


def size_of_giant(g: nx.Graph):
    giant = g.subgraph(max(nx.connected_components(g), key=len))
    return giant.number_of_nodes()


np.random.seed(5003)
nsample = 50
ps = np.arange(0, 1, .02)
N = size_of_giant(g)
target_attack_survive = np.array(
    [*map(size_of_giant, map(part(target_attack, g), ps))])/N
random_attack_survive = np.array(
    [[*map(size_of_giant, map(part(random_attack, g), ps))] for _ in range(nsample)]).mean(axis=0)/N


plt.plot(ps, target_attack_survive, marker='x', label="target attack")
plt.plot(ps, random_attack_survive, marker='.', label="random attack")
plt.xlabel("node removal rate")
plt.ylabel("size of giant component")
plt.legend()
plt.grid(True)
plt.savefig("../fig/attack.pdf")
plt.show()
