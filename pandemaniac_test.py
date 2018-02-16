import sim
import json
import numpy as np
import networkx as nx
from heapq import nlargest
from networkx.readwrite import json_graph

GRAPH_FILENAME = "testgraph1.json"
SEED_FILENAME = "seed_nodes.txt"
NUM_ITERATIONS = 50
NUM_SEEDS = 10

def output_list(nodes):
	with open(SEED_FILENAME, 'w') as fw:
		for i in range(NUM_ITERATIONS):
			for val in nodes["strategy" + str(i + 1)]:
				fw.write(val + "\n")

with open(GRAPH_FILENAME, 'r') as fr:
	graph = json.load(fr)

graphLength = len(graph)

# Get on NetworkX
nxGraph = nx.Graph()
for key in graph:
	for neighbor in graph[key]:
		nxGraph.add_edge(key, neighbor)
centMeasure = nx.degree_centrality(nxGraph)
#print(centMeasure)
topKtuples = nlargest(NUM_SEEDS * 2, centMeasure.items(), key=lambda e: e[1])
print (topKtuples)
graphLength = len(graph)
topK = [x[0] for x in topKtuples]

nodes = {}
for i in range(NUM_ITERATIONS):
	nodes["strategy" + str(i + 1)] = list(map(str, np.random.choice(topK, size=NUM_SEEDS, replace=False)))
	#nodes["strategy" + str(i + 1)] = list(map(str, np.random.choice(graphLength, size=NUM_SEEDS, replace=False)))
#output_list(nodes)
print(sim.run(graph, nodes))