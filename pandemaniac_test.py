import sim
import json
import numpy as np

GRAPH_FILENAME = "testgraph1.json"
SEED_FILENAME = "seed_nodes.txt"
NUM_ITERATIONS = 50
NUM_SEEDS = 40

def output_list(nodes):
	with open(SEED_FILENAME, 'w') as fw:
		for i in range(NUM_ITERATIONS):
			for val in nodes["strategy" + str(i + 1)]:
				fw.write(val + "\n")

with open(GRAPH_FILENAME, 'r') as fr:
	graph = json.load(fr)

graphLength = len(graph)
nodes = {}
for i in range(NUM_ITERATIONS):
	nodes["strategy" + str(i + 1)] = list(map(str, np.random.choice(graphLength, NUM_SEEDS)))
	#print (np.random.choice(graphLength, NUM_SEEDS))

#output_list(nodes)
print(sim.run(graph, nodes))