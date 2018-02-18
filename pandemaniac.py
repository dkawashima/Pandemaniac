import sim
import json
import numpy as np
import networkx as nx
import operator
import sys
import argparse
# INPUT FILE IS OF FORM x.y.z.json 
# where x is number of players, y is the number of seeds, z is ID # for graph

# testgraph1.json has 500 nodes
# testgraph2.json has 1000 nodes

def main():
    '''
    Parses arguments of the form
        pandemaniac.py <input graph> <strategy>

    where the strategies are:
    R/r : Random
    D/d: Degree centrality
    B/b: Betweenness centrality
    K/k: Katz centrality
    E/e: Eigenvector centrality
    '''
    parser = argparse.ArgumentParser("Pandemaniac Solver")
    parser.add_argument("-f", "--filename", help = 
        "Name (.json not required) of the file containing the input graph. " +
        "Should be in JSON format."
        , type=str, required=True)
    parser.add_argument("-n", "--num_seeds", help = "Number of seed nodes to " +
        "start out with.", 
        type=int, required=True)
    parser.add_argument("-s", "--strategy", help = 
        "Strategy to use: R/r - random, D/d - degree, B/b - betweenness", 
        type=str, required=True)

    args = parser.parse_args()
    if ".json" in args.filename:
        GRAPH_FILENAME = args.filename
    else:
        GRAPH_FILENAME = args.filename + ".json"

    G = load_graph(GRAPH_FILENAME)
    NUM_ITERATIONS = 50
    NUM_SEEDS = args.num_seeds;

    strategy = args.strategy.lower()
    if strategy == 'r':
        output = random_strategy(G, NUM_ITERATIONS, NUM_SEEDS) 
    elif strategy == 'd':   
        output = degree_centrality_strategy(G, NUM_ITERATIONS, NUM_SEEDS)
    elif strategy == 'b':
        output = betweenness_centrality_strategy(G, NUM_ITERATIONS, NUM_SEEDS)
    elif strategy == 'k':
        output = katz_centrality_strategy(G, NUM_ITERATIONS, NUM_SEEDS)
    elif strategy == 'e':
        output = eigenvector_centrality_strategy(G, NUM_ITERATIONS, NUM_SEEDS)
    raw_file = GRAPH_FILENAME[:-5] # Take out the .json extension
    output_filename = raw_file + "_" + args.strategy + '.txt'
    output_list(output_filename, output)
    return

def load_graph(filename):
    ''' Loads in the graph.

    Args:
        filename -- the file from which to read adjacency list from

    Returns:
        a networkx graph
    '''
    with open(filename, 'r') as fr:
        graph = json.load(fr)
    
    # graph is now the data in the file stored as a dictionary
    # with each key as a node, and the values as a list of
    # adjacent nodes

    # Create an undirected graph
    G = nx.Graph(graph)
    return G

def eigenvector_centrality_strategy(G, num_iterations, num_seeds):
    ''' Picks the top nodes based on eigenvector centrality

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the eigenvector centrality
    '''

    centralities_dict = nx.eigenvector_centrality(G)
    sorted_centralities = sorted(centralities_dict.items(), 
        key=operator.itemgetter(1), reverse=True)[:num_seeds]
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys * num_iterations

def katz_centrality_strategy(G, num_iterations, num_seeds):
    ''' Picks the top nodes based on eigenvector centrality

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the katz centrality
    '''

    centralities_dict = nx.katz_centrality(G)
    sorted_centralities = sorted(centralities_dict.items(), 
        key=operator.itemgetter(1), reverse=True)[:num_seeds]
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys * num_iterations

def degree_centrality_strategy(G, num_iterations, num_seeds):
    ''' Picks the top nodes based on degree centrality

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the degree centrality
    '''

    centralities_dict = nx.degree_centrality(G)
    sorted_centralities = sorted(centralities_dict.items(), 
        key=operator.itemgetter(1), reverse=True)[:num_seeds]
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys * num_iterations

def random_strategy(G, num_iterations, num_seeds):
    ''' Basic strategy of picking random nodes 

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the random strategy
    '''

    # The output file should contain (num_seeds * num_iterations) lines
    all_seed_values = []
    for i in range(num_iterations):
        seed_values = np.random.choice(len(G), num_seeds, 
            replace=False).tolist()
        all_seed_values.extend(seed_values)
    #assert(len(all_seed_values) == num_seeds * num_iterations)
    return all_seed_values

def betweenness_centrality_strategy(G, num_iterations, num_seeds):
    ''' Picks the top nodes based on betweenness centrality

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the betweenness centrality
    '''

    centralities_dict = nx.betweenness_centrality(G)
    sorted_centralities = sorted(centralities_dict.items(), 
        key=operator.itemgetter(1), reverse=True)[:num_seeds]
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys * num_iterations

def random_strategy(G, num_iterations, num_seeds):
    ''' Basic strategy of picking random nodes 

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the random strategy
    '''

    # The output file should contain (num_seeds * num_iterations) lines
    all_seed_values = []
    for i in range(num_iterations):
        seed_values = np.random.choice(len(G), num_seeds, 
            replace=False).tolist()
        all_seed_values.extend(seed_values)
    return all_seed_values

def output_list(filename, seed_values):
    ''' Writes the given seed values to an output file for submisson

    Args:
        filename --         the filename of the file for which the values go into
        seed_values --      list of seed_values to write (one per line)
    
    Returns: n/a
    '''
    with open(filename, 'w') as fw:
        fw.write("\n".join(map(str, seed_values)))
    return

main()

