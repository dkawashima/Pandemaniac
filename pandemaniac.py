import sim
import json
import numpy as np
import networkx as nx
import operator
import sys
from heapq import nlargest
from networkx.algorithms.approximation import min_weighted_vertex_cover
from collections import defaultdict
# testgraph1.json has 500 nodes
# testgraph2.json has 1000 nodes

def main():
    ''' Main function: Gets user input and creates necessary output files
    '''
    filename, num_players, num_seeds, strategies, proportions = get_user_input()
    GRAPH_FILENAME = filename + ".json"

    G = load_graph(GRAPH_FILENAME)
    NUM_ITERATIONS = 50
    for s in strategies:
        strategy = s.lower()
        if strategy == 'r':
            output = random_strategy(G, num_seeds) 
        elif strategy == 'd':   
            output = degree_centrality_strategy(G, num_seeds)
        elif strategy == 'e':
            output = eigenvector_centrality_strategy(G, num_seeds)
        elif strategy == 'b':
            output = betweenness_centrality_strategy(G, num_seeds)
        elif strategy == 'c':
            output = clustering_coefficient_strategy(G, num_seeds)
        elif strategy == 'cl':
            output = closeness_centrality_strategy(G, num_seeds)
        elif strategy == 'k':
            output = katz_centrality_strategy(G, num_seeds)
        elif strategy == 'm':
            output = mst_strategy(G, num_seeds)
        elif strategy == 's':
            output = dominating_set_strategy(G, num_seeds)
        elif strategy == 'v':
            output = vertex_cover_strategy(G, num_seeds)
        elif strategy == 'de':
            output = degree_eigenvector_centrality_mixed_strategy(G, num_seeds)
        elif strategy == 'mm':
            output = multiple_mixed_strategy(G, num_seeds)
        elif strategy == 'dc':
            output = degree_closeness_centrality_mixed_strategy(G, num_seeds)
        elif strategy == 'q':
            output = cliques_strategy(G, num_seeds)
        elif strategy == 't':
            output = triangles_strategy(G, num_seeds)
        elif strategy == 'f':
            output = final_strategy(G, num_seeds)
        elif strategy == 'fds':
            output = first_day_strategy(G, NUM_ITERATIONS, proportions)
        else:
            assert(False)
        if strategy != 'fds':
            output *= NUM_ITERATIONS
        raw_file = GRAPH_FILENAME[:-5] # Take out the .json extension
        output_filename = raw_file + "_" + strategy + '.txt'
        output_list(output_filename, output)
        print(strategy + " successfully generated.")
    return

def get_sorted_ranks(G, seeds_to_generate):
    ''' Picks top 40 nodes based on an equally weighted
    conglomeration of five strategies:

    degree centrality, number of triangles, eigenvector
    centrality, and closeness centrality, vertex cover

    Args:
    G -- the input graph
    seeds_to_generate -- number of nodes for each strat to produce,
        also the total number of nodes we want after blending

    Returns:
    Dict mapping node to importance
    '''
    triangle = triangles_strategy(G, seeds_to_generate)
    eigen = eigenvector_centrality_strategy(G, seeds_to_generate)
    closeness = closeness_centrality_strategy(G, seeds_to_generate)
    degree = degree_centrality_strategy(G, seeds_to_generate)
    vertex = vertex_cover_strategy(G, seeds_to_generate)
    total_ranks = defaultdict(float)
    offset = 4.0
    decrement = 0.2

    for i in range(0, seeds_to_generate):
        total_ranks[triangle[i]] += offset
        total_ranks[eigen[i]] += offset
        total_ranks[closeness[i]] += offset
        total_ranks[degree[i]] += offset
        total_ranks[vertex[i]] += offset
        offset -= decrement
    return nlargest(seeds_to_generate, total_ranks.items(), key=operator.itemgetter(1))

def choose_nodes(sorted_ranks, proportions):
    ''' From the result of get_sorted_ranks,
    choose according to proportions

    Args:
    sorted_ranks -- result of get_sorted_ranks (dict of node to importance value)

    Returns:
    List of 10 nodes chosen from top 40 most important
    '''
    indices = np.random.choice(10, size=10)
    # What different parts of indices correspond to:
    # indices[0]   -> 0-9
    # indices[1:3] -> 10-19
    # indices[3:6] -> 20-29
    # indices[6:]  -> 30-39
    slicing_indices = np.cumsum(proportions)
    indices[slicing_indices[0]:slicing_indices[1]] += 10
    indices[slicing_indices[1]:slicing_indices[2]] += 20
    indices[slicing_indices[2]:slicing_indices[3]] += 30

    return [sorted_ranks[i][0] for i in indices]

def first_day_strategy(G, NUM_ITERATIONS, proportions):
    ''' Picks top 40 nodes based on an equal weighting 
    and conglomeration of four strategies, and then 
    chooses 10 to return. Given that the top 40 are ordered
    by importance, it chooses 1 from 0-9, 2 from 10-19,
    3 from 20-29, and 4 from 30-39:

    Strategies:
    degree centrality, number of triangles, eigenvector
    centrality, and closeness centrality, vertex cover

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on this mixed strategy
    '''
    sorted_ranks = get_sorted_ranks(G, len(proportions) * 10)
    output = []
    for i in range(NUM_ITERATIONS):
        output += choose_nodes(sorted_ranks, proportions)
    return output

def final_strategy(G, num_seeds):
    ''' Picks top degreed nodes based on an equal weighting 
    and conglomeration of four strategies:

    degree centrality, number of triangles, eigenvector
    centrality, and closeness centrality

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on this mixed strategy
    '''
    seeds_to_generate = int(num_seeds * 1.5)
    triangle = triangles_strategy(G, seeds_to_generate)
    eigen = eigenvector_centrality_strategy(G, seeds_to_generate)
    closeness = closeness_centrality_strategy(G, seeds_to_generate)
    degree = degree_centrality_strategy(G, seeds_to_generate)
    vertex = vertex_cover_strategy(G, seeds_to_generate)
    total_ranks = defaultdict(float)
    offset = 4.0
    decrement = 0.2

    for i in range(0, seeds_to_generate):
        total_ranks[triangle[i]] += offset
        total_ranks[eigen[i]] += offset
        total_ranks[closeness[i]] += offset
        total_ranks[degree[i]] += offset
        total_ranks[vertex[i]] += offset
        offset -= decrement
    sorted_ranks = nlargest(num_seeds, total_ranks.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_ranks]
    return node_keys

def get_user_input():
    ''' Gets user input for name of graph and strategies to run
    where the strategies are:
    R/r: Random
    D/d: Degree centrality
    B/b: Betweenness centrality
    K/k: Katz centrality
    E/e: Eigenvector centrality
    C/c: Clustering coefficient
    Cl/cl: Closeness centrality
    M/m: Minimum spanning tree
    S/s: Dominating set
    V/v : Vertex cover
    DE/de : Degree/Eigenvalue Centrality Mixed Strategy
    DC/dc: Degree/ Closeness Centrality Mixed Strategy
    MM/mm : Multiple Mixed Strategy
    Q/q: Number of Cliques
    T/t: Number of triangles

    Also extrapolates the num_seeds from the name of the graph

    Arguments: n/a
    Returns:   name of graph, number of seeds, list of strategies
    '''
    filename = raw_input("Enter the filename (leave out .json extension) -> ")
    # INPUT FILE IS OF FORM x.y.z
    # where x is number of players, y is the number of seeds, z is ID # for graph
    filename_split = filename.strip().split('.')
    assert(len(filename_split) == 3)
    num_players = filename_split[0]
    num_seeds = filename_split[1]
    strategy_str = raw_input("Enter the strategies to run separated by spaces -> ")
    strategy_lst = strategy_str.strip().split()
    proportions_str = raw_input("Enter the proportions for fds separated by spaces -> ")
    proportions_lst = proportions.strip().split()

    return filename, int(num_players), int(num_seeds), strategy_lst, proportions_lst

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

def vertex_cover_strategy(G, num_seeds):
    ''' Picks top degreed nodes from vertex cover (VC) of 
    input graph. If there are less nodes in the VC than 
    num_seeds, then we pick from top degreed nodes of the 
    input graph.

    Given an undirected graph G = (V, E) and a function w 
    assigning nonnegative weights to its vertices, 
    find a minimum weight subset of V such that each edge in E is 
    incident to at least one vertex in the subset.
    
    Note: every vertex cover is a dominating set, but not every
    dominating set is a vertex cover

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on VC strategy
    '''

    vc_nodes = min_weighted_vertex_cover(G)
    nodes_to_remove = G.nodes() - vc_nodes
    subgraph = G.copy()
    subgraph.remove_nodes_from(nodes_to_remove)
    number_of_nodes = num_seeds
    if len(vc_nodes) < num_seeds:
        number_of_nodes = len(vc_nodes)
    centralities_dict = nx.degree_centrality(subgraph)
    sorted_centralities = nlargest(number_of_nodes, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]

    nodes_top_degrees = degree_centrality_strategy(G, num_seeds)
    # In case we need more seed nodes, we simply pull from the top degreed nodes of the input graph
    i = 0
    while (len(node_keys) != num_seeds):
        if nodes_top_degrees[i] not in node_keys:
            node_keys.append(nodes_top_degrees[i])
        i += 1
    assert(len(node_keys) == num_seeds)
    return node_keys

def dominating_set_strategy(G, num_seeds):
    ''' Picks top degreed nodes from dominating set(ds) of 
    input graph. If there are less nodes in the ds than 
    num_seeds, then we pick from top degreed nodes of the 
    input graph.

    A dominating set for a graph G = (V, E) is a node subset 
    D of V such that every node not in D is adjacent to at least 
    one member of D.

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on DS strategy
    '''

    ds_nodes = nx.dominating_set(G)
    nodes_to_remove = G.nodes() - ds_nodes
    subgraph = G.copy()
    subgraph.remove_nodes_from(nodes_to_remove)
    assert(len(G.nodes()) > len(subgraph.nodes()))
    number_of_nodes = num_seeds
    if len(ds_nodes) < num_seeds:
        number_of_nodes = len(ds_nodes)
    centralities_dict = nx.degree_centrality(subgraph)
    sorted_centralities = nlargest(number_of_nodes, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]

    nodes_top_degrees = degree_centrality_strategy(G, num_seeds)
    # In case we need more seed nodes, we simply pull from the top degreed nodes of the input graph
    i = 0
    while (len(node_keys) != num_seeds):
        if nodes_top_degrees[i] not in node_keys:
            node_keys.append(nodes_top_degrees[i])
        i += 1
    assert(len(node_keys) == num_seeds)
    return node_keys

def mst_strategy(G, num_seeds):
    ''' Picks top degreed nodes from MST of input graph.
    If there are less nodes in the MST than num_seeds, 
    then we pick from top degreed nodes of the input graph

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on MST
    '''

    mst = nx.minimum_spanning_tree(G)
    number_of_nodes = num_seeds
    if len(mst) < num_seeds:
        number_of_nodes = len(mst)
    centralities_dict = nx.degree_centrality(mst)
    sorted_centralities = nlargest(number_of_nodes, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]

    nodes_top_degrees = degree_centrality_strategy(G, num_seeds)
    # In case we need more seed nodes, we simply pull from the top degreed nodes of the input graph
    i = 0
    while (len(node_keys) != num_seeds):
        if nodes_top_degrees[i] not in node_keys:
            node_keys.append(nodes_top_degrees[i])
        i += 1
    assert(len(node_keys) == num_seeds)
    return node_keys

def clustering_coefficient_strategy(G, num_seeds):
    ''' Picks the top nodes based on clustering coefficient

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on clustering coefficients
    '''

    clustering_dict = nx.clustering(G)
    sorted_clustering_nodes = nlargest(num_seeds, clustering_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_clustering_nodes]
    return node_keys

def eigenvector_centrality_strategy(G, num_seeds):
    ''' Picks the top nodes based on eigenvector centrality

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the eigenvector centrality
    '''

    centralities_dict = nx.eigenvector_centrality(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys

def cliques_strategy(G, num_seeds):
    ''' Picks the top nodes based on number of cliques per node

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the cliques
    '''

    centralities_dict = nx.number_of_cliques(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys

def triangles_strategy(G, num_seeds):
    ''' Picks the top nodes based on number of triangles a node is involved in

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the triangles
    '''

    centralities_dict = nx.triangles(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys

def katz_centrality_strategy(G, num_seeds):
    ''' Picks the top nodes based on katz centrality

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the katz centrality
    '''

    centralities_dict = nx.katz_centrality(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys

def closeness_centrality_strategy(G, num_seeds):
    ''' Picks the top nodes based on closeness centrality

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the closeness centrality
    '''

    centralities_dict = nx.closeness_centrality(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys

def degree_centrality_strategy(G, num_seeds):
    ''' Picks the top nodes based on degree centrality

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the degree centrality
    '''

    centralities_dict = nx.degree_centrality(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys 

def degree_closeness_centrality_mixed_strategy(G, num_seeds):
    ''' Picks the top nodes based on mixed strategy of degree/closeness

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes comprised of 50% being based on degree centrality,
    and 50% based on closeness centrality
    '''

    degree_centralities_dict = nx.degree_centrality(G)
    sorted_degree_centralities = nlargest((num_seeds + 1)/ 2, degree_centralities_dict.items(), key=operator.itemgetter(1))
    degree_node_keys = [i[0] for i in sorted_degree_centralities]

    closeness_centralities_dict = nx.eigenvector_centrality(G)
    closeness_centralities_dict = {key: closeness_centralities_dict[key] for key in closeness_centralities_dict if key not in degree_node_keys}
    sorted_closeness_centralities = nlargest(num_seeds / 2, closeness_centralities_dict.items(), key=operator.itemgetter(1))
    
    node_keys = degree_node_keys + [i[0] for i in sorted_closeness_centralities]

    return node_keys 

def degree_eigenvector_centrality_mixed_strategy(G, num_seeds):
    ''' Picks the top nodes based on degree centrality

    Args:
        G --                the input graph
        num_iterations --   the number of rounds
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes comprised of 50% being based on degree centrality,
    and 50% based on eigenvector centrality

    The following permutation scored 250-249 over degree centrality on the
    TA_degree graph on Day 3.
    7 with degree centrality
    3 with eigenvector centrality
    '''

    degree_centralities_dict = nx.degree_centrality(G)
    sorted_degree_centralities = nlargest((num_seeds + 1)/ 2, degree_centralities_dict.items(), key=operator.itemgetter(1))
    degree_node_keys = [i[0] for i in sorted_degree_centralities]

    eigenvector_centralities_dict = nx.eigenvector_centrality(G)
    eigenvector_centralities_dict = {key: eigenvector_centralities_dict[key] for key in eigenvector_centralities_dict if key not in degree_node_keys}
    sorted_eigenvector_centralities = nlargest(num_seeds / 2, eigenvector_centralities_dict.items(), key=operator.itemgetter(1))
    
    node_keys = degree_node_keys + [i[0] for i in sorted_eigenvector_centralities]

    return node_keys 

def multiple_mixed_strategy(G, num_seeds):
    ''' Picks the top nodes based on four mixed strategies

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the degree centrality
    '''

    number_of_nodes = num_seeds

    # Degree Centralities (25%)
    degree_centralities_dict = nx.degree_centrality(G)
    sorted_degree_centralities = nlargest(num_seeds / 4, degree_centralities_dict.items(), key=operator.itemgetter(1))
    degree_node_keys = [i[0] for i in sorted_degree_centralities]

    # Eigenvector Centralities (25%)
    eigenvector_centralities_dict = nx.eigenvector_centrality(G)
    eigenvector_centralities_dict = {key: eigenvector_centralities_dict[key] for key in eigenvector_centralities_dict if key not in degree_node_keys}
    sorted_eigenvector_centralities = nlargest(num_seeds / 4, eigenvector_centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = degree_node_keys + [i[0] for i in sorted_eigenvector_centralities]

    # Vertex Cover (25%)
    vc_nodes = min_weighted_vertex_cover(G)
    nodes_to_remove = G.nodes() - vc_nodes
    subgraph = G.copy()
    subgraph.remove_nodes_from(nodes_to_remove)
    if len(vc_nodes) < num_seeds:
        number_of_nodes = len(vc_nodes)
    vc_centralities_dict = nx.degree_centrality(subgraph)
    vc_centralities_dict = {key: vc_centralities_dict[key] for key in vc_centralities_dict if key not in node_keys}
    sorted_vc_centralities = nlargest(min(num_seeds / 4, number_of_nodes), vc_centralities_dict.items(), key=operator.itemgetter(1))
    node_keys += [i[0] for i in sorted_vc_centralities]

    # MST (25%)
    mst = nx.minimum_spanning_tree(G)
    number_of_nodes = num_seeds
    if len(mst) < num_seeds:
        number_of_nodes = len(mst)
    mst_centralities_dict = nx.degree_centrality(mst)
    mst_centralities_dict = {key: mst_centralities_dict[key] for key in mst_centralities_dict if key not in node_keys}
    sorted_mst_centralities = nlargest(min(num_seeds / 4, number_of_nodes), mst_centralities_dict.items(), key=operator.itemgetter(1))
    node_keys += [i[0] for i in sorted_mst_centralities]

    nodes_top_degrees = degree_centrality_strategy(G, num_seeds)
    # In case we need more seed nodes, we simply pull from the top degreed nodes of the input graph
    i = 0
    while (len(node_keys) != num_seeds):
        if nodes_top_degrees[i] not in node_keys:
            node_keys.append(nodes_top_degrees[i])
        i += 1
    assert(len(node_keys) == num_seeds)


    return node_keys 

def betweenness_centrality_strategy(G, num_seeds):
    ''' Picks the top nodes based on betweenness centrality

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the betweenness centrality
    '''

    centralities_dict = nx.betweenness_centrality(G)
    sorted_centralities = nlargest(num_seeds, centralities_dict.items(), key=operator.itemgetter(1))
    node_keys = [i[0] for i in sorted_centralities]
    return node_keys

def random_strategy(G, num_seeds):
    ''' Basic strategy of picking random nodes 

    Args:
        G --                the input graph
        num_seeds --        the number of seed nodes to select

    Returns: list of output nodes based on the random strategy
    '''

    seed_values = np.random.choice(len(G), num_seeds, 
        replace=False).tolist()
    return seed_values

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

