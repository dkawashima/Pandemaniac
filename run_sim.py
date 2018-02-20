import sim
import sys
import json
import glob
import operator
from collections import defaultdict

'''
This program uses sim.py and runs strategies against one another.
Gives us an idea of how different strategies play against
each other.
'''

'''
===========
   USAGE        for sim.py
===========

>>> import sim
>>> sim.run([graph], [dict with keys as names and values as a list of nodes])

Returns a dictionary containing the names and the number of nodes they got.

Example:
>>> graph = {"2": ["6", "3", "7", "2"], "3": ["2", "7, "12"], ... }
>>> nodes = {"strategy1": ["1", "5"], "strategy2": ["5", "23"], ... }
>>> sim.run(graph, nodes)
>>> {"strategy1": 243, "strategy6": 121, "strategy2": 13}

Possible Errors:
- KeyError: Will occur if any seed nodes are invalid (i.e. do not exist on the
            graph).
'''

def load_graph(filename):
    ''' Loads in the graph into a dictionary

    Args:
        filename -- the file from which to read adjacency list from

    Returns:
        a dictionary with keys as nodes and values as lists of neighbors
    '''
    with open(filename, 'r') as fr:
        graph = json.load(fr)
    return graph

def load_strategies(specified_strategies, input_file):
    ''' Loads all the strategy files to be used in the simulation
    into a dictionary. Note that for each strategy (key), we will
    have a list of 50 * num_seeds elements in corresponding list 
    (values). We need this because we are running sim.py over
    50 iterations not just 1 and thus need seeds information for
    all 50 runs.

    Args: specified_strategies -- a list of strategies that the user inputs
                                  to be put up against one another.
          inputfile            -- name of the input graph file. Used to
                                  detect strategy files for the specific input 
                                  graph.

    Returns: dict of strategy(key) to list of seed_nodes(value) over all 50 runs
    '''
    strategy_files = glob.glob('./*.txt')
    strategy_files = [i for i in strategy_files if input_file in i]
    strategies_dict = dict()

    for i in strategy_files:
        # i has form "./x_r.txt" where x is the name of the actual file and 
        # r is the strategy (in this case, random)
        with open(i) as f:
            if (specified_strategies != [] and i[len(input_file) + 3:-4] not in specified_strategies):
                continue
            content = [x.strip() for x in f.readlines()]
            # the -4 is to avoid the '.txt' extension, the +3 is to avoid the './' and '_' of the strategy filenames
            strategies_dict[i[len(input_file) + 3:-4]] = content
    return strategies_dict

def run_simulations(strategies_dict, num_seeds, graph):
    ''' Runs 50 iterations of sim.py using specified strategies. Outputs
    the results of each run and also the final count of how many times each
    strategy took the most number of nodes in an iteration/run.

    Args: strategies_dict      -- dict of strategy(key) to list of seed_nodes(value) 
                                  over all 50 runs
          num_seeds            -- the number of seeds/nodes to pick per run
          graph                -- the input graph

    Returns: n/a                   
    '''
    nodes = dict()
    results_dict = defaultdict(int) # Stores how many times this strategy won out
    for i in range(50):
        start = i * num_seeds
        end = start + num_seeds
        for s in strategies_dict: # s should only be r, b, or d
            nodes[s] = strategies_dict[s][start:end]
        # run simulation for each iteration
        results = sim.run(graph, nodes)
        print(results)
        best_strategy = max(results.iteritems(), key=operator.itemgetter(1))[0]
        results_dict[best_strategy] += 1
    print
    print("----------------------- OVERALL RESULTS -----------------------")
    print(results_dict)

def get_user_input():
    ''' Gets graph filename and strategies from user input.

    Args:   n/a

    Returns: ofilename, list of strategies                   
    '''
    input_file = raw_input("Enter the graph filename (leave out .json extension) -> ")
    strategy_str = raw_input("Enter the strategies to run separated by spaces " +
                    "(leave blank to run all strategies in current folder) -> ")
    strategy_lst = strategy_str.strip().split()
    return input_file.strip(), strategy_lst

def main():
    ''' Runs different strategies against one another

    If no strategies are specified, then by default, the program will test
    all strategies in the current folder against one another

    The logic in this program relies on the fact that if our input graph has
    filename <x.json> then our strategy outputs will have filenames
    <x_b.txt>, <x_r.txt>, <x_d.txt>, etc. This follows the naming conventions
    of pandemaniac.py. 
    '''

    # Get user inputs
    input_file, specified_strategies = get_user_input()
    json_input = input_file + '.json'
    # INPUT FILE IS OF FORM x.y.z
    # where x is number of players, y is the number of seeds, z is ID # for graph
    num_seeds = int(input_file.split('.')[1])

    # Read in JSON graph
    graph = load_graph(json_input)

    # Load in strategies information
    strategies_dict = load_strategies(specified_strategies, input_file)
    # Run the 50 iterations now
    run_simulations(strategies_dict, num_seeds, graph)

main()

