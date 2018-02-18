import sim
import sys
import json
import glob
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

def main():
    '''
    Expected usage is run_sim.py <inputgraph_filename> <strategy> <strategy1> ... num_seeds
    
    If no strategies are specified, then by default, the program will test
    all strategies in the current folder against one another

    The logic in this program relies on the fact that if our input graph has
    filename <x.json> then our strategy outputs will have filenames
    <x_b.txt>, <x_r.txt>, <x_d.txt>, etc. This follows the naming conventions
    of pandemaniac.py. 
    '''
    input_file = sys.argv[1]
    if ('.json') in input_file:
        input_file = input_file[:-5]
    json_input = input_file + '.json'
    specified_strategies = sys.argv[2:len(sys.argv)-1]
    num_seeds = int(sys.argv[-1])
    # Read in JSON graph
    with open(json_input, 'r') as fr:
        graph = json.load(fr)
    strategy_files = glob.glob('./*.txt')
    nodes = dict()
    for i in strategy_files:
        # i has form "./x_r.txt" where x is the name of the actual file and 
        # r is the strategy (in this case, random)
        if (specified_strategies != [] and i[len(input_file) + 3] not in specified_strategies):
            continue
        for k in range(50: # 50 iterations
            
        with open(i) as f:
            content = []
            for j in range(num_seeds):
                content.append(f.readline().strip())
            #print(content)
            # the -4 is to avoid the '.txt' extension, the +3 is to avoid the './' and '_' of the strategy filenames
            nodes[i[len(input_file) + 3:-4]] = content   # keys should only be r, b, or d

    # run simulation
    results = sim.run(graph, nodes)
    print(results)

main()