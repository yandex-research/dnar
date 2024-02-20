# Discrete Neural Algorithmic Reasoning
This repository contains the code to reproduce the experiments from "Discrete Neural Algorithmic Reasoning" paper. 

## Setup
Before running the source code, make sure to install the project dependencies:
```bash
pip install -r requirements.txt
```

## Main experiments

### Algorithms
- Breadth-first search
- Depth-first search
- Minimum spanning tree (Prim's algorithm)
- Maximum Independent Set (randomized)
- Shortest paths (Dijkstra's algorithm)
### Step-wise learning with hints
```bash
python training.py --config_path configs/*algorithm_name*_stepwise.yaml
```

### Sequential learning with hints
```bash
python training.py --config_path configs/*algorithm_name*_sequential.yaml
```
For no-hint experiments, please set the `use_hints: false` in the corresponding config file.


### States generation
You can find states (hints) generation procedures for each algorithm in `state_algorithms.py`.

### Test examples
Also, you can directly test states transitions and attention ranges of the obtained model, as described in the Section 6 of the paper. You can find the examples of such tests in `test_state_transition.py`.
For example, for BFS model, we can explicitly check whether an undiscovered node becomes discovered if it receives a message from a discovered node:

```python
assert node_state_transition(model, sender_state=DISCOVERED, reciever_state=UNDISCOVERED) == DISCOVERED
```
We note that the hint sequence does not specify the edge along which the message will be sent, and the tests should take into account the actual dynamics of the particular model.

