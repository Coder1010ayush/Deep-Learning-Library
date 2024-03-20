""""

    this file contains all the function and implementatons are that
    can be reused again during implementation of neuaral network architectures
"""
import os
import graphlib
from graphviz import Digraph
def visualize_graph(self, filename='computation_graph'):
        dot = Digraph()
        
        visited = set()

        # Define a recursive function to traverse the graph and add nodes and edges
        def add_node(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.sign == '':
                    sign = "leaf node"
                else:
                    sign = {tensor.sign}
                dot.node(str(tensor.id), f'{tensor.data}\ngradient: {tensor.grad}\nOperation: {sign}')
                for child in tensor.children:
                    dot.edge(str(tensor.id), str(child.id))
                    add_node(child)
        
        add_node(self)

        dot.render(filename, format='png', cleanup=True)