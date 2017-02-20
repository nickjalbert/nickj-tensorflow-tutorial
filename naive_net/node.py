import math
import random
import pickle

class Net(object):
    def __init__(self, layer_counts):
        self.layer_counts = layer_counts
        self.layers = {}
        node_id = 0
        for layer_id, node_count in enumerate(self.layer_counts):
            self.layers[layer_id] = [Node(node_id+j, layer_id, self.layers) for j in range(node_count)]
            node_id += node_count

class Node(object):
    def __init__(self, node_id, layer_id, layers):
        self.node_id = node_id
        self.layer_id = layer_id 
        self.layers = layers
        self.input_weights = None
        if self.layer_id > 0:
            self.input_weights = [.5 - random.random() for i in range(len(self.layers[self.layer_id-1]))]
        self.bias = .5 - random.random()


    def get_output(self, inputs):
        assert len(inputs) == len(self.input_weights)
        activation = sum([inputs[i]*self.input_weights[i] for i in range(len(inputs))])
        activation += self.bias
        return 1/(1 + math.e**(-1* activation))



if __name__ == "__main__":
    # Training data
    with open('500-training.pkl', 'rb') as fin:
        result = pickle.load(fin)
        print(sorted([(rid, rlabel) for rid, (rlabel, rdata) in result.items()]))

    # Basic node sanity
    node = Node(0,0,{})
    node.bias = -.5
    node.input_weights = [.2,.2]
    assert node.get_output([1000, 1000]) > .99
    assert node.get_output([-1000, -1000]) < .01
    node.bias = 0
    assert .49 < node.get_output([0, 0]) < .51

    net = Net([4, 2, 4])

    for layer_id, layer in net.layers.items():
        print('Layer {} has {} nodes'.format(layer_id, len(layer)))
        for node in layer:
            print('\tNode {} in layer {} has input weights {} and bias {}'.format(node.node_id, node.layer_id, node.input_weights, node.bias))
        print()

