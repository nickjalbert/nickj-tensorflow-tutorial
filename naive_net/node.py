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


def sanity_checks():
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
        if layer_id in [0,2]:
            assert len(layer) == 4
        else:
            assert len(layer) == 2
        for node in layer:
            if node.layer_id == 0:
                assert node.node_id in [0,1,2,3]
                assert node.input_weights is None
                assert node.bias is not None
            elif node.layer_id == 1:
                assert node.node_id in [4,5]
                assert len(node.input_weights) == 4
                assert node.bias is not None
            else:
                assert node.node_id in [6,7,8,9]
                assert len(node.input_weights) == 2
                assert node.bias is not None

if __name__ == "__main__":
    # Training data
    with open('500-training.pkl', 'rb') as fin:
        training_data = pickle.load(fin)
    assert len(training_data) == 500

    sanity_checks()

    net = Net([784, 15, 10])



