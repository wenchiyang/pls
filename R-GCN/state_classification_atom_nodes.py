import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLError
from dgl.data import DGLDataset
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.utils.data import DataLoader
from sampling_traces import unpickle_from_file, coord_to_node_id, node_id_to_coord, sample
from problog.logic import Term, Constant
import dgl.nn.pytorch as dglnn

LAYOUT = "testGrid" # We can change the layout by changing relenvs_pip/relvens/envs/pacman/layouts/testGrid.lay
NUM_SAMPLE_TRACES = 100
trace_file = LAYOUT+"_traces_"+str(NUM_SAMPLE_TRACES)+".p"
MAX_ARITY = 2 # FIXME: do not hard code this. This is the max arity of state predicates

# We use the standard message and reduce functions
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

###############################################################################
# We define a R-GCN layer, this layer will be used to build the graph classifier network
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv_input = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')

        # self.conv_hidden = dglnn.HeteroGraphConv({
        #     rel: dglnn.GraphConv(hid_feats, hid_feats)
        #     for rel in rel_names}, aggregate='sum')

        self.conv_output = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs, **kwargs):
        # inputs is features of nodes
        h = self.conv_input(graph, inputs, **kwargs)

        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv_output(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn1 = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        # We need to manually port the edge features to the rgcn layer
        etypes = [key[1] for key in g.edata['arg_pos'].keys()]
        edge_features = {
            etype: {
                'edge_weight': g.edges[etype].data['arg_pos']
            } for etype in etypes}
        kwargs = {'mod_kwargs': edge_features}
        # call the layer
        # h = self.rgcn1(g, h, **kwargs)
        h = self.rgcn1(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)




# The Pacman dataset: It converts traces to a list of (state, label) examples
class PacmanDataset(DGLDataset):
    def __init__(self, trace_file, width, n_classes):
        self.trace_file = trace_file
        self.width = width # for calculating links between connections
        self.n_classes = n_classes
        super().__init__(name='pacman')

    def process(self):
        self.graphs = []
        self.labels = []

        traces = unpickle_from_file(self.trace_file)
        for trace in traces:
            for state, action, reward in trace:
                # get all atom types
                atom_types = list(set([atom.functor for atom in state]))
                # create a name dict, an edge dict, an edge feature dict
                name_node_dict = {'constant': {}}
                node_count = {'constant': 0}
                edges = {}
                edge_features = {}
                for atom_type in atom_types:
                    name_node_dict[atom_type] = {}
                    node_count[atom_type] = 0
                    edges[(atom_type, atom_type+'-to-constant', 'constant')] = []
                    edges[('constant', 'constant-to-'+atom_type, atom_type)] = []
                    edge_features[(atom_type, atom_type+'-to-constant', 'constant')] = []
                    edge_features[('constant', 'constant-to-'+atom_type, atom_type)] = []

                for atom in state:
                    if atom.functor == 'ghost':
                        atom = Term('ghost', atom.args[1])
                    atom_type = atom.functor
                    constant_names = atom.args

                    # register the atom node to name_node_dict if not exists
                    if atom not in name_node_dict[atom_type]:
                        name_node_dict[atom_type][atom] = node_count[atom_type]
                        node_count[atom_type] += 1

                    for index, constant_name in enumerate(constant_names):
                        # register the constant node if not exists
                        if constant_name not in name_node_dict['constant']:
                            name_node_dict['constant'][constant_name] = node_count['constant']
                            node_count['constant'] += 1
                        # add the atom-to-constant and constant-to-atom edges
                        edges[(atom_type, atom_type+'-to-constant', 'constant')].append(
                            (name_node_dict[atom_type][atom], name_node_dict['constant'][constant_name]))
                        edges[('constant', 'constant-to-'+atom_type, atom_type)].append(
                            (name_node_dict['constant'][constant_name], name_node_dict[atom_type][atom]))
                        # add the atom-to-constant and constant-to-atom features indicating the index of the objects
                        edge_features[(atom_type, atom_type+'-to-constant', 'constant')].append(
                            np.eye(MAX_ARITY)[index])
                        edge_features[('constant', 'constant-to-'+atom_type, atom_type)].append(
                            np.eye(MAX_ARITY)[index])

                # create the graph
                graph = dgl.heterograph(edges)
                # add edge features and node features
                for atom_type in atom_types:
                    graph.edges[atom_type+'-to-constant'].data['arg_pos'] = \
                        th.tensor(edge_features[(atom_type, atom_type+'-to-constant', 'constant')], dtype=th.float32)
                    graph.edges['constant-to-'+atom_type].data['arg_pos'] = \
                        th.tensor(edge_features[('constant', 'constant-to-'+atom_type, atom_type)], dtype=th.float32)
                    graph.nodes[atom_type].data['feat'] = \
                        th.ones((graph.number_of_nodes(atom_type), MAX_ARITY), dtype=th.float32)
                graph.nodes['constant'].data['feat'] = \
                    th.ones((graph.number_of_nodes('constant'), MAX_ARITY), dtype=th.float32)


                # create the graph label
                temp_label = self.safe_label_of_state(state)
                # convert the labels for a binary classification task
                if self.n_classes == 2:
                    label = 0 if temp_label == 0 else 1
                elif self.n_classes == 4:
                    label = temp_label

                # save the graph and the graph label
                self.graphs.append(graph)
                self.labels.append(th.tensor(label))

                if label == 0:
                    # Increase weight of dangerous states by adding more times
                    self.graphs.append(graph)
                    self.labels.append(th.tensor(label))
                    self.graphs.append(graph)
                    self.labels.append(th.tensor(label))


                ################ Draw #####################
                # We prepare name tags of node and save in node_name_dict {node_id: node_name}
                node_name_dict = {type: {} for type in name_node_dict}
                # reverse keys and values of name_node_dict
                for type in name_node_dict:
                    node_name_dict[type] = {v: k for k, v in name_node_dict[type].items()}
                    if type == 'constant':
                        # translate constants from location ids to coordinates
                        node_name_dict['constant'] = {k: node_id_to_coord(rel_width, v) for k, v in
                                                      node_name_dict['constant'].items()}
                    else:
                        # translate atom locations to coordinates
                        node_name_dict[type] = {k: Term(v.functor, *[node_id_to_coord(rel_width, arg) for arg in v.args]) for k, v in node_name_dict[type].items() }

                # self.draw(graph, node_name_dict, atom_types, label)



    def safe_label_of_state(self, state):
        """
        Return the label of the given state.

        label=3 iff dist(pacman, ghost) >= 3
        label=2 iff dist(pacman, ghost) ==  2
        label=1 iff dist(pacman, ghost) ==  1
        label=0 iff dist(pacman, ghost) ==  0
        """
        # Extract the location node
        for atom in state:
            if atom.functor == 'pacman':
                pacman_loc_i, pacman_loc_j = node_id_to_coord(self.width, atom.args[0])
            elif atom.functor == 'ghost':
                ghost_loc_i, ghost_loc_j = node_id_to_coord(self.width, atom.args[1])
        euclidean_dist = abs(pacman_loc_i - ghost_loc_i) + abs(pacman_loc_j - ghost_loc_j)

        if euclidean_dist == 0:
            return 0
        elif euclidean_dist == 1:
            return 1
        elif euclidean_dist == 2:
            if pacman_loc_i == ghost_loc_i:
                middle_j = (pacman_loc_j + ghost_loc_j) / 2
                middle_node_id = coord_to_node_id(self.width, pacman_loc_i, middle_j)
                if Term('wall', Constant(middle_node_id)) in state:
                    return 3
                else:
                    return 2
            elif pacman_loc_j == ghost_loc_j:
                middle_i = (pacman_loc_i + ghost_loc_i) / 2
                middle_node_id = coord_to_node_id(self.width, middle_i, pacman_loc_j)
                if Term('wall', Constant(middle_node_id)) in state:
                    return 3
                else:
                    return 2
            else:
                middle_node_id_1 = coord_to_node_id(self.width, pacman_loc_i, ghost_loc_j)
                middle_node_id_2 = coord_to_node_id(self.width, ghost_loc_i, pacman_loc_j)
                if Term('wall', Constant(middle_node_id_1)) in state and Term('wall', Constant(middle_node_id_2)) in state:
                    return 3
                else:
                    return 2
        else:
            return 3

    def draw(self, graph, node_name_dict, atom_types, graph_label):
        # We split a relational graph into homogeneous graphs
        homo_graphs_n, rel_graphs, homo_graphs = to_homo_networkx(graph)

        fig, ax = plt.subplots(5, 2, figsize=(18, 80))
        fig.suptitle(f"Class: {graph_label}", fontsize=28)
        for homo_graph_n, rel_graph, homo_graph in zip(homo_graphs_n, rel_graphs, homo_graphs):
            (srctype, edgetype, dsttype) = rel_graph.canonical_etypes[0]
            type = srctype if srctype != 'constant' else dsttype
            i = atom_types.index(type)
            j = 0 if edgetype.startswith("constant") else 1
            # combine two sets of nodes: srctype oand dsttype
            local_node_name_mapping = {**node_name_dict[srctype],
                                       **{k + len(node_name_dict[srctype]): v
                                          for k, v in node_name_dict[dsttype].items()}}
            # draw the bipartite
            pos = nx.drawing.layout.bipartite_layout(homo_graph_n, node_name_dict[srctype], aspect_ratio=0.3)
            nx.draw_networkx(homo_graph_n, pos=pos, labels=local_node_name_mapping, ax=ax[i][j], node_color="#cee4f2")
            ax[i][j].set_title(f'Edge: {rel_graph.canonical_etypes[0]}')
        plt.show()

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def to_homo_networkx(g):
    homo_graphs_n = []
    rel_graphs = []
    homo_graphs = []
    for stype, etype, dtype in g.canonical_etypes:
        rel_graph = g[stype, etype, dtype]
        homo_graph = dgl.to_homogeneous(rel_graph)
        homo_graph_n = homo_graph.to_networkx()

        homo_graphs_n.append(homo_graph_n)
        rel_graphs.append(rel_graph)
        homo_graphs.append(homo_graph)
    return homo_graphs_n, rel_graphs, homo_graphs

###############################################################################
# When a model is trained, we can use the following method to evaluate the performance of the model
def evaluate(model, g, labels):
    model.eval()
    with th.no_grad():
        logits = model(g)
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        fp = th.sum((indices == True) & (labels == False))
        tp = th.sum((indices == True) & (labels == True))
        # tn = th.sum((indices == False) & (labels == False))
        fn = th.sum((indices == False) & (labels == True))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = 2 * (precision * recall) / (precision + recall)

        return correct.item() * 1.0 / len(labels), f1, precision, recall

# This is to sample mini batches
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.tensor(labels)

###############################################################################
# We then train the network as follows:

# Randomly sample some traces of the pacman domain
# You can comment out this line to always use the same traces
# trace_file, rel_width = sample(layout=LAYOUT, sampling_episodes=NUM_SAMPLE_TRACES)
rel_width = 5
# Some hyper parameters
in_dim = MAX_ARITY
hidden_dim = 8
n_classes = 2 # binary classification
learning_rate = 1e-2


# Create dataset
trainset = PacmanDataset(trace_file, rel_width, n_classes)
batch_size = len(trainset)

# Use PyTorch's DataLoader and the collate function defined before.
data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)



atom_types = ['wall', 'pacman', 'ghost', 'link', 'food']
rel_names = []
for atom_type in atom_types:
    rel_names.append(atom_type+'-to-constant')
    rel_names.append('constant-to-'+atom_type)

# Create graph classifier and train
graph_classifier = HeteroClassifier(in_dim, hidden_dim, n_classes, rel_names)
graph_loss_func = nn.CrossEntropyLoss()
graph_optimizer = th.optim.Adam(graph_classifier.parameters(), lr=learning_rate)
dur = []
for epoch in range(50):
    t0 = time.time()
    for (iter, (g, labels)) in enumerate(data_loader):
        graph_classifier.train()
        graph_logits = graph_classifier(g)
        graph_logp = F.log_softmax(graph_logits, 1)
        graph_loss = graph_loss_func(graph_logp, labels.long())

        graph_optimizer.zero_grad()
        graph_loss.backward()
        graph_optimizer.step()

    dur.append(time.time() - t0)

    # evaluate
    graph_acc, graph_f1, graph_precision, graph_recall = evaluate(graph_classifier, g, labels)
    print(f"Epoch {epoch:05d} | "
          f"Loss {graph_loss.item():.4f} | "
          f"Test Acc {graph_acc:.4f} | "
          f"Test F1 {graph_f1:.4f} | "
          f"Test Precision {graph_precision:.4f} | "
          f"Test Recall {graph_recall:.4f} | "
          f"Time(s) {np.mean(dur):.4f} | "
          # f"Predected Labels {indices}"
          )

print("done")