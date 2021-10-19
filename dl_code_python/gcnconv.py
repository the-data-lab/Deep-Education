import torch as th
from torch import nn
from torch.nn import init

import sparse

# def get_degree_tensor(snaph):
#     num_vcount = snaph.get_vcount()
#     # print(num_vcount)
#     nebr_reader = gone.nebr_reader_t()
#     result = th.zeros(num_vcount, 1)
#     for current_vertex in range(num_vcount):
#         # print ("Visited the node:", current_vertex)
#         nebr_count = snaph.get_nebrs_out(current_vertex, nebr_reader)
#         result[current_vertex] = nebr_count
#
#     return result


# pylint: disable=W0235
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm= True,
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree= True):
        super(GraphConv, self).__init__()
        if norm not in (True, 'both', 'right'):
            raise Exception('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        num_vcount = graph.get_vcount()
        # if self._norm == 'both':
        #     # get a tensor degree for every node
        #     degree = get_degree_tensor(graph)
        #     norm = th.pow(degree, -0.5)
        #     feat_src = feat * norm
        #print(feat)
        if weight is not None:
            if self.weight is not None:
                raise Exception('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat1 = th.matmul(feat, weight)
                #print(feat1)
            dim = feat1.size(1)
            # feature means the vertex input_feature, rst is the output 1d numpy array
            rst = sparse.run_gspmm(graph, feat1, self._norm, num_vcount, dim)
            #print(rst)
        else:
            dim = feat.size(1)
            # aggregate first then mult W
            rst = sparse.run_gspmm(graph, feat, self._norm, num_vcount, dim)
            if weight is not None:
                rst = th.matmul(rst, weight)

        # if self._norm != 'none':
        #     degs = graph.in_degrees().float().clamp(min=1)
        #     if self._norm == 'both':
        #         norm = th.pow(degs, -0.5)
        #     else:
        #         norm = 1.0 / degs
        #     shp = norm.shape + (1,) * (feat_dst.dim() - 1)
        #     norm = th.reshape(norm, shp)
        #     rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)
        #print(rst)
        return rst
"""
class GCN(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats = in_feats, out_feats = n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(in_feats = n_hidden, out_feats = n_hidden))
        # output layer
        self.layers.append(GraphConv(in_feats = n_hidden, out_feats = n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            #if i != 0:
                #h = self.dropout(h)
            h = layer(self.graph, h)
            #print (h)
        return h

"""
class GCN(nn.Module):
    def __init__(self, graph, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.graph = graph
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
    def forward(self, inputs):
        h = self.conv1(self.graph, inputs)
        h = th.relu(h)
        h = self.conv2(self.graph, h)
        return h
