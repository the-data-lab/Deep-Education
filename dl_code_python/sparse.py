import torch as th
import gp_apis


class GSpmm(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, X, norm, num_vcount, dim):
        res = gp_apis.gp_gspmm(graph, X, num_vcount, dim, 0, norm)  # do not specify the reduce operation
        ctx.backward_cache = graph, norm, num_vcount, dim
        return res

    @staticmethod
    def backward(ctx, dZ):
        graph, norm, num_vcount, dim = ctx.backward_cache
        res = gp_apis.gp_gspmm(graph, dZ, num_vcount, dim, 1, norm)  # do not specify the reduce operation
        return None, res, None, None, None

# the gspmv has only 1 input, and then apply different operations such as sum, max on it
def run_gspmm(graph, X, norm, num_vcount, dim):
    return GSpmm.apply(graph, X, norm, num_vcount, dim)
