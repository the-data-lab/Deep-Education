import pygraph as gone
import torch as th
import gp_apis

#from gone.enumOP import enumOP

# importing enum for enumerations 
import enum

# creating enumerations using class 
#class enumOP(enum.Enum): 
#    eSUM = 0
#    eMAX = 1
#    eMIN = 2
#    eSUB = 3 
#    eMUL = 4
#    eDIV = 5

#
# __all__ = ['gspmm', 'gsddmm', 'edge_softmax']
#



class GSpmv(th.autograd.Function):
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


class GApplyEdges(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, X, Y):
        #print("apply_edge_forwards")
        dim = X.size(1)
        num_ecount = graph.get_edge_count()

        res = gp_apis.gp_gsddmme(graph,X, Y, num_ecount, dim, gone.enumOP.eSUM, 0)
        
        ctx.backward_cache = graph, dim
        return res

    @staticmethod
    def backward(ctx, dZ):
        #print("apply_edge_backward")
        graph, dim = ctx.backward_cache
        num_vcount = graph.get_vcount();
       
        resX = gp_apis.gp_gspmmw(graph, dZ, num_vcount, dim, gone.enumOP.eSUM, 1)
        resY = gp_apis.gp_gspmmw(graph, dZ, num_vcount, dim, gone.enumOP.eSUM, 0)

        return None, resX, resY


class GApplyEdge_heads(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, X, Y):
        #print("apply_edge_heads_forwards")
        heads = X.size(1)
        dim = X.size(2)
        #print("dim0 dim1", X.size(), dim)
        num_ecount = graph.get_edge_count()
        
        res = gp_apis.gp_gsddmme2d(graph, X, Y, num_ecount, heads, gone.enumOP.eSUM, 0)

        ctx.backward_cache = graph, dim, heads
        #print("res_forward", res)
        return res

    @staticmethod
    def backward(ctx, dZ):
        #print("GApplyEdge_heads_backward")
        graph, dim, heads = ctx.backward_cache
        num_vcount = graph.get_vcount();
        
        resX = gp_apis.gp_gspmmw2d(graph, dZ, num_vcount, heads, dim, gone.enumOP.eSUM, 1)
        resY = gp_apis.gp_gspmmw2d(graph, dZ, num_vcount, heads, dim, gone.enumOP.eSUM, 0)
        
        return None, resX, resY


class EdgeSoftmax(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, efficient_score, dim):
        #print("edge_softmax_forward")
        num_vcount = graph.get_vcount()
        num_ecount = graph.get_edge_count();
        #print(num_ecount, dim)
        
        # for score_max
        score_max = gp_apis.gp_gspmmw(graph, efficient_score, num_vcount, dim, gone.enumOP.eMAX, 0)
        
        # sub from score_max
        score = gp_apis.gp_gsddmm(graph, score_max, efficient_score, num_ecount, dim, gone.enumOP.eSUB, 0)
        
        # apply expo for score
        score_exp = th.exp(score)
        
        # todo score_sum
        score_sum = gp_apis.gp_gspmmw(graph, score_exp, num_vcount, dim, gone.enumOP.eSUM, 0)
        
        # todo score % score_sum.out is | E |
        out = gp_apis.gp_gsddmm(graph, score_sum, score_exp, num_ecount, dim, gone.enumOP.eDIV, 0)

        ctx.backward_cache = graph, dim, out
        return out

    @staticmethod
    def backward(ctx, dZ):
        #print("edge_softmax_backward")
        graph, dim, out = ctx.backward_cache
        sds = out * dZ

        num_vcount = graph.get_vcount();
        num_ecount = graph.get_edge_count();
        
        accum = gp_apis.gp_gspmmw(graph, sds, num_vcount, dim, gone.enumOP.eSUM, 0)
        
        temp = gp_apis.gp_gsddmm(graph, accum, out, num_ecount, dim, gone.enumOP.eMUL, 0)
        
        grad_score = sds - temp
        #print("grad_score", grad_score)
        
        return None, grad_score, None


class EdgeSoftmax_heads(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, efficient_score, dim):
        #print("EdgeSoftmax_heads_forward")
        heads = efficient_score.size(1)
        num_vcount = graph.get_vcount()
        num_ecount = graph.get_edge_count()
        feat = th.utils.dlpack.to_dlpack(efficient_score)
       
        # for score_max
        score_max = gp_apis.gp_gspmmw2d(graph, efficient_score, num_vcount, heads, 1, gone.enumOP.eMAX, 0)
        
        # sub from score_max
        score = gp_apis.gp_gsddmm2d(graph, score_max, efficient_score, num_ecount, heads, gone.enumOP.eSUB, 0)
        
        # apply expo for score
        score_exp = th.exp(score)
        
        # todo score_sum
        score_sum = gp_apis.gp_gspmmw2d(graph, score_exp, num_vcount, heads, 1, gone.enumOP.eSUM, 0)
        
        # todo score % score_sum.out is | E |
        out = gp_apis.gp_gsddmm2d(graph, score_sum, score_exp, num_ecount, heads, gone.enumOP.eDIV, 0)

        ctx.backward_cache = graph, out, heads
        #print("XXX EdgeSoftmax_heads_forward_out", out.size())
        return out

    @staticmethod
    def backward(ctx, dZ):
        #print("EdgeSoftmax_heads_backward")
        graph, out, heads = ctx.backward_cache
        sds = out * dZ

        num_vcount = graph.get_vcount()
        num_ecount = graph.get_edge_count()
        
        accum = gp_apis.gp_gspmmw2d(graph, sds, num_vcount, heads, 1, gone.enumOP.eSUM, 0)
        
        temp = gp_apis.gp_gsddmm2d(graph, accum, out, num_ecount, heads, gone.enumOP.eMUL, 0)
        
        grad_score = sds - temp
        #print("grad_score", grad_score)

        return None, grad_score, None



class GSpmv_op(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, X, edge_score_by_softmax, dim):
        # input is for each edge, edge_score_by_softmax is also refer to each edge
        #print("begin_gspmv_op_forward")
        num_vcount = graph.get_vcount()
        
        rst = gp_apis.gp_gspmmw_op(graph, X, edge_score_by_softmax, num_vcount, dim, gone.enumOP.eSUM, 0)
        
        ctx.backward_cache = graph, X, edge_score_by_softmax, dim
        return rst


    @staticmethod
    def backward(ctx, dZ):
        #print("begin_gspmv_op_backward")
        graph, X, edge_score_by_softmax, dim = ctx.backward_cache
        reverse = 1
        num_vcount = graph.get_vcount();
        num_ecount = graph.get_edge_count();
        
        res = gp_apis.gp_gspmmw_op(graph, dZ, edge_score_by_softmax, num_vcount, dim,  gone.enumOP.eSUM, reverse)
        escore = gp_apis.gp_gsddmme(graph, X, dZ, num_ecount, 1, gone.enumOP.eMUL, 0)
        
        return None, res, escore, None


class GSpmv_op_heads(th.autograd.Function):
    @staticmethod
    def forward(ctx, X, graph, edge_score_by_softmax, dim):
        #print("GSpmv_op_heads_forward")
        # input is for each edge, edge_score_by_softmax is also refer to each edge
        num_vcount = graph.get_vcount()
        heads = edge_score_by_softmax.size(1)
        
        rst = gp_apis.gp_gspmmw_op2d(graph, X, edge_score_by_softmax, num_vcount, heads, dim, gone.enumOP.eSUM, 0)
        
        ctx.backward_cache = graph, X, edge_score_by_softmax, dim, heads
        #print("GSpmv_op_heads_res_forward", rst)
        return rst


    @staticmethod
    def backward(ctx, dZ):
        #print("GSpmv_op_heads_backward")
        #print ("9999")
        graph, X, edge_score_by_softmax, dim, heads = ctx.backward_cache
        reverse = 1
        num_vcount = graph.get_vcount()
        num_ecount = graph.get_edge_count()
        
        res = gp_apis.gp_gspmmw_op2d(graph, dZ, edge_score_by_softmax, num_vcount, heads, dim,  gone.enumOP.eSUM, reverse)
        escore = gp_apis.gp_gsddmme2d(graph, X, dZ, num_ecount, heads, gone.enumOP.eMUL, 0)
        
        dZ_dl = th.utils.dlpack.to_dlpack(dZ)
        edge_score_by_softmax_dl = th.utils.dlpack.to_dlpack(edge_score_by_softmax)
        X_dl = th.utils.dlpack.to_dlpack(X)

        return res, None, escore, None



# the gspmv has only 1 input, and then apply different operations such as sum, max on it
def run_gspmm(graph, X, norm, num_vcount, dim):
    return GSpmv.apply(graph, X, norm, num_vcount, dim)


# the gspmv_op has 2 inputs, one is edge_score, another one is edge_softmax score
def run_gspmv_op(graph, X, edge_score_by_softmax, num_vcount, dim):
    return GSpmv_op.apply(graph, X, edge_score_by_softmax, dim)


def run_gspmv_op_heads(graph, X, edge_score_by_softmax, num_vcount, dim):
    return GSpmv_op_heads.apply(X, graph, edge_score_by_softmax, dim)

def apply_edge(graph, el, er):
    return GApplyEdges.apply(graph, el, er)

def apply_edge_heads(graph, el, er):
    return GApplyEdge_heads.apply(graph, el, er)



def edge_softmax(graph, efficient_score, num_vcount, dim):
    return EdgeSoftmax.apply(graph, efficient_score, dim)

def edge_softmax_heads(graph, efficient_score, num_vcount, dim):
    result = EdgeSoftmax_heads.apply(graph, efficient_score, dim)
    return result

