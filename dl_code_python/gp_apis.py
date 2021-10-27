import torch as th
import torch.utils.dlpack
import kernel as gpk
import datetime

def gp_gspmm(g, X, dim0, dim1, inverse, norm):
    X_dl = th.utils.dlpack.to_dlpack(X)

    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.spmm(g, X_dl, res_dl, inverse, norm)  # do not specify the reduce operation

    return res

def gp_gspmmw(g, X, dim0, dim1, op, inverse):
    #print("spmmw", X.size(0), X.size(1))
    X_dl = th.utils.dlpack.to_dlpack(X)
    #print("dim0, 1 is: ", dim0, dim1)

    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)
    g_start = datetime.datetime.now()
    gpk.gspmmw(g, X_dl, res_dl, op, inverse)
    g_end = datetime.datetime.now()
    diff = g_end - g_start
    print ("gspmmw time is:", diff)

    return res

def gp_gspmmw2d(g, X, dim0, dim1, dim2, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1, dim2)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.spmmw2d(g, X_dl, res_dl, op, inverse)

    return res

def gp_gsddmme(g, X, Y, dim0, dim1, op, inverse):
    #print("sddmme", X.size(0), X.size(1), Y.size(0), Y.size(1))

    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    #print("dim0, 1 is: ", dim0, dim1)
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)
    g_start = datetime.datetime.now()
    gpk.gsddmme(g, X_dl, Y_dl, res_dl, op, inverse)
    g_end = datetime.datetime.now()
    diff = g_end - g_start
    print ('gsddmme time is:', diff)

    return res

def gp_gsddmme2d(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gsddmme2d(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gsddmm(g, X, Y, dim0, dim1, op, inverse):
    #print("sddmm", X.size(0), X.size(1), Y.size(0), Y.size(1))

    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    #print("dim0, 1 is: ", dim0, dim1)

    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)
    g_start = datetime.datetime.now()
    gpk.gsddmm(g, X_dl, Y_dl, res_dl, op, inverse)
    g_end = datetime.datetime.now()
    diff = g_end - g_start
    print ('gsddmm time is:', diff)

    return res

def gp_gsddmm2d(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.sddmm2d(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

def gp_gspmmw_op(g, X, Y, dim0, dim1, op, inverse):
    #print("spmmw_op", X.size(0), X.size(1), Y.size(0), Y.size(1))
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    #print("dim0, 1 is: ", dim0, dim1)
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)
    g_start = datetime.datetime.now()
    gpk.gspmmw_op(g, X_dl, Y_dl, res_dl, op, inverse)
    g_end = datetime.datetime.now()
    diff = g_end - g_start
    print ('gspmmw_op time is:', diff)

    return res

def gp_gspmmw_op2d(g, X, Y, dim0, dim1, dim2, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1, dim2)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.spmmw_op2d(g, X_dl, Y_dl, res_dl, op, inverse)

    return res

##################################
def gp_spmmw_model(g, X, Y, Z, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    Z_dl = th.utils.dlpack.to_dlpack(Z)
 
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmmw_model(g, X_dl, Y_dl, Z_dl, res_dl, op, inverse)

    return res

def gp_spmmw_model_without_bias(g, X, Y, dim0, dim1, op, inverse):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    #Z_dl = th.utils.dlpack.to_dlpack(Z)
 
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmmw_model_without_bias(g, X_dl, Y_dl, res_dl, op, inverse)

    return res


def gp_sddmm_model(g, X, Y, dim0, dim1, op):
    X_dl = th.utils.dlpack.to_dlpack(X)
    Y_dl = th.utils.dlpack.to_dlpack(Y)
    
    # declare the output tensor here
    res = th.zeros(dim0, dim1)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gsddmme_model(g, X_dl, Y_dl, res_dl, op)

    return res
