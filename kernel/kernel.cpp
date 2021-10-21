#include <cassert>
#include <iostream>
#include <limits>

#include "kernel.h"

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;


void _gspmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{
    //cout << "spmm " << op << "reverse = " << reverse << endl;

    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here.    
}

void invoke_gspmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
         return _gspmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
         return _gspmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }
}
