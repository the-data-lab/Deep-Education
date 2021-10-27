#include <cassert>
#include <iostream>
#include <limits>
#include<thread>

#include "kernel.h"

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;


void _spmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{   
    // (v_number*v_number) conv (v_number, dim) => v_number*dim
    // if norm, l2_norm(output)
    vid_t* node_index = snaph->offset;
    vid_t* node_value = snaph->nebrs;
    for(int i=0; i<snaph->v_count; i++){
        for(int j=0; j<output.col_count; j++){
            int tmp_conv_num = 0;
            float* output_row_address = output[i];
            for(int k=node_index[i]; k<node_index[i+1]; k++){
                if(op==eSUM){
                    // cout << "True " << endl;
                    tmp_conv_num += (float)node_value[k]*input[node_value[k]][j];
                }
            }
            output[i][j] = tmp_conv_num;
        }
        if(norm){
            float degree = node_index[i+1] - node_index[i];
            output.row_normalize(i, degree);
        }
    }
    // cout << "output col number " << output.col_count << endl;

    // cout << "spmm " << op << "reverse = " << reverse << endl;

    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here.    
}

void invoke_spmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
         return _spmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
         return _spmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }
}
