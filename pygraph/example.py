import queue
import numpy as np
import collections
import pygraph as pg

import datetime

def memoryview_to_np(memview, nebr_dt):
    arr = np.array(memview, copy=False)
    #a = arr.view(nebr_dt).reshape(nebr_reader.get_degree())
    a = arr.view(nebr_dt)
    return a;

#simple plain CSR 
def test_csr():    
    outdir = ""
    num_sources = 1 
    num_threads = 2 
    graph  = pg.init(1,1, outdir, num_sources, num_threads) # Indicate one pgraph, and one vertex type
    
    tid0 = graph.init_vertex_type(1000, True, "gtype"); # initiate the vertex type
    edge_dt = np.dtype([('src', np.int32), ('dst', np.int32)])
    flags = pg.enumGraph.eUdir;
    pgraph = graph.create_schema(flags, tid0, "friend", edge_dt); #initiate the pgraph
    
    ifile = "smallworld.txt";
    dd = np.zeros(1024, edge_dt); 
    edge_count = 0;
    with open(ifile) as f:
        for line in f: # read rest of lines
            x = line.split();
            #print(x);
            if x[0] != "#": 
                dd[edge_count] = (x[0], x[1]);
                edge_count += 1;
                if (edge_count == 1024):
                    pgraph.add_edges(dd, 1024); # You can call this API many times, if needed
                    edge_count = 0;

    #print(edge_count);
    pgraph.add_edges(dd, edge_count); # You can call this API many times, if needed
    pgraph.wait(); # required for the time-being. You cannot add edges after this API.
   

    offset_csr1, offset_csc1, nebrs_csr1, nebrs_csc1 = pg.create_csr_view(pgraph);
    offset_dt = np.dtype([('offset', np.int32)])
    csr_dt =  np.dtype([('dst', np.int32)])

    offset_csr = memoryview_to_np(offset_csr1, offset_dt);
    offset_csc = memoryview_to_np(offset_csc1, offset_dt);
    nebrs_csr  = memoryview_to_np(nebrs_csr1, csr_dt);
    nebrs_csc  = memoryview_to_np(nebrs_csc1, csr_dt);

    print(offset_csr.tolist());
    print(nebrs_csr.tolist());

def test_lanl_graph_python():
    
    edge_dt = np.dtype([('src', np.int32), ('dst', np.int32), 
                       ('time', np.int32),
                       ('duration', np.int32),
                       ('protocol', np.int32),
                       ('src_port', np.int16),
                       ('dst_port', np.int16),
                       ('src_Packets', np.int32),
                       ('dst_packets', np.int32),
                       ('src_Bytes', np.int32),
                       ('dst_Bytes', np.int32)
                      ])
    nebr_dt = np.dtype([('dst', np.int32), 
                       ('time', np.int32),
                       ('duration', np.int32),
                       ('protocol', np.int32),
                       ('src_port', np.int16),
                       ('dst_port', np.int16),
                       ('src_Packets', np.int32),
                       ('dst_packets', np.int32),
                       ('src_Bytes', np.int32),
                       ('dst_Bytes', np.int32)
    ])
    
    flags = pg.enumGraph.eUdir;
    outdir = ""
    graph  = pg.init(1,1, outdir, 1, 4) # Indicate one pgraph, and one vertex type
    tid0 = graph.init_vertex_type(31142, False, "gtype"); # initiate the vertex type
    pgraph = graph.create_schema(flags, tid0, "friend", edge_dt); #initiate the pgraph
   
    ifile = "/home/datalab/data/lanl17/nf_day-02.csv"; # test.csv;
    
    dd = np.zeros(10000, edge_dt); 
    edge_count = 0;
    start = datetime.datetime.now()
    with open(ifile) as f:
        for line in f: # read rest of lines
            x = line.split(",");
            #print(x);
            if x[0] != "#":
                src_id = graph.add_vertex(x[2], tid0);
                dst_id = graph.add_vertex(x[3], tid0);
                #print(x[2], x[3], src_id, dst_id)
                dd[edge_count] = (src_id, dst_id, x[0], x[1], x[4], x[5], x[6], x[7], x[8], x[9], x[10]);
                edge_count += 1;
            if(edge_count == 10000):
                pgraph.add_edges(dd, edge_count);
                edge_count = 0
    pgraph.add_edges(dd, edge_count);
    
    pgraph.wait(); # You can't call add_edges() after wait(). The need of it will be removed in future.
    end = datetime.datetime.now()
    diff = end - start;
    print("graph creation time = ", diff)
    

if __name__=="__main__":
    test_csr();
    #test_lanl_graph_python();
