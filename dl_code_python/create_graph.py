import pygraph as pg
import kernel
import numpy as np
import datetime

def memoryview_to_np(memview, nebr_dt):
    arr = np.array(memview, copy=False)
    a = arr.view(nebr_dt)
    return a;

def create_csr_graph_simple(ifile, num_vcount, ingestion_flag):
    num_sources = 1
    num_thread = 2

    edge_dt = np.dtype([('src', np.int32), ('dst', np.int32)])
    csr_dt = np.dtype([('dst', np.int32)])
    offset_dt = np.dtype([('offset', np.int32)])

    outdir = ""
    graph = pg.init(1, 1, outdir, num_sources, num_thread)  # Indicate one pgraph, and one vertex type
    tid0 = graph.init_vertex_type(num_vcount, True, "gtype") # initiate the vertex type
    pgraph = graph.create_schema(ingestion_flag, tid0, "friend", edge_dt) # initiate the pgraph

    # creating graph directly from file requires some efforts. Hope to fix that later
    manager = graph.get_pgraph_manager(0)
    manager.add_edges_from_dir(ifile, ingestion_flag)  # ifile has no weights
    pgraph.wait()  # You can't call add_edges() after wait(). The need of it will be removed in future.
    #manager.run_bfs(1)

    
    offset_csr1, offset_csc1, nebrs_csr1, nebrs_csc1 = pg.create_csr_view(pgraph);
    offset_csr = memoryview_to_np(offset_csr1, offset_dt);
    offset_csc = memoryview_to_np(offset_csc1, offset_dt);
    nebrs_csr = memoryview_to_np(nebrs_csr1, csr_dt);
    nebrs_csc = memoryview_to_np(nebrs_csc1, csr_dt);
    
    kernel_graph_flag = 0; #eADJ graph
    csr_graph = kernel.init_graph(offset_csr, nebrs_csr, offset_csc, nebrs_csc, kernel_graph_flag, num_vcount);

    return csr_graph;

def create_csr_graph(ifile, num_vcount, ingestion_flag):
    num_sources = 1
    num_thread = 2

    edge_dt = np.dtype([('src', np.int32), ('dst', np.int32), ('edgeid', np.int32)])
    csr_dt = np.dtype([('dst', np.int32), ('edgeid', np.int32)])
    offset_dt = np.dtype([('offset', np.int32)])

    outdir = ""
    graph = pg.init(1, 1, outdir, num_sources, num_thread)  # Indicate one pgraph, and one vertex type
    tid0 = graph.init_vertex_type(num_vcount, True, "gtype") # initiate the vertex type
    pgraph = graph.create_schema(ingestion_flag, tid0, "friend", edge_dt) # initiate the pgraph

    # creating graph directly from file requires some efforts. Hope to fix that later
    manager = graph.get_pgraph_managerW(0) # This assumes single weighted graph, edgeid is the weight
    manager.add_edges_from_dir(ifile, ingestion_flag)  # ifile has no weights, edgeid will be generated
    pgraph.wait()  # You can't call add_edges() after wait(). The need of it will be removed in future.
    #manager.run_bfs(1)

    
    offset_csr1, offset_csc1, nebrs_csr1, nebrs_csc1 = pg.create_csr_view(pgraph);
    offset_csr = memoryview_to_np(offset_csr1, offset_dt);
    offset_csc = memoryview_to_np(offset_csc1, offset_dt);
    nebrs_csr = memoryview_to_np(nebrs_csr1, csr_dt);
    nebrs_csc = memoryview_to_np(nebrs_csc1, csr_dt);
    
    kernel_graph_flag = 0; #eADJ graph
    csr_graph = kernel.init_graph(offset_csr, nebrs_csr, offset_csc, nebrs_csc, kernel_graph_flag, num_vcount);

    return csr_graph;
