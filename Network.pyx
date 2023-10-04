# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
# distutils: language = c
# cython: c_string_type=unicode, c_string_encoding=utf8
import os
import zarr as zr
import xarray as xr
import numpy as np
cimport numpy as cnp
cimport cython as cy
from cython.parallel import prange
from time import time


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True) 
@cy.profile(True)
cpdef net(str p_inpnet,
Py_ssize_t p_n,
Py_ssize_t p_edg_num):

    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] edgw
        cnp.ndarray[cnp.int32_t, ndim=1] G_deg, G_wdeg, G_cudeg, shift, G_adj, G_wadj, adj_i, wgh_i, aw
        # for memoryview
        cnp.int32_t[:, :] edgw_v
        cnp.int32_t[:] G_deg_v, G_wdeg_v, shift_v, G_cudeg_v, G_adj_v, G_wadj_v
        Py_ssize_t i=0, e=0, edg_num=0
        cnp.int32_t nd1=0, nd2=0, num=0
        str OUP_DEG = ''
        str OUP_WDEG = ''
        str OUP_CUDEG = ''
        str OUP_ADJ = ''
        str OUP_WADJ = ''
        str OUP_GRAPH = ''
        str OUP_WGRAPH = ''

    print(p_inpnet, p_n, p_edg_num)
    OUP_DEG = f'{p_inpnet}_deg.npy'
    OUP_WDEG = f'{p_inpnet}_wdeg.npy'
    OUP_CUDEG = f'{p_inpnet}_cudeg.npy'
    OUP_ADJ = f'{p_inpnet}_adj.npy'
    OUP_WADJ = f'{p_inpnet}_wadj.npy'
    OUP_GRAPH = f'{p_inpnet}_gph.graph'
    OUP_WGRAPH = f'{p_inpnet}_wgph.graph'

    # node, node, similarity, confidence
    edgw = np.load(f'{p_inpnet}_Udwp_1.npy')[:, :3].astype(np.int32) - 1
    edgw_v = edgw
    edg_num = edgw.shape[0]
    if edg_num != p_edg_num:
        print("Input edge number wrong!")
        os._exit(0)

    # output degree
    if not os.path.exists(OUP_DEG):
        G_deg = np.zeros(p_n, dtype=np.int32)
        G_wdeg = np.zeros(p_n, dtype=np.int32)
        G_deg_v = G_deg
        G_wdeg_v = G_wdeg
        for e in range(p_edg_num):
            nd1, nd2 = edgw_v[e][0], edgw_v[e][1]
            G_deg_v[nd1] += 1
            G_deg_v[nd2] += 1
            G_wdeg_v[nd1] += (edgw_v[e][2] + 1)
            G_wdeg_v[nd2] += (edgw_v[e][2] + 1)
        np.save(OUP_DEG, G_deg)
        np.save(OUP_WDEG, G_wdeg)
    else:
        G_deg = np.load(OUP_DEG).astype(np.int32)
        G_wdeg = np.load(OUP_WDEG).astype(np.int32)
        G_deg_v = G_deg
        G_wdeg_v = G_wdeg

    # output cudegree
    if not os.path.exists(OUP_CUDEG):
        G_cudeg = np.zeros(p_n + 1, dtype=np.int32)
        G_cudeg_v = G_cudeg
        num = 0
        for i in range(p_n + 1):
            G_cudeg_v[i] = num
            if i < p_n:
                num += G_deg_v[i]
        np.save(OUP_CUDEG, G_cudeg)
    else:
        G_cudeg = np.load(OUP_CUDEG).astype(np.int32)
        G_cudeg_v = G_cudeg

    # output adjacency list
    if not os.path.exists(OUP_ADJ):
        shift = G_cudeg.copy()
        shift_v = shift
        G_adj = np.zeros(np.sum(G_deg), dtype=np.int32)
        G_wadj = np.zeros(np.sum(G_deg), dtype=np.int32)
        G_adj_v = G_adj
        G_wadj_v = G_wadj
        for e in range(p_edg_num):
            nd1, nd2 = edgw_v[e][0], edgw_v[e][1]
            G_adj_v[shift_v[nd1]] = nd2
            G_adj_v[shift_v[nd2]] = nd1
            G_wadj_v[shift_v[nd1]] = (edgw_v[e][2] + 1)
            G_wadj_v[shift_v[nd2]] = (edgw_v[e][2] + 1)
            shift_v[nd1] += 1
            shift_v[nd2] += 1
        np.save(OUP_ADJ, G_adj)
        np.save(OUP_WADJ, G_wadj)
        del shift
    else:
        G_adj = np.load(OUP_ADJ).astype(np.int32)
        G_wadj = np.load(OUP_WADJ).astype(np.int32)
        G_adj_v = G_adj
        G_wadj_v = G_wadj

    # to .graph file
    del edgw
    if not os.path.exists(OUP_GRAPH):
        with open(OUP_GRAPH, 'w', newline='', encoding='UTF-8') as gph:
            gph.write(str(p_n) + ' ' + str(p_edg_num) + '\n')
        with open(OUP_GRAPH, 'a', newline='', encoding='UTF-8') as gph:
            for i in range(p_n):
                if G_deg_v[i] != 0:
                    adj_i = G_adj[G_cudeg_v[i]:G_cudeg_v[i + 1]] + 1
                    np.savetxt(gph, adj_i, fmt='%d', newline=' ')
                gph.write('\n')
    if not os.path.exists(OUP_WGRAPH):
        with open(OUP_WGRAPH, 'w', newline='', encoding='UTF-8') as gph:
            gph.write(str(p_n) + ' ' + str(p_edg_num) + ' ' + str(1) + '\n')
        with open(OUP_WGRAPH, 'a', newline='', encoding='UTF-8') as gph:
            for i in range(p_n):
                if G_deg_v[i] != 0:
                    adj_i = G_adj[G_cudeg_v[i]:G_cudeg_v[i + 1]] + 1
                    wgh_i = G_wadj[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    aw = np.vstack([adj_i, wgh_i]).T.flatten().astype(np.int32)
                    np.savetxt(gph, aw, fmt='%d', newline=' ')
                gph.write('\n')
        print(".graph file output done")