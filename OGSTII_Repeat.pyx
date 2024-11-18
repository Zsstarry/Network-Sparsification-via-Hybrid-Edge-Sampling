# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
# distutils: language = c
# cython: c_string_type=unicode, c_string_encoding=utf8
import os
import zarr as zr
import xarray as xr
import numpy as np
import networkit as nk
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
cpdef void sparsification(
str p_inpnet,
Py_ssize_t p_n,
Py_ssize_t p_edg_num,
cnp.float64_t p_scaling,
cnp.float64_t p_tolerance,
str p_method,
str p_oupnet,
Py_ssize_t p_repeat,
str p_suffix):

    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] e23w, PFM
        cnp.ndarray[cnp.float32_t, ndim=1] e2c, e3c, ewe, G2c, G3c, Gwe, pfm_g2c, pfm_g3c, pfm_gwe, g2c, wg2c, g3c, wg3c, gwe, wgwe, rpt, G_adjp
        cnp.ndarray[cnp.int32_t, ndim=1] G_deg, G_wdeg, G_cudeg, G_adj, G_wadj, G2c1, G3c1, Gwe1, aw, mk3, mk3h
        cnp.ndarray[cnp.int8_t, ndim=1] mkG, L, L_new
        # for memoryview
        cnp.float32_t[:] g2c_v, wg2c_v, g3c_v, wg3c_v, gwe_v, wgwe_v, e2c_v, e3c_v, ewe_v, G2c_v, G3c_v, Gwe_v, g_lcc_v, g_lcc_t_v, G_adjp_v, Gnnip_v
        cnp.int32_t[:] G_deg_v, G_wdeg_v, G_cudeg_v, G_adj_v, G_wadj_v, Gnni_v, Gnnj_v, nij_v, Gni_v, G_ni_v, Seq_n_v, Gwnni_v, Gwnnj_v, tri_v, mk3_v, mk3h_v
        cnp.int8_t[:] g_adj_v, mkG_v, gnni_v, gnnj_v, L_v, L_new_v, g_scc_v, g_scc_t_v
        Py_ssize_t i=0, j=0, ni=0, nj=0, n_gnnij=0, ij=0, r=0, nn=0, mk=0, mk_ni=0, mk_nj=0, e=0, i_seq=0
        cnp.float64_t gain=0, h=0, g2c_pre=0, g3c_pre=0, gwe_pre=0, g2c_new=0, g3c_new=0, gwe_new=0, cnt_e=0, nnwe_new=0, iwe_new=0, jwe_new=0, g_gcc_init=0, G_gcc=0, g_gcc_t=0, g_gcc=0, g_scc_init=0, G_scc=0, g_scc_t=0, g_scc=0, p=0
        cnp.float32_t flag=0, i2c_new=0, j2c_new=0, i3c_new=0, j3c_new=0, wi2c_new=0, wj2c_new=0, wi3c_new=0, wj3c_new=0
        str INP_DEG = ''
        str INP_WDEG = ''
        str INP_CUDEG = ''
        str INP_ADJ = ''
        str INP_WADJ = ''
        str INP_E23W = ''
        str OUP_GRAPH = ''
        str OUP_RT = ''


    print(p_inpnet, p_n, p_edg_num, p_scaling, p_method, p_suffix)
    INP_DEG = f'{p_inpnet}_deg.npy'
    INP_WDEG = f'{p_inpnet}_wdeg.npy'
    INP_CUDEG = f'{p_inpnet}_cudeg.npy'
    INP_ADJ = f'{p_inpnet}_adj.npy'
    INP_WADJ = f'{p_inpnet}_wadj.npy'
    if p_suffix != 'No':
        INP_E23W = f'{p_inpnet}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_e23w_G23w_CyO_{p_suffix}.npy'
        OUP_RT = f'{p_inpnet}_rt{p_repeat}_StageII_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
    else:
        INP_E23W = f'{p_inpnet}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_e23w_G23w_CyO.npy'
        OUP_RT = f'{p_inpnet}_rt{p_repeat}_StageII_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'


    e23w = np.load(INP_E23W).astype(np.float32)
    e2c = e23w[:, 0]
    e3c = e23w[:, 1]
    ewe = e23w[:, 2]
    G2c = e23w[:, 3]
    G3c = e23w[:, 4]
    Gwe = e23w[:, 5]
    e2c_v = e2c
    e3c_v = e3c
    ewe_v = ewe
    G2c_v = G2c
    G3c_v = G3c
    Gwe_v = Gwe
    G2c1 = np.where(G2c != 0)[0].astype(np.int32)
    G3c1 = np.where(G3c != 0)[0].astype(np.int32)
    Gwe1 = np.where(Gwe != 0)[0].astype(np.int32)
    del e23w


    G_deg = np.load(INP_DEG).astype(np.int32)
    G_wdeg = np.load(INP_WDEG).astype(np.int32)
    G_cudeg = np.load(INP_CUDEG).astype(np.int32)
    G_adj = np.load(INP_ADJ).astype(np.int32)
    G_wadj = np.load(INP_WADJ).astype(np.int32)
    G_deg_v = G_deg
    G_wdeg_v = G_wdeg
    G_cudeg_v = G_cudeg
    G_adj_v = G_adj
    G_wadj_v = G_wadj
    # Seq_n_v = np.arange(p_n).astype(np.int32)
    Seq_n_v = np.argsort(G_deg).astype(np.int32)


    rpt = np.zeros(p_repeat, dtype=np.float32)
    for rp in range(p_repeat):
        if p_suffix != 'No':
            OUP_GRAPH = f'{p_oupnet}_wgph_{str(rp)}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.graph'
        else:
            OUP_GRAPH = f'{p_oupnet}_wgph_{str(rp)}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.graph'


        mk3 = np.zeros(p_n, dtype=np.int32) - 1
        mk3h = np.zeros(p_n, dtype=np.int32) - 1
        mkG = np.zeros(G_adj.shape[0], dtype=np.int8)
        mk3_v = mk3
        mk3h_v = mk3h
        mkG_v = mkG
        g2c = np.zeros(p_n, dtype=np.float32)
        wg2c = np.zeros(p_n, dtype=np.float32)
        g3c = np.zeros(p_n, dtype=np.float32)
        wg3c = np.zeros(p_n, dtype=np.float32)
        gwe = np.zeros(p_n, dtype=np.float32)
        wgwe = np.zeros(p_n, dtype=np.float32)
        g2c_v = g2c
        wg2c_v = wg2c
        g3c_v = g3c
        wg3c_v = wg3c
        gwe_v = gwe
        wgwe_v = wgwe
        g_adj = np.zeros(G_adj.shape[0], dtype=np.int8)
        g_adj_v = g_adj
        G_adjp = np.zeros(G_adj.shape[0], dtype=np.float32)
        G_adjp_v = G_adjp
        g_lcc_v = np.zeros(p_n, dtype=np.float32)
        g_lcc_t_v = np.zeros(p_n, dtype=np.float32)
        g_scc_v = np.zeros(p_n, dtype=np.int8)
        g_scc_t_v = np.zeros(p_n, dtype=np.int8)


        # step 1: initialization
        t1 = time()
        if p_suffix == 'No':
            g_gcc_init, G_gcc = initialization_using_the_original_graph(p_n, g_adj_v, g2c_v, g3c_v, gwe_v, G2c_v, G3c_v, Gwe_v)
        elif p_suffix in ['No_RESInit_SG', 'No_RRESInit_SG']:
            g_scc_init, G_scc, g_gcc_init, G_gcc = initialization_using_LD_RE_LRE_SG(p_inpnet, p_n, p_scaling, p_suffix, g_adj_v, g2c_v, g3c_v, gwe_v, g_scc_v, g_scc_t_v, g_lcc_v, g_lcc_t_v, G2c_v, G3c_v, Gwe_v, G_deg_v, G_cudeg_v, G_adj_v)
            g_scc = g_scc_init
            g_scc_t = g_scc_init
            g_gcc = g_gcc_init
            g_gcc_t = g_gcc_init
        elif p_suffix in ['No_RESInit_SWG', 'No_RRESInit_SWG']:
            g_scc_init, G_scc, g_gcc_init, G_gcc = initialization_using_LD_RE_LRE_SWG(p_inpnet, p_n, p_scaling, p_suffix, g_adj_v, g2c_v, wg2c_v, g3c_v, wg3c_v, gwe_v, wgwe_v, g_scc_v, g_scc_t_v, g_lcc_v, g_lcc_t_v, G2c_v, G3c_v, Gwe_v, G_deg_v, G_wdeg_v, G_cudeg_v, G_adj_v, G_wadj_v)
            g_scc = g_scc_init
            g_scc_t = g_scc_init
            g_gcc = g_gcc_init
            g_gcc_t = g_gcc_init
        elif p_suffix in ['No_RESInitP', 'No_RRESInitP']:
            initialization_using_LD_RE_LRE_P(p_inpnet, p_n, p_edg_num, p_scaling, p_suffix, g_adj_v, g2c_v, g3c_v, gwe_v, G_adjp_v, G2c_v, G3c_v, Gwe_v, G_deg_v, G_cudeg_v, G_adj_v)
        elif p_suffix in ['No_LDInit', 'No_RESInit', 'No_RRESInit', 'No_RESInit_GCC', 'No_RRESInit_GCC', 'No_RESInit_WGCC', 'No_RRESInit_WGCC', 'GDCC_RESInit', 'GDCC_RRESInit']:
            g_gcc_init, G_gcc = initialization_using_LD_RE_LRE_GCC(p_inpnet, p_n, p_scaling, p_suffix, g_adj_v, g2c_v, g3c_v, gwe_v, g_lcc_v, g_lcc_t_v, G2c_v, G3c_v, Gwe_v, G_deg_v, G_cudeg_v, G_adj_v)
            g_gcc = g_gcc_init
            g_gcc_t = g_gcc_init
        elif p_suffix in ['No_RESInit_SCC', 'No_RRESInit_SCC']:
            g_scc_init, G_scc = initialization_using_LD_RE_LRE_SCC(p_inpnet, p_n, p_scaling, p_suffix, g_adj_v, g2c_v, g3c_v, gwe_v, g_scc_v, g_scc_t_v, G_deg_v, G_cudeg_v, G_adj_v)
            g_scc = g_scc_init
            g_scc_t = g_scc_init
        t2 = time() - t1


        # Step 2: update g
        t3 = time()
        nij_v = np.zeros(np.max(G_deg), dtype=np.int32)
        tri_v = np.zeros(p_n, dtype=np.int32)
        L_new = np.ones(p_n, dtype=np.int8)
        L = np.ones(p_n, dtype=np.int8)
        L_new_v = L_new
        L_v = L
        PFM = np.zeros((6, p_n), dtype=np.float32)
        pfm_g2c = np.zeros(p_n, dtype=np.float32)
        pfm_g3c = np.zeros(p_n, dtype=np.float32)
        pfm_gwe = np.zeros(p_n, dtype=np.float32)
        r = 0


        # pfm_g2c[G2c1] = np.absolute(g2c - e2c)[G2c1] / G2c[G2c1]
        # pfm_g3c[G3c1] = np.absolute(g3c - e3c)[G3c1] / G3c[G3c1]
        # print('C23 Init:', np.mean(pfm_g2c + pfm_g3c))


        while True:  # termination condition
            r += 1
            for i in range(p_n):
                L_v[i] = L_new_v[i]
                L_new_v[i] = 0
            for e in range(p_edg_num * 2): # 2* p_edg_num or mkG_v.shape[0]
                mkG_v[e] = 1
            cnt_e = 0
            for i_seq in range(p_n):
                i = Seq_n_v[i_seq]
                if L_v[i] == 1 and G_deg_v[i] != 0:
                    for ni in range(G_deg_v[i]):      
                        gnni_v = g_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                        Gnni_v = G_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                        Gnnip_v = G_adjp_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                        Gwnni_v = G_wadj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]] #
                        if ni == 0:  # reduce computational cost
                            # mk3[Gnni[gnnip == 1]] = 1
                            # for mk in range(G_deg_v[i]): #
                            #     if gnni_v[mk] == 1: #
                            #         mk3_v[Gnni_v[mk]] = 1 #
                            idx = 0 #
                            for mk in range(G_deg_v[i]): #
                                mk3h_v[Gnni_v[mk]] = idx #
                                if gnni_v[mk] == 1: #
                                    mk3_v[Gnni_v[mk]] = idx #
                                idx += 1 #

                        # to make sure each round, iterating each edge only once
                        if mkG_v[G_cudeg_v[i] + ni] == 1:
                            mk_ni = ni
                            j = Gnni_v[ni]
                            gnnj_v = g_adj_v[G_cudeg_v[j]:G_cudeg_v[j + 1]]
                            Gnnj_v = G_adj_v[G_cudeg_v[j]:G_cudeg_v[j + 1]]
                            Gwnnj_v = G_wadj_v[G_cudeg_v[j]:G_cudeg_v[j + 1]] #

                            mkG_v[G_cudeg_v[i] + ni] = 0
                            for nj in range(G_deg_v[j]):
                                if Gnnj_v[nj] == i:
                                    mk_nj = nj
                                    mkG_v[G_cudeg_v[j] + nj] = 0
                                    break

                            # n_gnnij = 0 #
                            # for nj in range(G_deg_v[j]): #
                            #     if gnnj_v[nj] == 1 and mk3_v[Gnnj_v[nj]] == 1: #
                            #         nij_v[n_gnnij] = Gnnj_v[nj] #
                            #         n_gnnij += 1 #
                            n_gnnij = 0 #
                            for nj in range(G_deg_v[j]): #
                                if gnnj_v[nj] == 1 and mk3_v[Gnnj_v[nj]] != -1: #
                                    nij_v[n_gnnij] = Gnnj_v[nj] #
                                    tri_v[i] += (Gwnni_v[ni] + Gwnni_v[mk3_v[Gnnj_v[nj]]]) #
                                    tri_v[j] += (Gwnni_v[ni] + Gwnnj_v[nj]) #
                                    tri_v[Gnnj_v[nj]] += (Gwnni_v[mk3_v[Gnnj_v[nj]]] + Gwnnj_v[nj]) #
                                    n_gnnij += 1 #
                        else:
                            continue
                        
                        # gain of e by retaining its strategy ===============
                        g2c_pre = 0
                        g3c_pre = 0
                        gwe_pre = 0
                        g2c_new = 0
                        g3c_new = 0
                        gwe_new = 0
                        h = 0
                        # 2-cliques
                        h = abs((g2c_v[i] - e2c_v[i]) / G2c_v[i])
                        g2c_pre += h
                        h = abs((g2c_v[j] - e2c_v[j]) / G2c_v[j])
                        g2c_pre += h
                        # 3-cliques
                        if G3c_v[i] != 0:
                            h = abs((g3c_v[i] - e3c_v[i]) / G3c_v[i])
                            g3c_pre += h
                        if G3c_v[j] != 0:
                            h = abs((g3c_v[j] - e3c_v[j]) / G3c_v[j])
                            g3c_pre += h
                        for ij in range(n_gnnij):
                            nn = nij_v[ij]
                            if G3c_v[nn] != 0:
                                h = abs((g3c_v[nn] - e3c_v[nn]) / G3c_v[nn])
                                g3c_pre += h
                        # wedges
                        if Gwe_v[i] != 0:
                            h = abs((gwe_v[i] - ewe_v[i]) / Gwe_v[i])
                            gwe_pre += h
                        if Gwe_v[j] != 0:
                            h = abs((gwe_v[j] - ewe_v[j]) / Gwe_v[j])
                            gwe_pre += h
                        for ij in range(n_gnnij):
                            nn = nij_v[ij]
                            if Gwe_v[nn] != 0:
                                h = abs((gwe_v[nn] - ewe_v[nn]) / Gwe_v[nn])
                                gwe_pre += h

                        # gain of e by changing its strategy ================
                        # 2-cliques
                        if gnni_v[mk_ni] == 1:  # e in current g
                            flag = -1  # to remove
                        elif gnni_v[mk_ni] == 0:
                            flag = 1  # to insert
                        else:
                            print('gnni_v error!!!!')
                        # 2-cliques
                        i2c_new = g2c_v[i] + flag
                        j2c_new = g2c_v[j] + flag
                        h = abs((i2c_new- e2c_v[i]) / G2c_v[i])
                        g2c_new += h
                        h = abs((j2c_new - e2c_v[j]) / G2c_v[j])
                        g2c_new += h
                        # 3-cliques
                        i3c_new = g3c_v[i] + flag * n_gnnij
                        j3c_new = g3c_v[j] + flag * n_gnnij
                        if G3c_v[i] != 0:
                            h = abs((i3c_new - e3c_v[i]) / G3c_v[i])
                            g3c_new += h
                        if G3c_v[j] != 0:
                            h = abs((j3c_new - e3c_v[j]) / G3c_v[j])
                            g3c_new += h
                        for ij in range(n_gnnij):
                            nn = nij_v[ij]
                            if G3c_v[nn] != 0:
                                h = abs((g3c_v[nn] + flag - e3c_v[nn]) / G3c_v[nn])
                                g3c_new += h
                        # wedges
                        iwe_new = i2c_new * (i2c_new - 1) / 2 - i3c_new
                        jwe_new = j2c_new * (j2c_new - 1) / 2 - j3c_new
                        if Gwe_v[i] != 0:
                            h = abs((iwe_new - ewe_v[i]) / Gwe_v[i])
                            gwe_new += h
                        if Gwe_v[j] != 0:
                            h = abs((jwe_new - ewe_v[j]) / Gwe_v[j])
                            gwe_new += h
                        for ij in range(n_gnnij):
                            nn = nij_v[ij]
                            nnwe_new = g2c_v[nn] * (g2c_v[nn] - 1) / 2 - (g3c_v[nn] + flag)
                            if Gwe_v[nn] != 0:
                                h = abs((nnwe_new - ewe_v[nn]) / Gwe_v[nn])
                                gwe_new += h


                        if p_method == 'GMN23':
                            gain = g2c_pre + g3c_pre - g2c_new - g3c_new
                        elif p_method == 'GMN23w':
                            gain = g2c_pre + g3c_pre + gwe_pre - g2c_new - g3c_new - gwe_new
                        elif p_method == 'GMN2':
                            gain = g2c_pre - g2c_new
                        elif p_method == 'GMN3':
                            gain = g3c_pre - g3c_new
                        elif p_method == 'GMN2w':
                            gain = g2c_pre + gwe_pre - g2c_new - gwe_new
                        elif p_method == 'GMN3w':
                            gain = g3c_pre + gwe_pre - g3c_new - gwe_new
                        else:
                            print('Error !!!!!!!!!!!!!!!!!!!!!!!')


                        if 'GCC' in p_suffix:
                            # print('GCC')
                            # clustering coefficient
                            if 'WGCC' in p_suffix:
                                print('Weighted version to be decided!!!!')
                            else:
                                if i2c_new > 1:
                                    g_lcc_t_v[i] = <float>((2 * i3c_new) / (i2c_new * (i2c_new - 1)))
                                else:
                                    g_lcc_t_v[i] = 0
                                g_gcc_t = g_gcc - g_lcc_v[i] + g_lcc_t_v[i]
                                if j2c_new > 1:
                                    g_lcc_t_v[j] = <float>((2 * j3c_new) / (j2c_new * (j2c_new - 1)))
                                else:
                                    g_lcc_t_v[j] = 0
                                g_gcc_t = g_gcc_t - g_lcc_v[j] + g_lcc_t_v[j]
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if g2c_v[nn] > 1:
                                        g_lcc_t_v[nn] = <float>((2 * (g3c_v[nn] + flag)) / (g2c_v[nn] * (g2c_v[nn] - 1)))
                                    else:
                                        g_lcc_t_v[nn] = 0
                                    g_gcc_t = g_gcc_t - g_lcc_v[nn] + g_lcc_t_v[nn]
                            if gain > 0 and abs(g_gcc_t - G_gcc) <= abs(g_gcc_init - G_gcc):
                                cnt_e += 1
                                L_new_v[i] = 1
                                L_new_v[j] = 1
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    L_new_v[nn] = 1
                                # g_adj_v
                                if flag == -1:  # to remove
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                    mk3_v[j] = -1
                                elif flag == 1:  # to insert
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                    mk3_v[j] = mk3h_v[j]
                                else:
                                    print('flag error!!!!!!!')
                                # 2-cliques                    
                                g2c_v[i] = i2c_new
                                g2c_v[j] = j2c_new
                                # 3-cliques
                                g3c_v[i] = i3c_new
                                g3c_v[j] = j3c_new
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if G3c_v[nn] != 0:
                                        g3c_v[nn] = g3c_v[nn] + flag
                                # wedges
                                gwe_v[i] = <float>(iwe_new)
                                gwe_v[j] = <float>(jwe_new)
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if Gwe_v[nn] != 0:
                                        gwe_v[nn] = <float>(g2c_v[nn] * (g2c_v[nn] - 1) / 2 - g3c_v[nn])
                                # clustering coefficient
                                g_lcc_v[i] = g_lcc_t_v[i]
                                g_lcc_v[j] = g_lcc_t_v[j]
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    g_lcc_v[nn] = g_lcc_t_v[nn]
                                g_gcc = g_gcc_t
                            else:
                                # clustering coefficient
                                g_lcc_t_v[i] = g_lcc_v[i]
                                g_lcc_t_v[j] = g_lcc_v[j]
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    g_lcc_t_v[nn] = g_lcc_v[nn]
                                g_gcc_t = g_gcc
                            tri_v[i] = 0
                            tri_v[j] = 0
                            for ij in range(n_gnnij):
                                tri_v[nij_v[ij]] = 0
                        elif 'SCC' in p_suffix:
                            # print('SCC')
                            # largest connected component
                            if flag == -1:
                                if n_gnnij == 0:
                                    g_scc_t = 0
                                else:
                                    g_scc_t = g_scc
                            elif flag == 1:
                                if g_scc_v[i] == 0 and g_scc_v[j] == 1:
                                    g_scc_t = g_scc + 1
                                    g_scc_t_v[i] = 1
                                elif g_scc_v[i] == 1 and g_scc_v[j] == 0:
                                    g_scc_t = g_scc + 1
                                    g_scc_t_v[j] = 1
                                else:
                                    g_scc_t = g_scc
                            if gain > 0 and g_scc_t >= g_scc_init:
                                # print(g_scc_t, g_scc_init)
                                cnt_e += 1
                                L_new_v[i] = 1
                                L_new_v[j] = 1
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    L_new_v[nn] = 1
                                # g_adj_v
                                if flag == -1:  # to remove
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                    mk3_v[j] = -1
                                elif flag == 1:  # to insert
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                    mk3_v[j] = mk3h_v[j]
                                else:
                                    print('flag error!!!!!!!')
                                # 2-cliques                    
                                g2c_v[i] = i2c_new
                                g2c_v[j] = j2c_new
                                # 3-cliques
                                g3c_v[i] = i3c_new
                                g3c_v[j] = j3c_new
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if G3c_v[nn] != 0:
                                        g3c_v[nn] = g3c_v[nn] + flag
                                # wedges
                                gwe_v[i] = <float>(iwe_new)
                                gwe_v[j] = <float>(jwe_new)
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if Gwe_v[nn] != 0:
                                        gwe_v[nn] = <float>(g2c_v[nn] * (g2c_v[nn] - 1) / 2 - g3c_v[nn])
                                # largest connected component
                                g_scc_v[i] = g_scc_t_v[i]
                                g_scc_v[j] = g_scc_t_v[j]
                                g_scc = g_scc_t
                            else:
                                # largest connected component
                                g_scc_t_v[i] = g_scc_v[i]
                                g_scc_t_v[j] = g_scc_v[j]
                                g_scc_t = g_scc
                            tri_v[i] = 0
                            tri_v[j] = 0
                            for ij in range(n_gnnij):
                                tri_v[nij_v[ij]] = 0
                        elif 'InitP' in p_suffix:
                            p = np.random.uniform(0, 1, 1)[0]
                            if gain > 0:
                                if (p > Gnnip_v[mk_ni] and flag == -1) or (p <= Gnnip_v[mk_ni] and flag == 1):
                                    cnt_e += 1
                                    L_new_v[i] = 1
                                    L_new_v[j] = 1
                                    for ij in range(n_gnnij):
                                        nn = nij_v[ij]
                                        L_new_v[nn] = 1
                                    # g_adj_v
                                    if flag == -1:  # to remove
                                        g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                        g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                        mk3_v[j] = -1
                                    elif flag == 1:  # to insert
                                        g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                        g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                        mk3_v[j] = mk3h_v[j]
                                    else:
                                        print('flag error!!!!!!!')
                                    # 2-cliques                    
                                    g2c_v[i] = i2c_new
                                    g2c_v[j] = j2c_new
                                    # 3-cliques
                                    g3c_v[i] = i3c_new
                                    g3c_v[j] = j3c_new
                                    for ij in range(n_gnnij):
                                        nn = nij_v[ij]
                                        if G3c_v[nn] != 0:
                                            g3c_v[nn] = g3c_v[nn] + flag
                                    # wedges
                                    gwe_v[i] = <float>(iwe_new)
                                    gwe_v[j] = <float>(jwe_new)
                                    for ij in range(n_gnnij):
                                        nn = nij_v[ij]
                                        if Gwe_v[nn] != 0:
                                            gwe_v[nn] = <float>(g2c_v[nn] * (g2c_v[nn] - 1) / 2 - g3c_v[nn])
                            tri_v[i] = 0
                            tri_v[j] = 0
                            for ij in range(n_gnnij):
                                tri_v[nij_v[ij]] = 0
                        elif 'SG' in p_suffix:
                            # for global clustering coefficient
                            if i2c_new > 1:
                                g_lcc_t_v[i] = <float>((2 * i3c_new) / (i2c_new * (i2c_new - 1)))
                            else:
                                g_lcc_t_v[i] = 0
                            g_gcc_t = g_gcc - g_lcc_v[i] + g_lcc_t_v[i]
                            if j2c_new > 1:
                                g_lcc_t_v[j] = <float>((2 * j3c_new) / (j2c_new * (j2c_new - 1)))
                            else:
                                g_lcc_t_v[j] = 0
                            g_gcc_t = g_gcc_t - g_lcc_v[j] + g_lcc_t_v[j]
                            for ij in range(n_gnnij):
                                nn = nij_v[ij]
                                if g2c_v[nn] > 1:
                                    g_lcc_t_v[nn] = <float>((2 * (g3c_v[nn] + flag)) / (g2c_v[nn] * (g2c_v[nn] - 1)))
                                else:
                                    g_lcc_t_v[nn] = 0
                                g_gcc_t = g_gcc_t - g_lcc_v[nn] + g_lcc_t_v[nn]
                            # for largest connected component
                            if flag == -1:
                                if n_gnnij == 0:
                                    g_scc_t = 0
                                else:
                                    g_scc_t = g_scc
                            elif flag == 1:
                                if g_scc_v[i] == 0 and g_scc_v[j] == 1:
                                    g_scc_t = g_scc + 1
                                    g_scc_t_v[i] = 1
                                elif g_scc_v[i] == 1 and g_scc_v[j] == 0:
                                    g_scc_t = g_scc + 1
                                    g_scc_t_v[j] = 1
                                else:
                                    g_scc_t = g_scc
                            if gain > 0 and g_scc_t >= g_scc_init and abs(g_gcc_t - G_gcc) <= abs(g_gcc_init - G_gcc):
                                cnt_e += 1
                                L_new_v[i] = 1
                                L_new_v[j] = 1
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    L_new_v[nn] = 1
                                # g_adj_v
                                if flag == -1:  # to remove
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                    mk3_v[j] = -1
                                elif flag == 1:  # to insert
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                    mk3_v[j] = mk3h_v[j]
                                else:
                                    print('flag error!!!!!!!')
                                # 2-cliques                    
                                g2c_v[i] = i2c_new
                                g2c_v[j] = j2c_new
                                # 3-cliques
                                g3c_v[i] = i3c_new
                                g3c_v[j] = j3c_new
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if G3c_v[nn] != 0:
                                        g3c_v[nn] = g3c_v[nn] + flag
                                # wedges
                                gwe_v[i] = <float>(iwe_new)
                                gwe_v[j] = <float>(jwe_new)
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if Gwe_v[nn] != 0:
                                        gwe_v[nn] = <float>(g2c_v[nn] * (g2c_v[nn] - 1) / 2 - g3c_v[nn])
                                # global clustering coefficient
                                g_lcc_v[i] = g_lcc_t_v[i]
                                g_lcc_v[j] = g_lcc_t_v[j]
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    g_lcc_v[nn] = g_lcc_t_v[nn]
                                g_gcc = g_gcc_t
                                # largest connected component
                                g_scc_v[i] = g_scc_t_v[i]
                                g_scc_v[j] = g_scc_t_v[j]
                                g_scc = g_scc_t
                            tri_v[i] = 0
                            tri_v[j] = 0
                            for ij in range(n_gnnij):
                                tri_v[nij_v[ij]] = 0
                        elif 'SWG' in p_suffix:
                            # for global clustering coefficient
                            if i2c_new > 1:
                                wi2c_new = wg2c_v[i] + flag * Gwnni_v[mk_ni]
                                wi3c_new = wg3c_v[i] + flag * tri_v[i]
                                g_lcc_t_v[i] = <float>((wi3c_new) / ((wi2c_new) * (i2c_new - 1)))
                            else:
                                g_lcc_t_v[i] = 0
                            g_gcc_t = g_gcc - g_lcc_v[i] + g_lcc_t_v[i]
                            if j2c_new > 1:
                                wj2c_new = wg2c_v[j] + flag * Gwnnj_v[mk_nj]
                                wj3c_new = wg3c_v[j] + flag * tri_v[j]
                                g_lcc_t_v[j] = <float>((wj3c_new) / ((wj2c_new) * (j2c_new - 1)))
                            else:
                                g_lcc_t_v[j] = 0
                            g_gcc_t = g_gcc_t - g_lcc_v[j] + g_lcc_t_v[j]
                            for ij in range(n_gnnij):
                                nn = nij_v[ij]
                                if g2c_v[nn] > 1:
                                    g_lcc_t_v[nn] = <float>((wg3c_v[nn] + flag * tri_v[nn]) / (wg2c_v[nn] * (g2c_v[nn] - 1)))
                                else:
                                    g_lcc_t_v[nn] = 0
                                g_gcc_t = g_gcc_t - g_lcc_v[nn] + g_lcc_t_v[nn]
                            # for largest connected component
                            if flag == -1:
                                if n_gnnij == 0:
                                    g_scc_t = 0
                                else:
                                    g_scc_t = g_scc
                            elif flag == 1:
                                if g_scc_v[i] == 0 and g_scc_v[j] == 1:
                                    g_scc_t = g_scc + 1
                                    g_scc_t_v[i] = 1
                                elif g_scc_v[i] == 1 and g_scc_v[j] == 0:
                                    g_scc_t = g_scc + 1
                                    g_scc_t_v[j] = 1
                                else:
                                    g_scc_t = g_scc
                            if gain > 0 and g_scc_t >= g_scc_init and abs(g_gcc_t - G_gcc) <= abs(g_gcc_init - G_gcc):
                                cnt_e += 1
                                L_new_v[i] = 1
                                L_new_v[j] = 1
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    L_new_v[nn] = 1
                                # g_adj_v
                                if flag == -1:  # to remove
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                    mk3_v[j] = -1
                                elif flag == 1:  # to insert
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                    mk3_v[j] = mk3h_v[j]
                                else:
                                    print('flag error!!!!!!!')
                                # 2-cliques                    
                                g2c_v[i] = i2c_new
                                wg2c_v[i] = wi2c_new
                                g2c_v[j] = j2c_new
                                wg2c_v[j] = wj2c_new
                                # 3-cliques
                                g3c_v[i] = i3c_new
                                wg3c_v[i] = wi3c_new
                                g3c_v[j] = j3c_new
                                wg3c_v[j] = wj3c_new
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if G3c_v[nn] != 0:
                                        g3c_v[nn] = g3c_v[nn] + flag
                                        wg3c_v[nn] = wg3c_v[nn] + flag
                                # wedges
                                gwe_v[i] = <float>(iwe_new)
                                gwe_v[j] = <float>(jwe_new)
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if Gwe_v[nn] != 0:
                                        gwe_v[nn] = <float>(g2c_v[nn] * (g2c_v[nn] - 1) / 2 - g3c_v[nn])
                                # global clustering coefficient
                                g_lcc_v[i] = g_lcc_t_v[i]
                                g_lcc_v[j] = g_lcc_t_v[j]
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    g_lcc_v[nn] = g_lcc_t_v[nn]
                                g_gcc = g_gcc_t
                                # largest connected component
                                g_scc_v[i] = g_scc_t_v[i]
                                g_scc_v[j] = g_scc_t_v[j]
                                g_scc = g_scc_t
                            else:
                                # global clustering coefficient
                                g_lcc_t_v[i] = g_lcc_v[i]
                                g_lcc_t_v[j] = g_lcc_v[j]
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    g_lcc_t_v[nn] = g_lcc_v[nn]
                                g_gcc_t = g_gcc
                                # largest connected component
                                g_scc_t_v[i] = g_scc_v[i]
                                g_scc_t_v[j] = g_scc_v[j]
                                g_scc_t = g_scc
                            tri_v[i] = 0
                            tri_v[j] = 0
                            for ij in range(n_gnnij):
                                tri_v[nij_v[ij]] = 0
                        else:
                            if gain > 0:
                                cnt_e += 1
                                L_new_v[i] = 1
                                L_new_v[j] = 1
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    L_new_v[nn] = 1
                                # g_adj_v
                                if flag == -1:  # to remove
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                    mk3_v[j] = -1
                                elif flag == 1:  # to insert
                                    g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                    g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                    mk3_v[j] = mk3h_v[j]
                                else:
                                    print('flag error!!!!!!!')
                                # 2-cliques                    
                                g2c_v[i] = i2c_new
                                g2c_v[j] = j2c_new
                                # 3-cliques
                                g3c_v[i] = i3c_new
                                g3c_v[j] = j3c_new
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if G3c_v[nn] != 0:
                                        g3c_v[nn] = g3c_v[nn] + flag
                                # wedges
                                gwe_v[i] = <float>(iwe_new)
                                gwe_v[j] = <float>(jwe_new)
                                for ij in range(n_gnnij):
                                    nn = nij_v[ij]
                                    if Gwe_v[nn] != 0:
                                        gwe_v[nn] = <float>(g2c_v[nn] * (g2c_v[nn] - 1) / 2 - g3c_v[nn])
                            tri_v[i] = 0
                            tri_v[j] = 0
                            for ij in range(n_gnnij):
                                tri_v[nij_v[ij]] = 0
                                
                    # mk3[Gnni[gnnip == 1]] = 0
                    for mk in range(G_deg_v[i]):
                        mk3h_v[Gnni_v[mk]] = -1 #
                        mk3_v[Gnni_v[mk]] = -1 #
                        # if gnni_v[mk] == 1: #
                        #     mk3_v[Gnni_v[mk]] = -1 #
            # performance
            print(r, cnt_e, time() - t3 + t2)
            pfm_g2c[G2c1] = np.absolute(g2c - e2c)[G2c1] / G2c[G2c1]
            pfm_g3c[G3c1] = np.absolute(g3c - e3c)[G3c1] / G3c[G3c1]
            pfm_gwe[Gwe1] = np.absolute(gwe - ewe)[Gwe1] / Gwe[Gwe1]
            PFM[0, r] = <float>cnt_e
            PFM[5, r] = <float>(time() - t3 + t2)
            if p_method == 'GMN2':
                PFM[4, r] = np.mean(pfm_g2c)
                print('C2:', PFM[4, r])
            elif p_method == 'GMN3':
                PFM[4, r] = np.mean(pfm_g3c)
                print('C3:', PFM[4, r])
            elif p_method == 'GMN23':
                PFM[4, r] = np.mean(pfm_g2c + pfm_g3c)
                print('C23:', PFM[4, r])
            elif p_method == 'GMN3w':
                PFM[4, r] = np.mean(pfm_g3c + pfm_gwe)
                print('C3w:', PFM[4, r])
            elif p_method == 'GMN23w':
                PFM[4, r] = np.mean(pfm_g2c + pfm_g3c + pfm_gwe)
                print('C23WE:', PFM[4, r])
            if r >= 2 and PFM[4, r - 1] - PFM[4, r] <= p_tolerance:
                # if "SG" in p_suffix or "SWG" in p_suffix:
                #     print(g_scc_init, g_scc, G_scc)
                #     print(g_gcc_init, g_gcc, G_gcc)
                #     print(np.sum(g_adj) / (2 * p_edg_num))
                #     print(int(np.sum(g2c.astype(np.int32)) / 2) == int(np.sum(g_adj.astype(np.int32)) / 2))
                PFM[4, r] = np.mean(pfm_g2c + pfm_g3c + pfm_gwe)
                PFM[1, :] = pfm_g2c
                PFM[2, :] = pfm_g3c
                PFM[3, :] = pfm_gwe
                break
        rpt[rp] = <float>(time() - t3 + t2)
        print(time() - t3 + t2)
        print(int(np.sum(g2c.astype(np.int32)) / 2) == int(np.sum(g_adj.astype(np.int32)) / 2))
        with open(OUP_GRAPH, 'w', newline='', encoding='UTF-8') as gph:
            gph.write(str(p_n) + ' ' + str(int(np.sum(g2c.astype(np.int32)) / 2)) + ' ' + str(1) + '\n')
        with open(OUP_GRAPH, 'a', newline='', encoding='UTF-8') as gph:
            for i in range(p_n):
                if g2c_v[i] != 0:
                    Gni_v = G_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    Gwnni_v = G_wadj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    gni_v = g_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    aw = np.vstack([np.asarray(Gni_v)[np.asarray(gni_v) == 1] + 1, np.asarray(Gwnni_v)[np.asarray(gni_v) == 1]]).T.flatten().astype(np.int32)
                    np.savetxt(gph, aw, fmt='%d', newline=' ')
                gph.write('\n')
    np.save(OUP_RT, np.array([np.mean(rpt), np.std(rpt)]))


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef (cnp.float64_t, cnp.float64_t) initialization_using_the_original_graph(
Py_ssize_t p_n,
cnp.int8_t[:] p_g_adj_v,
cnp.float32_t[:] p_g2c_v,
cnp.float32_t[:] p_g3c_v,
cnp.float32_t[:] p_gwe_v,
cnp.float32_t[:] p_G2c_v,
cnp.float32_t[:] p_G3c_v,
cnp.float32_t[:] p_Gwe_v):

    cdef:
        Py_ssize_t e=0

    for e in range(p_g_adj_v.shape[0]):
        p_g_adj_v[e] = 1
    for i in range(p_n):
        p_g2c_v[i] = p_G2c_v[i]
        p_g3c_v[i] = p_G3c_v[i]
        p_gwe_v[i] = p_Gwe_v[i]
    return 1, 1


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef void initialization_using_LD_RE_LRE_P(
str p_inpnet,
Py_ssize_t p_n,
Py_ssize_t p_edg_num,
cnp.float64_t p_scaling,
str p_suffix,
cnp.int8_t[:] p_g_adj_v,
cnp.float32_t[:] p_g2c_v,
cnp.float32_t[:] p_g3c_v,
cnp.float32_t[:] p_gwe_v,
cnp.float32_t[:] p_G_adjp_v,
cnp.float32_t[:] p_G2c_v,
cnp.float32_t[:] p_G3c_v,
cnp.float32_t[:] p_Gwe_v,
cnp.int32_t[:] p_G_deg_v,
cnp.int32_t[:] p_G_cudeg_v,
cnp.int32_t[:] p_G_adj_v):

    cdef:
        cnp.ndarray[cnp.float32_t, ndim=1] adjp
        cnp.ndarray[cnp.int32_t, ndim=2] g_edg, G_edg
        cnp.ndarray[cnp.int32_t, ndim=1] g_deg, g_cudeg, shift, g_adj, G_shift
        cnp.float32_t[:] adjp_v
        cnp.int8_t[:] tg_ni_v
        cnp.int32_t[:] g_deg_v, g_cudeg_v, shift_v, g_adj_v, g_ni_v, g_nj_v, mk_v, G_ni_v, G_shift_v
        cnp.int32_t[:, :] g_edg_v, G_edg_v
        cnp.int32_t eid=0, idx=0, nd1=0, nd2=0, num=0, u=0, v=0, w=0
        Py_ssize_t e=0, i=0, ii=0, jj=0, ig=0, iG=0, j=0, g_edg_num=0
        object G, g


    G = nk.readGraph(f'{p_inpnet}_wgph.graph', nk.Format.METIS)
    G_edg = np.load(f'{p_inpnet}_Udwp_1.npy')[:, :2].astype(np.int32) - 1
    G_edg_v = G_edg
    adjp = np.zeros(p_edg_num, dtype=np.float32)
    adjp_v = adjp
    G.indexEdges()
    for r in range(25):
        if 'LD' in p_suffix:
            g = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, 0.1)
        elif 'RRES' in p_suffix:
            g = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, 0.1)
        elif 'RES' in p_suffix:
            g = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, 0.1)
        g_edg_num = g.numberOfEdges()
        g_edg = np.zeros((g_edg_num, 2), dtype=np.int32)
        g_edg_v = g_edg
        eid = 0
        for u, v in g.iterEdges():
            g_edg_v[eid, 0] = u
            g_edg_v[eid, 1] = v
            eid += 1
        if eid != g_edg_num:
            os._exit(0)
        del g
        g_edg = g_edg[np.lexsort((g_edg[:, 1], g_edg[:, 0])), :]
        g_edg_v = g_edg
        eid = 0
        for e in range(g_edg_num):
            while g_edg_v[e, 0] != G_edg_v[eid, 0] or g_edg_v[e, 1] != G_edg_v[eid, 1]:
                eid += 1
            adjp_v[eid] += 1
            eid += 1
        del g_edg
        # print(g_edg_num, np.where(adjp == (r + 1))[0].shape[0])
    # os._exit(0)
    adjp = adjp / 25
    adjp_v = adjp
    G_shift = np.zeros(p_n, dtype=np.int32)
    G_shift_v = G_shift
    for i in range(p_n):
        G_shift_v[i] = p_G_cudeg_v[i]
    for e in range(p_edg_num):
        nd1, nd2 = G_edg_v[e][0], G_edg_v[e][1]
        p_G_adjp_v[G_shift_v[nd1]] = adjp_v[e]
        p_G_adjp_v[G_shift_v[nd2]] = adjp_v[e]
        G_shift_v[nd1] += 1
        G_shift_v[nd2] += 1
    # print(np.where(adjp == 1)[0].shape[0])
    # os._exit(0)
    del G_edg, adjp


    if 'LD' in p_suffix:
        g = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RRES' in p_suffix:
        g = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RES' in p_suffix:
        g = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    del G

    # deg_by_nk = np.array(nk.centrality.DegreeCentrality(g).run().scores())
    # lcc_by_nk = np.array(nk.centrality.LocalClusteringCoefficient(g).run().scores())

    g_edg_num = g.numberOfEdges()
    g_edg = np.zeros((g_edg_num, 2), dtype=np.int32)
    g_edg_v = g_edg
    eid = 0
    for u, v in g.iterEdges():
        g_edg_v[eid, 0] = u
        g_edg_v[eid, 1] = v
        eid += 1
    if eid != g_edg_num:
        os._exit(0)
    del g
    g_edg = g_edg[np.lexsort((g_edg[:, 1], g_edg[:, 0])), :]
    g_edg_v = g_edg
    g_deg = np.zeros(p_n, dtype=np.int32)
    g_deg_v = g_deg
    for e in range(g_edg_num):
        nd1, nd2 = g_edg_v[e][0], g_edg_v[e][1]
        g_deg_v[nd1] += 1
        g_deg_v[nd2] += 1
        p_g2c_v[nd1] += 1
        p_g2c_v[nd2] += 1
    g_cudeg = np.zeros(p_n + 1, dtype=np.int32)
    g_cudeg_v = g_cudeg
    num = 0
    for i in range(p_n + 1):
        g_cudeg_v[i] = num
        if i < p_n:
            num += g_deg_v[i]
    shift = g_cudeg.copy()
    shift_v = shift
    g_adj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_adj_v = g_adj
    for e in range(g_edg_num):
        nd1, nd2 = g_edg_v[e][0], g_edg_v[e][1]
        g_adj_v[shift_v[nd1]] = nd2
        g_adj_v[shift_v[nd2]] = nd1
        shift_v[nd1] += 1
        shift_v[nd2] += 1
    del g_edg


    mk_v = np.zeros(p_n, dtype=np.int32) - 1
    for i in range(p_n):
        if g_deg_v[i] > 1:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            idx = 0
            for ii in range(g_deg_v[i]):
                mk_v[g_ni_v[ii]] = idx
                idx += 1
            for ii in range(g_deg_v[i] - 1):
                j = g_ni_v[ii]
                g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                for jj in range(g_deg_v[j]):
                    if mk_v[g_nj_v[jj]] != -1:
                        idx = mk_v[g_nj_v[jj]]
                        p_g3c_v[i] += 1
                mk_v[j] = -1
            mk_v[g_ni_v[ii + 1]] = -1


    for i in range(p_n):
        p_gwe_v[i] = <float>(0.5 * p_g2c_v[i] * (p_g2c_v[i] - 1) - p_g3c_v[i])

    
    # g_lcc = np.zeros(p_n)
    # for i in range(p_n):
    #     if p_g2c_v[i] > 1:
    #         g_lcc[i] = <float>(p_g3c_v[i] / (0.5 * p_g2c_v[i] * (p_g2c_v[i] - 1)))
    # print(np.array_equal(deg_by_nk, np.asarray(p_g2c_v)))
    # d = lcc_by_nk - g_lcc
    # print(np.array_equal(lcc_by_nk, g_lcc))
    # print(np.max(abs(d[d != 0])))
    # os._exit(0)

    for i in range(p_n):
        if g_deg_v[i] != 0:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            G_ni_v = p_G_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            tg_ni_v = p_g_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            ig = 0
            iG = 0
            while (ig != g_deg_v[i] and iG != p_G_deg_v[i]):
                if g_ni_v[ig] == G_ni_v[iG]:
                    tg_ni_v[iG] = 1
                    ig += 1
                    iG += 1
                elif g_ni_v[ig] < G_ni_v[iG]:
                    ig += 1
                else:
                    iG += 1
    del g_adj


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef (cnp.float64_t, cnp.float64_t) initialization_using_LD_RE_LRE_GCC(
str p_inpnet,
Py_ssize_t p_n,
cnp.float64_t p_scaling,
str p_suffix,
cnp.int8_t[:] p_g_adj_v,
cnp.float32_t[:] p_g2c_v,
cnp.float32_t[:] p_g3c_v,
cnp.float32_t[:] p_gwe_v,
cnp.float32_t[:] p_g_lcc_v,
cnp.float32_t[:] p_g_lcc_t_v,
cnp.float32_t[:] p_G2c_v,
cnp.float32_t[:] p_G3c_v,
cnp.float32_t[:] p_Gwe_v,
cnp.int32_t[:] p_G_deg_v,
cnp.int32_t[:] p_G_cudeg_v,
cnp.int32_t[:] p_G_adj_v):

    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] g_edgw
        cnp.ndarray[cnp.int32_t, ndim=1] g_deg, g_cudeg, shift, g_adj, g_wadj
        cnp.ndarray[cnp.float32_t, ndim=1] g_wdeg
        cnp.int8_t[:] tg_ni_v
        cnp.int32_t[:] g_deg_v, g_cudeg_v, shift_v, g_adj_v, g_wadj_v, g_ni_v, g_wni_v, g_nj_v, mk_v, G_ni_v
        cnp.float32_t[:] G_lcc_v, g_wdeg_v
        cnp.int32_t[:, :] g_edgw_v
        cnp.int32_t eid=0, idx=0, nd1=0, nd2=0, num=0, u=0, v=0, w=0, g3c=0, wg3c=0
        Py_ssize_t e=0, i=0, ii=0, jj=0, ig=0, iG=0, j=0, g_edg_num=0
        cnp.float64_t g_gcc=0, G_gcc=0
        object G, g


    G = nk.readGraph(f'{p_inpnet}_wgph.graph', nk.Format.METIS)
    G.indexEdges()
    if 'LD' in p_suffix:
        g = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RRES' in p_suffix:
        g = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RES' in p_suffix:
        g = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    del G


    g_edg_num = g.numberOfEdges()
    g_edgw = np.zeros((g_edg_num, 3), dtype=np.int32)
    g_edgw_v = g_edgw
    eid = 0
    for u, v, w in g.iterEdgesWeights():
        g_edgw_v[eid, 0] = u
        g_edgw_v[eid, 1] = v
        g_edgw_v[eid, 2] = w - 1
        eid += 1
    if eid != g_edg_num:
        os._exit(0)
    del g
    g_edgw = g_edgw[np.lexsort((g_edgw[:, 1], g_edgw[:, 0])), :]
    g_edgw_v = g_edgw
    g_deg = np.zeros(p_n, dtype=np.int32)
    g_wdeg = np.zeros(p_n, dtype=np.float32)
    g_deg_v = g_deg
    g_wdeg_v = g_wdeg
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_deg_v[nd1] += 1
        g_deg_v[nd2] += 1
        p_g2c_v[nd1] += 1
        p_g2c_v[nd2] += 1
        g_wdeg_v[nd1] += (g_edgw_v[e][2] + 1)
        g_wdeg_v[nd2] += (g_edgw_v[e][2] + 1)
    g_cudeg = np.zeros(p_n + 1, dtype=np.int32)
    g_cudeg_v = g_cudeg
    num = 0
    for i in range(p_n + 1):
        g_cudeg_v[i] = num
        if i < p_n:
            num += g_deg_v[i]
    shift = g_cudeg.copy()
    shift_v = shift
    g_adj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_wadj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_adj_v = g_adj
    g_wadj_v = g_wadj
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_adj_v[shift_v[nd1]] = nd2
        g_adj_v[shift_v[nd2]] = nd1
        g_wadj_v[shift_v[nd1]] = (g_edgw_v[e][2] + 1)
        g_wadj_v[shift_v[nd2]] = (g_edgw_v[e][2] + 1)
        shift_v[nd1] += 1
        shift_v[nd2] += 1
    del g_edgw


    G_gcc = 0
    g_gcc = 0
    if 'WGCC' in p_suffix:
        G_lcc_v = np.load(f'{p_inpnet}_Gstats.npy').astype(np.float32)[2, :]
        for i in range(p_n):
            G_gcc += G_lcc_v[i]
        mk_v = np.zeros(p_n, dtype=np.int32) - 1
        for i in range(p_n):
            if g_deg_v[i] > 1:
                g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
                g_wni_v = g_wadj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
                idx = 0
                for ii in range(g_deg_v[i]):
                    mk_v[g_ni_v[ii]] = idx
                    idx += 1
                wg3c = 0
                for ii in range(g_deg_v[i] - 1):
                    j = g_ni_v[ii]
                    g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                    for jj in range(g_deg_v[j]):
                        if mk_v[g_nj_v[jj]] != -1:
                            idx = mk_v[g_nj_v[jj]]
                            wg3c += (g_wni_v[ii] + g_wni_v[idx])
                            p_g3c_v[i] += 1
                    mk_v[j] = -1
                mk_v[g_ni_v[ii + 1]] = -1
                p_g_lcc_v[i] = <float>(wg3c / (g_wdeg_v[i] * (p_g2c_v[i] - 1)))
                p_g_lcc_t_v[i] = p_g_lcc_v[i]
                g_gcc += p_g_lcc_v[i]
    else:
        for i in range(p_n):
            if p_G2c_v[i] > 1:
                G_gcc += ((2 * p_G3c_v[i]) / (p_G2c_v[i] * (p_G2c_v[i] - 1)))
        mk_v = np.zeros(p_n, dtype=np.int32) - 1
        for i in range(p_n):
            if g_deg_v[i] > 1:
                g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
                g_wni_v = g_wadj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
                idx = 0
                for ii in range(g_deg_v[i]):
                    mk_v[g_ni_v[ii]] = idx
                    idx += 1
                g3c = 0
                for ii in range(g_deg_v[i] - 1):
                    j = g_ni_v[ii]
                    g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                    for jj in range(g_deg_v[j]):
                        if mk_v[g_nj_v[jj]] != -1:
                            idx = mk_v[g_nj_v[jj]]
                            g3c += 1
                            p_g3c_v[i] += 1
                    mk_v[j] = -1
                mk_v[g_ni_v[ii + 1]] = -1
                # must be p_g2c_v, this is float
                p_g_lcc_v[i] = <float>(2 * g3c / (p_g2c_v[i] * (p_g2c_v[i] - 1)))
                p_g_lcc_t_v[i] = p_g_lcc_v[i]
                g_gcc += p_g_lcc_v[i]
    del g_wadj


    for i in range(p_n):
        p_gwe_v[i] = <float>(0.5 * p_g2c_v[i] * (p_g2c_v[i] - 1) - p_g3c_v[i])


    for i in range(p_n):
        if g_deg_v[i] != 0:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            G_ni_v = p_G_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            tg_ni_v = p_g_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            ig = 0
            iG = 0
            while (ig != g_deg_v[i] and iG != p_G_deg_v[i]):
                if g_ni_v[ig] == G_ni_v[iG]:
                    tg_ni_v[iG] = 1
                    ig += 1
                    iG += 1
                elif g_ni_v[ig] < G_ni_v[iG]:
                    ig += 1
                else:
                    iG += 1
    del g_adj
    return g_gcc, G_gcc


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef (cnp.float64_t, cnp.float64_t) initialization_using_LD_RE_LRE_SCC(
str p_inpnet,
Py_ssize_t p_n,
cnp.float64_t p_scaling,
str p_suffix,
cnp.int8_t[:] p_g_adj_v,
cnp.float32_t[:] p_g2c_v,
cnp.float32_t[:] p_g3c_v,
cnp.float32_t[:] p_gwe_v,
cnp.int8_t[:] p_g_scc_v,
cnp.int8_t[:] p_g_scc_t_v,
cnp.int32_t[:] p_G_deg_v,
cnp.int32_t[:] p_G_cudeg_v,
cnp.int32_t[:] p_G_adj_v):

    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] g_edgw
        cnp.ndarray[cnp.int32_t, ndim=1] g_deg, g_cudeg, shift, g_adj, g_wadj
        cnp.ndarray[cnp.float32_t, ndim=1] g_wdeg
        cnp.int8_t[:] tg_ni_v
        cnp.int32_t[:] g_deg_v, g_cudeg_v, shift_v, g_adj_v, g_wadj_v, g_ni_v, g_wni_v, g_nj_v, mk_v, G_ni_v
        cnp.float32_t[:] G_lcc_v, g_wdeg_v
        cnp.int32_t[:, :] g_edgw_v
        cnp.int32_t eid=0, idx=0, nd1=0, nd2=0, num=0, u=0, v=0, w=0, g3c=0, wg3c=0
        Py_ssize_t e=0, i=0, ii=0, jj=0, ig=0, iG=0, j=0, g_edg_num=0
        cnp.float64_t g_scc=0, G_scc=0
        object G, g, gscc


    G = nk.readGraph(f'{p_inpnet}_wgph.graph', nk.Format.METIS)
    G_scc = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, False).numberOfNodes()
    G.indexEdges()
    if 'LD' in p_suffix:
        g = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RRES' in p_suffix:
        g = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RES' in p_suffix:
        g = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    del G


    gscc = nk.components.ConnectedComponents.extractLargestConnectedComponent(g, False)
    g_scc = gscc.numberOfNodes()
    for u, v in gscc.iterEdges():
        if p_g_scc_v[u] == 0:
            p_g_scc_v[u] = 1
            p_g_scc_t_v[u] = 1
        if p_g_scc_v[v] == 0:
            p_g_scc_v[v] = 1
            p_g_scc_t_v[v] = 1
    del gscc


    g_edg_num = g.numberOfEdges()
    g_edgw = np.zeros((g_edg_num, 3), dtype=np.int32)
    g_edgw_v = g_edgw
    eid = 0
    for u, v, w in g.iterEdgesWeights():
        g_edgw_v[eid, 0] = u
        g_edgw_v[eid, 1] = v
        g_edgw_v[eid, 2] = w - 1
        eid += 1
    if eid != g_edg_num:
        os._exit(0)
    del g
    g_edgw = g_edgw[np.lexsort((g_edgw[:, 1], g_edgw[:, 0])), :]
    g_edgw_v = g_edgw
    g_deg = np.zeros(p_n, dtype=np.int32)
    g_wdeg = np.zeros(p_n, dtype=np.float32)
    g_deg_v = g_deg
    g_wdeg_v = g_wdeg
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_deg_v[nd1] += 1
        g_deg_v[nd2] += 1
        p_g2c_v[nd1] += 1
        p_g2c_v[nd2] += 1
        g_wdeg_v[nd1] += (g_edgw_v[e][2] + 1)
        g_wdeg_v[nd2] += (g_edgw_v[e][2] + 1)
    g_cudeg = np.zeros(p_n + 1, dtype=np.int32)
    g_cudeg_v = g_cudeg
    num = 0
    for i in range(p_n + 1):
        g_cudeg_v[i] = num
        if i < p_n:
            num += g_deg_v[i]
    shift = g_cudeg.copy()
    shift_v = shift
    g_adj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_adj_v = g_adj
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_adj_v[shift_v[nd1]] = nd2
        g_adj_v[shift_v[nd2]] = nd1
        shift_v[nd1] += 1
        shift_v[nd2] += 1
    del g_edgw


    mk_v = np.zeros(p_n, dtype=np.int32) - 1
    for i in range(p_n):
        if g_deg_v[i] > 1:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            idx = 0
            for ii in range(g_deg_v[i]):
                mk_v[g_ni_v[ii]] = idx
                idx += 1
            g3c = 0
            for ii in range(g_deg_v[i] - 1):
                j = g_ni_v[ii]
                g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                for jj in range(g_deg_v[j]):
                    if mk_v[g_nj_v[jj]] != -1:
                        idx = mk_v[g_nj_v[jj]]
                        g3c += 1
                        p_g3c_v[i] += 1
                mk_v[j] = -1
            mk_v[g_ni_v[ii + 1]] = -1


    for i in range(p_n):
        p_gwe_v[i] = <float>(0.5 * p_g2c_v[i] * (p_g2c_v[i] - 1) - p_g3c_v[i])


    for i in range(p_n):
        if g_deg_v[i] != 0:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            G_ni_v = p_G_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            tg_ni_v = p_g_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            ig = 0
            iG = 0
            while (ig != g_deg_v[i] and iG != p_G_deg_v[i]):
                if g_ni_v[ig] == G_ni_v[iG]:
                    tg_ni_v[iG] = 1
                    ig += 1
                    iG += 1
                elif g_ni_v[ig] < G_ni_v[iG]:
                    ig += 1
                else:
                    iG += 1
    del g_adj
    return g_scc, G_scc


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef (cnp.float64_t, cnp.float64_t, cnp.float64_t, cnp.float64_t) initialization_using_LD_RE_LRE_SG(
str p_inpnet,
Py_ssize_t p_n,
cnp.float64_t p_scaling,
str p_suffix,
cnp.int8_t[:] p_g_adj_v,
cnp.float32_t[:] p_g2c_v,
cnp.float32_t[:] p_g3c_v,
cnp.float32_t[:] p_gwe_v,
cnp.int8_t[:] p_g_scc_v,
cnp.int8_t[:] p_g_scc_t_v,
cnp.float32_t[:] p_g_lcc_v,
cnp.float32_t[:] p_g_lcc_t_v,
cnp.float32_t[:] p_G2c_v,
cnp.float32_t[:] p_G3c_v,
cnp.float32_t[:] p_Gwe_v,
cnp.int32_t[:] p_G_deg_v,
cnp.int32_t[:] p_G_cudeg_v,
cnp.int32_t[:] p_G_adj_v):

    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] g_edgw
        cnp.ndarray[cnp.int32_t, ndim=1] g_deg, g_cudeg, shift, g_adj, g_wadj
        cnp.ndarray[cnp.float32_t, ndim=1] g_wdeg
        cnp.int8_t[:] tg_ni_v
        cnp.int32_t[:] g_deg_v, g_cudeg_v, shift_v, g_adj_v, g_wadj_v, g_ni_v, g_wni_v, g_nj_v, mk_v, G_ni_v
        cnp.float32_t[:] G_lcc_v, g_wdeg_v
        cnp.int32_t[:, :] g_edgw_v
        cnp.int32_t eid=0, idx=0, nd1=0, nd2=0, num=0, u=0, v=0, w=0, g3c=0, wg3c=0
        Py_ssize_t e=0, i=0, ii=0, jj=0, ig=0, iG=0, j=0, g_edg_num=0
        cnp.float64_t g_scc=0, G_scc=0, g_gcc=0, G_gcc=0
        object G, g, gscc


    G = nk.readGraph(f'{p_inpnet}_wgph.graph', nk.Format.METIS)
    G_scc = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, False).numberOfNodes()
    G.indexEdges()
    if 'LD' in p_suffix:
        g = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RRES' in p_suffix:
        g = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RES' in p_suffix:
        g = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    del G


    gscc = nk.components.ConnectedComponents.extractLargestConnectedComponent(g, False)
    g_scc = gscc.numberOfNodes()
    for u, v in gscc.iterEdges():
        if p_g_scc_v[u] == 0:
            p_g_scc_v[u] = 1
            p_g_scc_t_v[u] = 1
        if p_g_scc_v[v] == 0:
            p_g_scc_v[v] = 1
            p_g_scc_t_v[v] = 1
    del gscc


    g_edg_num = g.numberOfEdges()
    g_edgw = np.zeros((g_edg_num, 3), dtype=np.int32)
    g_edgw_v = g_edgw
    eid = 0
    for u, v, w in g.iterEdgesWeights():
        g_edgw_v[eid, 0] = u
        g_edgw_v[eid, 1] = v
        g_edgw_v[eid, 2] = w - 1
        eid += 1
    if eid != g_edg_num:
        os._exit(0)
    del g
    g_edgw = g_edgw[np.lexsort((g_edgw[:, 1], g_edgw[:, 0])), :]
    g_edgw_v = g_edgw
    g_deg = np.zeros(p_n, dtype=np.int32)
    g_wdeg = np.zeros(p_n, dtype=np.float32)
    g_deg_v = g_deg
    g_wdeg_v = g_wdeg
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_deg_v[nd1] += 1
        g_deg_v[nd2] += 1
        p_g2c_v[nd1] += 1
        p_g2c_v[nd2] += 1
        g_wdeg_v[nd1] += (g_edgw_v[e][2] + 1)
        g_wdeg_v[nd2] += (g_edgw_v[e][2] + 1)
    g_cudeg = np.zeros(p_n + 1, dtype=np.int32)
    g_cudeg_v = g_cudeg
    num = 0
    for i in range(p_n + 1):
        g_cudeg_v[i] = num
        if i < p_n:
            num += g_deg_v[i]
    shift = g_cudeg.copy()
    shift_v = shift
    g_adj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_adj_v = g_adj
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_adj_v[shift_v[nd1]] = nd2
        g_adj_v[shift_v[nd2]] = nd1
        shift_v[nd1] += 1
        shift_v[nd2] += 1
    del g_edgw


    G_gcc = 0
    g_gcc = 0
    for i in range(p_n):
        if p_G2c_v[i] > 1:
            G_gcc += ((2 * p_G3c_v[i]) / (p_G2c_v[i] * (p_G2c_v[i] - 1)))
    mk_v = np.zeros(p_n, dtype=np.int32) - 1
    for i in range(p_n):
        if g_deg_v[i] > 1:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            idx = 0
            for ii in range(g_deg_v[i]):
                mk_v[g_ni_v[ii]] = idx
                idx += 1
            g3c = 0
            for ii in range(g_deg_v[i] - 1):
                j = g_ni_v[ii]
                g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                for jj in range(g_deg_v[j]):
                    if mk_v[g_nj_v[jj]] != -1:
                        idx = mk_v[g_nj_v[jj]]
                        g3c += 1
                        p_g3c_v[i] += 1
                mk_v[j] = -1
            mk_v[g_ni_v[ii + 1]] = -1
            # must be p_g2c_v, this is float
            p_g_lcc_v[i] = <float>(2 * g3c / (p_g2c_v[i] * (p_g2c_v[i] - 1)))
            p_g_lcc_t_v[i] = p_g_lcc_v[i]
            g_gcc += p_g_lcc_v[i]


    for i in range(p_n):
        p_gwe_v[i] = <float>(0.5 * p_g2c_v[i] * (p_g2c_v[i] - 1) - p_g3c_v[i])


    for i in range(p_n):
        if g_deg_v[i] != 0:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            G_ni_v = p_G_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            tg_ni_v = p_g_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            ig = 0
            iG = 0
            while (ig != g_deg_v[i] and iG != p_G_deg_v[i]):
                if g_ni_v[ig] == G_ni_v[iG]:
                    tg_ni_v[iG] = 1
                    ig += 1
                    iG += 1
                elif g_ni_v[ig] < G_ni_v[iG]:
                    ig += 1
                else:
                    iG += 1
    del g_adj
    return g_scc, G_scc, g_gcc, G_gcc


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef (cnp.float64_t, cnp.float64_t, cnp.float64_t, cnp.float64_t) initialization_using_LD_RE_LRE_SWG(
str p_inpnet,
Py_ssize_t p_n,
cnp.float64_t p_scaling,
str p_suffix,
cnp.int8_t[:] p_g_adj_v,
cnp.float32_t[:] p_g2c_v,
cnp.float32_t[:] p_wg2c_v,
cnp.float32_t[:] p_g3c_v,
cnp.float32_t[:] p_wg3c_v,
cnp.float32_t[:] p_gwe_v,
cnp.float32_t[:] p_wgwe_v,
cnp.int8_t[:] p_g_scc_v,
cnp.int8_t[:] p_g_scc_t_v,
cnp.float32_t[:] p_g_lcc_v,
cnp.float32_t[:] p_g_lcc_t_v,
cnp.float32_t[:] p_G2c_v,
cnp.float32_t[:] p_G3c_v,
cnp.float32_t[:] p_Gwe_v,
cnp.int32_t[:] p_G_deg_v,
cnp.int32_t[:] p_G_wdeg_v,
cnp.int32_t[:] p_G_cudeg_v,
cnp.int32_t[:] p_G_adj_v,
cnp.int32_t[:] p_G_wadj_v):

    cdef:
        cnp.ndarray[cnp.int32_t, ndim=2] g_edgw
        cnp.ndarray[cnp.int32_t, ndim=1] g_deg, g_cudeg, shift, g_adj, g_wadj
        cnp.int8_t[:] tg_ni_v
        cnp.int32_t[:] g_deg_v, g_cudeg_v, shift_v, g_adj_v, g_wadj_v, g_ni_v, g_wni_v, g_nj_v, mk_v, G_ni_v, G_wni_v, G_nj_v
        cnp.float32_t[:] G_lcc_v
        cnp.int32_t[:, :] g_edgw_v
        cnp.int32_t eid=0, idx=0, nd1=0, nd2=0, num=0, u=0, v=0, w=0, g3c=0, wg3c=0, wG3c=0
        Py_ssize_t e=0, i=0, ii=0, jj=0, ig=0, iG=0, j=0, g_edg_num=0
        cnp.float64_t g_scc=0, G_scc=0, g_gcc=0, G_gcc=0
        object G, g, gscc


    G = nk.readGraph(f'{p_inpnet}_wgph.graph', nk.Format.METIS)
    G_scc = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, False).numberOfNodes()
    G.indexEdges()
    if 'LD' in p_suffix:
        g = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RRES' in p_suffix:
        g = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, p_scaling)
    elif 'RES' in p_suffix:
        g = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
    del G


    gscc = nk.components.ConnectedComponents.extractLargestConnectedComponent(g, False)
    g_scc = gscc.numberOfNodes()
    for u, v in gscc.iterEdges():
        if p_g_scc_v[u] == 0:
            p_g_scc_v[u] = 1
            p_g_scc_t_v[u] = 1
        if p_g_scc_v[v] == 0:
            p_g_scc_v[v] = 1
            p_g_scc_t_v[v] = 1
    del gscc


    g_edg_num = g.numberOfEdges()
    g_edgw = np.zeros((g_edg_num, 3), dtype=np.int32)
    g_edgw_v = g_edgw
    eid = 0
    for u, v, w in g.iterEdgesWeights():
        g_edgw_v[eid, 0] = u
        g_edgw_v[eid, 1] = v
        g_edgw_v[eid, 2] = w - 1
        eid += 1
    if eid != g_edg_num:
        os._exit(0)
    del g
    g_edgw = g_edgw[np.lexsort((g_edgw[:, 1], g_edgw[:, 0])), :]
    g_edgw_v = g_edgw
    g_deg = np.zeros(p_n, dtype=np.int32)
    g_deg_v = g_deg
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_deg_v[nd1] += 1
        g_deg_v[nd2] += 1
        p_g2c_v[nd1] += 1
        p_g2c_v[nd2] += 1
        p_wg2c_v[nd1] += (g_edgw_v[e][2] + 1)
        p_wg2c_v[nd2] += (g_edgw_v[e][2] + 1)
    g_cudeg = np.zeros(p_n + 1, dtype=np.int32)
    g_cudeg_v = g_cudeg
    num = 0
    for i in range(p_n + 1):
        g_cudeg_v[i] = num
        if i < p_n:
            num += g_deg_v[i]
    shift = g_cudeg.copy()
    shift_v = shift
    g_adj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_wadj = np.zeros(np.sum(g_deg), dtype=np.int32)
    g_adj_v = g_adj
    g_wadj_v = g_wadj
    for e in range(g_edg_num):
        nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
        g_adj_v[shift_v[nd1]] = nd2
        g_adj_v[shift_v[nd2]] = nd1
        g_wadj_v[shift_v[nd1]] = (g_edgw_v[e][2] + 1)
        g_wadj_v[shift_v[nd2]] = (g_edgw_v[e][2] + 1)
        shift_v[nd1] += 1
        shift_v[nd2] += 1
    del g_edgw


    G_gcc = 0
    g_gcc = 0
    # Weighted
    mk_v = np.zeros(p_n, dtype=np.int32) - 1
    for i in range(p_n):
        if p_G_deg_v[i] > 1:
            G_ni_v = p_G_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            G_wni_v = p_G_wadj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            idx = 0
            for ii in range(p_G_deg_v[i]):
                mk_v[G_ni_v[ii]] = idx
                idx += 1
            wG3c = 0
            for ii in range(p_G_deg_v[i] - 1):
                j = G_ni_v[ii]
                G_nj_v = p_G_adj_v[p_G_cudeg_v[j]:p_G_cudeg_v[j + 1]]
                for jj in range(p_G_deg_v[j]):
                    if mk_v[G_nj_v[jj]] != -1:
                        wG3c += (G_wni_v[ii] + G_wni_v[mk_v[G_nj_v[jj]]])
                mk_v[j] = -1
            mk_v[G_ni_v[ii + 1]] = -1
            # print(wG3c, p_G_wdeg_v[i], p_G_deg_v[i], <float>(<float>(wG3c) / <float>(p_G_wdeg_v[i] * (p_G_deg_v[i] - 1))))
            G_gcc += <float>(<float>(wG3c) / <float>(p_G_wdeg_v[i] * (p_G_deg_v[i] - 1)))
    # os._exit(0)
    mk_v = np.zeros(p_n, dtype=np.int32) - 1
    for i in range(p_n):
        if g_deg_v[i] > 1:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            g_wni_v = g_wadj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            idx = 0
            for ii in range(g_deg_v[i]):
                mk_v[g_ni_v[ii]] = idx
                idx += 1
            wg3c = 0
            for ii in range(g_deg_v[i] - 1):
                j = g_ni_v[ii]
                g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                for jj in range(g_deg_v[j]):
                    if mk_v[g_nj_v[jj]] != -1:
                        wg3c += (g_wni_v[ii] + g_wni_v[mk_v[g_nj_v[jj]]])
                        p_g3c_v[i] += 1
                mk_v[j] = -1
            mk_v[g_ni_v[ii + 1]] = -1
            # Weighted
            p_wg3c_v[i] = <float>(wg3c)
            p_g_lcc_v[i] = <float>(p_wg3c_v[i] / (p_wg2c_v[i] * (p_g2c_v[i] - 1)))
            p_g_lcc_t_v[i] = p_g_lcc_v[i]
            g_gcc += p_g_lcc_v[i]


    for i in range(p_n):
        p_gwe_v[i] = <float>(0.5 * p_g2c_v[i] * (p_g2c_v[i] - 1) - p_g3c_v[i])
        p_wgwe_v[i] = <float>(p_wg2c_v[i] * (p_g2c_v[i] - 1) - p_wg3c_v[i])


    for i in range(p_n):
        if g_deg_v[i] != 0:
            g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
            G_ni_v = p_G_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            tg_ni_v = p_g_adj_v[p_G_cudeg_v[i]:p_G_cudeg_v[i + 1]]
            ig = 0
            iG = 0
            while (ig != g_deg_v[i] and iG != p_G_deg_v[i]):
                if g_ni_v[ig] == G_ni_v[iG]:
                    tg_ni_v[iG] = 1
                    ig += 1
                    iG += 1
                elif g_ni_v[ig] < G_ni_v[iG]:
                    ig += 1
                else:
                    iG += 1
    del g_adj
    return g_scc, G_scc, g_gcc, G_gcc