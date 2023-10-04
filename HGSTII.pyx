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
cpdef sparsification(str p_inpnet,
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
        cnp.ndarray[cnp.float32_t, ndim=1] e2c, e3c, ewe, G2c, G3c, Gwe, pfm_c2, pfm_c3, pfm_we, c2, c3, we, rpt, twe
        cnp.ndarray[cnp.int32_t, ndim=2] tg_edg, rd
        cnp.ndarray[cnp.int32_t, ndim=1] G_deg, G_cudeg, G_adj, G_wadj, G2c1, G3c1, Gwe1, tc2, tg_cudeg, tshift, tg_adj, tc3, aw, nd_seq
        cnp.ndarray[cnp.int8_t, ndim=1] g_adj, mk3, mkG, L, L_new
        # for memoryview
        cnp.float32_t[:] c2_v, c3_v, we_v, e2c_v, e3c_v, ewe_v, G2c_v, G3c_v, Gwe_v, twe_v
        cnp.int32_t[:, :] tg_edg_v, rd_v
        cnp.int32_t[:] G_deg_v, G_cudeg_v, G_adj_v, G_wadj_v, Gnn1_v, Gnn2_v, n12_v, Gni_v, tc2_v, tg_cudeg_v, tshift_v, tg_adj_v, tc3_v, mk_v, tg_ni_v, tg_nj_v, G_ni_v
        cnp.int8_t[:] g_adj_v, mk3_v, mkG_v, gnn1p_v, gnn2p_v, L_v, L_new_v, g_ni_v, gnip_v
        Py_ssize_t i=0, j=0, ni=0, nj=0, n_gnn12=0, ij=0, r=0, nn=0, mk=0, mk_ni=0, mk_nj=0, rp=0, u=0, v=0, ii=0, jj=0, itg=0, iG=0, ir=0
        cnp.float64_t gain=0, h=0, gc2_pre=0, gc3_pre=0, gwe_pre=0, gc2_new=0, gc3_new=0, gwe_new=0, cnt_e=0, wenn_new=0, we1_new=0, we2_new=0
        cnp.float32_t flag=0, c21_new=0, c22_new=0, c31_new=0, c32_new=0
        cnp.int32_t nd1=0, nd2=0, num=0, idx=0
        str INP_DEG = ''
        str INP_CUDEG = ''
        str INP_ADJ = ''
        str INP_WADJ = ''
        str INP_E23W = ''
        str OUP_GRAPH = ''
        str OUP_RT = ''
        object G, tg


    print(p_inpnet, p_n, p_edg_num, p_scaling, p_tolerance, p_method, p_oupnet, p_suffix)
    INP_DEG = f'{p_inpnet}_deg.npy'
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


    nd_seq = np.arange(p_n).astype(np.int32)
    rd = np.zeros((p_repeat, p_n), dtype=np.int32)
    for rp in range(p_repeat):
        np.random.shuffle(nd_seq)
        rd[rp, :] = nd_seq
    rd_v = rd


    rpt = np.zeros(p_repeat, dtype=np.float32)
    for rp in range(p_repeat):
        if p_suffix != 'No':
            OUP_GRAPH = f'{p_oupnet}_wgph_{str(rp)}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.graph'
        else:
            OUP_GRAPH = f'{p_oupnet}_wgph_{str(rp)}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.graph'


        # if not os.path.exists(OUP_GRAPH):
        # starting GAME core ===========================================
        # Step 1: initialize g with G
        G_deg = np.load(INP_DEG).astype(np.int32)
        G_cudeg = np.load(INP_CUDEG).astype(np.int32)
        G_adj = np.load(INP_ADJ).astype(np.int32)
        G_wadj = np.load(INP_WADJ).astype(np.int32)
        mk3 = np.zeros(p_n, dtype=np.int8)
        mkG = np.zeros(G_adj.shape[0], dtype=np.int8)
        G_deg_v = G_deg
        G_cudeg_v = G_cudeg
        G_adj_v = G_adj
        G_wadj_v = G_wadj
        mk3_v = mk3
        mkG_v = mkG
        # 2-cliques of the current g
        t1 = time()
        if 'Init' in p_suffix:
            G = nk.readGraph(f'{p_inpnet}_gph.graph', nk.Format.METIS)
            G.indexEdges()
            if 'LD' in p_suffix:
                tg = nk.sparsification.LocalDegreeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
            elif 'RES' in p_suffix:
                tg = nk.sparsification.RandomEdgeSparsifier().getSparsifiedGraphOfSize(G, p_scaling)
            elif 'RRES' in p_suffix:
                tg = nk.sparsification.LocalSparsifier(nk.sparsification.RandomEdgeSparsifier()).getSparsifiedGraphOfSize(G, p_scaling)
            # print('G: ', G.numberOfNodes(), G.numberOfEdges() * 2)
            # print('g: ', tg.numberOfNodes(), tg.numberOfEdges() * 2)
            del G
            tg_edg = np.zeros((tg.numberOfEdges(), 2), dtype=np.int32)
            tg_edg_v = tg_edg
            tc2 = np.zeros(p_n, dtype=np.int32)
            tc2_v = tc2
            idx = 0
            for u, v in tg.iterEdges():
                nd1 = <int>(u)
                nd2 = <int>(v)
                tg_edg_v[idx, 0] = nd1
                tg_edg_v[idx, 1] = nd2
                tc2_v[nd1] += 1
                tc2_v[nd2] += 1
                idx += 1
            del tg
            tg_edg = tg_edg[np.lexsort((tg_edg[:, 1],
                                        tg_edg[:, 0])).astype(np.int32)]
            tg_edg_v = tg_edg
            tg_cudeg = np.zeros(p_n + 1, dtype=np.int32)
            tg_cudeg_v = tg_cudeg
            num = 0
            for i in range(p_n + 1):
                tg_cudeg_v[i] = num
                if i < p_n:
                    num += tc2_v[i]
            tshift = tg_cudeg.copy()
            tshift_v = tshift
            tg_adj = np.zeros(np.sum(tc2_v), dtype=np.int32)
            tg_adj_v = tg_adj
            for idx in range(tg_edg_v.shape[0]):
                nd1, nd2 = tg_edg_v[idx][0], tg_edg_v[idx][1]
                tg_adj_v[tshift_v[nd1]] = nd2
                tg_adj_v[tshift_v[nd2]] = nd1
                tshift_v[nd1] += 1
                tshift_v[nd2] += 1
            del tg_edg
            tc3 = np.zeros(p_n, dtype=np.int32)
            tc3_v = tc3
            mk_v = np.zeros(p_n, dtype=np.int32) - 1
            for i in range(p_n):
                if tc2_v[i] > 1:
                    tg_ni_v = tg_adj_v[tg_cudeg_v[i]:tg_cudeg_v[i + 1]]
                    idx = 0
                    for ii in range(tc2_v[i]):
                        mk_v[tg_ni_v[ii]] = idx
                        idx += 1
                    for ii in range(tc2_v[i] - 1):
                        j = tg_ni_v[ii]
                        tg_nj_v = tg_adj_v[tg_cudeg_v[j]:tg_cudeg_v[j + 1]]
                        for jj in range(tc2_v[j]):
                            if mk_v[tg_nj_v[jj]] != -1:
                                idx = mk_v[tg_nj_v[jj]]
                                tc3_v[i] += 1
                        mk_v[j] = -1
                    mk_v[tg_ni_v[ii + 1]] = -1
            g_adj = np.zeros(G_adj.shape[0], dtype=np.int8)
            g_adj_v = g_adj
            for i in range(p_n):
                if tc2_v[i] != 0:
                    tg_ni_v = tg_adj_v[tg_cudeg_v[i]:tg_cudeg_v[i + 1]]
                    G_ni_v = G_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    g_ni_v = g_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    itg = 0
                    iG = 0
                    while (itg != tc2_v[i] and iG != G_deg_v[i]):
                        if tg_ni_v[itg] == G_ni_v[iG]:
                            g_ni_v[iG] = 1
                            itg += 1
                            iG += 1
                        elif tg_ni_v[itg] < G_ni_v[iG]:
                            itg += 1
                        else:
                            iG += 1
            del tg_adj
            twe = np.zeros(p_n, dtype=np.float32)
            twe_v = twe
            for i in range(p_n):
                twe_v[i] = <float>(0.5 * tc2_v[i] * (tc2_v[i] - 1) - tc3_v[i])
            c2 = tc2.copy().astype(np.float32)
            c3 = tc3.copy().astype(np.float32)
            we = twe.copy().astype(np.float32)
            c2_v = c2
            c3_v = c3
            we_v = we
            t2 = time() - t1
        else:
            g_adj = np.ones(G_adj.shape[0], dtype=np.int8)
            g_adj_v = g_adj
            c2 = G2c.copy().astype(np.float32)
            c3 = G3c.copy().astype(np.float32)
            we = Gwe.copy().astype(np.float32)
            c2_v = c2
            c3_v = c3
            we_v = we
            t2 = time() - t1
        n12_v = np.zeros(np.max(G_deg), dtype=np.int32)
        print("Initialization with G done!!!!!!!!!!!!!")


        # Step 2: update g
        L_new = np.ones(p_n, dtype=np.int8)
        L = np.ones(p_n, dtype=np.int8)
        L_new_v = L_new
        L_v = L
        # performance
        # changed edge number, Δ2, Δ3, Δwe, gain, time consumption
        PFM = np.zeros((6, p_n), dtype=np.float32)
        pfm_c2 = np.zeros(p_n, dtype=np.float32)
        pfm_c3 = np.zeros(p_n, dtype=np.float32)
        pfm_we = np.zeros(p_n, dtype=np.float32)
        r = 0
        t3 = time()
        while True:  # termination condition
            r += 1
            L_v[:] = L_new_v[:]
            L_new_v[:] = 0
            mkG_v[:] = 1
            cnt_e = 0
            for ird in range(p_n):
                i = rd_v[rp, ird]
                if L_v[i] == 1 and G_deg_v[i] != 0:
                    for ni in range(G_deg_v[i]):      
                        gnn1p_v = g_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                        Gnn1_v = G_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                        if ni == 0:  # reduce computational cost
                            # mk3[Gnn1[gnn1p == 1]] = 1
                            for mk in range(G_deg_v[i]):
                                if gnn1p_v[mk] == 1:
                                    mk3_v[Gnn1_v[mk]] = 1

                        # to make sure each round, iterating each edge only once
                        if mkG_v[G_cudeg_v[i] + ni] == 1:
                            mk_ni = ni
                            j = Gnn1_v[ni]
                            Gnn2_v = G_adj_v[G_cudeg_v[j]:G_cudeg_v[j + 1]]
                            gnn2p_v = g_adj_v[G_cudeg_v[j]:G_cudeg_v[j + 1]]

                            mkG_v[G_cudeg_v[i] + ni] = 0
                            for nj in range(G_deg_v[j]):
                                if Gnn2_v[nj] == i:
                                    mk_nj = nj
                                    mkG_v[G_cudeg_v[j] + nj] = 0
                                    break

                            n_gnn12 = 0
                            for nj in range(G_deg_v[j]):
                                if gnn2p_v[nj] == 1 and mk3_v[Gnn2_v[nj]] == 1:
                                    n12_v[n_gnn12] = Gnn2_v[nj]
                                    n_gnn12 += 1
                        else:
                            continue
                        
                        # gain of e by retaining its strategy ===============
                        gc2_pre = 0
                        gc3_pre = 0
                        gwe_pre = 0
                        gc2_new = 0
                        gc3_new = 0
                        gwe_new = 0
                        h = 0
                        # 2-cliques
                        h = abs((c2_v[i] - e2c_v[i]) / G2c_v[i])
                        gc2_pre += h
                        h = abs((c2_v[j] - e2c_v[j]) / G2c_v[j])
                        gc2_pre += h
                        # 3-cliques
                        if G3c_v[i] != 0:
                            h = abs((c3_v[i] - e3c_v[i]) / G3c_v[i])
                            gc3_pre += h
                        if G3c_v[j] != 0:
                            h = abs((c3_v[j] - e3c_v[j]) / G3c_v[j])
                            gc3_pre += h
                        for ij in range(n_gnn12):
                            nn = n12_v[ij]
                            if G3c_v[nn] != 0:
                                h = abs((c3_v[nn] - e3c_v[nn]) / G3c_v[nn])
                                gc3_pre += h
                        # wedges
                        if Gwe_v[i] != 0:
                            h = abs((we_v[i] - ewe_v[i]) / Gwe_v[i])
                            gwe_pre += h
                        if Gwe_v[j] != 0:
                            h = abs((we_v[j] - ewe_v[j]) / Gwe_v[j])
                            gwe_pre += h
                        for ij in range(n_gnn12):
                            nn = n12_v[ij]
                            if Gwe_v[nn] != 0:
                                h = abs((we_v[nn] - ewe_v[nn]) / Gwe_v[nn])
                                gwe_pre += h

                        # gain of e by changing its strategy ================
                        # 2-cliques
                        if gnn1p_v[mk_ni] == 1:  # e in current g
                            flag = -1  # to remove
                        else:
                            flag = 1  # to insert
                        # 2-cliques
                        c21_new = c2_v[i] + flag
                        c22_new = c2_v[j] + flag
                        h = abs((c21_new- e2c_v[i]) / G2c_v[i])
                        gc2_new += h
                        h = abs((c22_new - e2c_v[j]) / G2c_v[j])
                        gc2_new += h
                        # 3-cliques
                        c31_new = c3_v[i] + flag * n_gnn12
                        c32_new = c3_v[j] + flag * n_gnn12
                        if G3c_v[i] != 0:
                            h = abs((c31_new - e3c_v[i]) / G3c_v[i])
                            gc3_new += h
                        if G3c_v[j] != 0:
                            h = abs((c32_new - e3c_v[j]) / G3c_v[j])
                            gc3_new += h
                        for ij in range(n_gnn12):
                            nn = n12_v[ij]
                            if G3c_v[nn] != 0:
                                h = abs((c3_v[nn] + flag - e3c_v[nn]) / G3c_v[nn])
                                gc3_new += h
                        # wedges
                        we1_new = c21_new * (c21_new - 1) / 2 - c31_new
                        we2_new = c22_new * (c22_new - 1) / 2 - c32_new
                        if Gwe_v[i] != 0:
                            h = abs((we1_new - ewe_v[i]) / Gwe_v[i])
                            gwe_new += h
                        if Gwe_v[j] != 0:
                            h = abs((we2_new - ewe_v[j]) / Gwe_v[j])
                            gwe_new += h
                        for ij in range(n_gnn12):
                            nn = n12_v[ij]
                            wenn_new = c2_v[nn] * (c2_v[nn] - 1) / 2 - (c3_v[nn] + flag)
                            if Gwe_v[nn] != 0:
                                h = abs((wenn_new - ewe_v[nn]) / Gwe_v[nn])
                                gwe_new += h

                        # the final choice and updating =====================
                        # g_adj_v, c2_v, c3_v, we_v, L_new_v
                        if p_method == 'GMN23':
                            gain = gc2_pre + gc3_pre - gc2_new - gc3_new
                        elif p_method == 'GMN23w':
                            gain = gc2_pre + gc3_pre + gwe_pre - gc2_new - gc3_new - gwe_new
                        elif p_method == 'GMN2':
                            gain = gc2_pre - gc2_new
                        elif p_method == 'GMN3':
                            gain = gc3_pre - gc3_new
                        elif p_method == 'GMN2w':
                            gain = gc2_pre + gwe_pre - gc2_new - gwe_new
                        elif p_method == 'GMN3w':
                            gain = gc3_pre + gwe_pre - gc3_new - gwe_new
                        else:
                            print('Error !!!!!!!!!!!!!!!!!!!!!!!')
                        if gain > 0:
                            cnt_e += 1
                            L_new_v[i] = 1
                            L_new_v[j] = 1
                            for ij in range(n_gnn12):
                                nn = n12_v[ij]
                                L_new_v[nn] = 1
                            # g_adj_v
                            if flag == -1:  # to remove
                                g_adj_v[G_cudeg_v[i] + mk_ni] = 0
                                g_adj_v[G_cudeg_v[j] + mk_nj] = 0
                                mk3_v[j] = 0
                            if flag == 1:  # to insert
                                g_adj_v[G_cudeg_v[i] + mk_ni] = 1
                                g_adj_v[G_cudeg_v[j] + mk_nj] = 1
                                mk3_v[j] = 1
                            # 2-cliques                    
                            c2_v[i] = c21_new
                            c2_v[j] = c22_new
                            # 3-cliques
                            c3_v[i] = c31_new
                            c3_v[j] = c32_new
                            for ij in range(n_gnn12):
                                nn = n12_v[ij]
                                if G3c_v[nn] != 0:
                                    c3_v[nn] = c3_v[nn] + flag
                            # wedges
                            we_v[i] = <float>(we1_new)
                            we_v[j] = <float>(we2_new)
                            for ij in range(n_gnn12):
                                nn = n12_v[ij]
                                if Gwe_v[nn] != 0:
                                    we_v[nn] = <float>(c2_v[nn] * (c2_v[nn] - 1) / 2 - (c3_v[nn] + flag))
                    # mk3[Gnn1[gnn1p == 1]] = 0
                    for mk in range(G_deg_v[i]):
                        if gnn1p_v[mk] == 1:
                            mk3_v[Gnn1_v[mk]] = 0
            # performance
            # changed edge number, Δ2, Δ3, Δwe
            print(r, cnt_e, time() - t3 + t2)
            pfm_c2[G2c1] = np.absolute(c2 - e2c)[G2c1] / G2c[G2c1]
            pfm_c3[G3c1] = np.absolute(c3 - e3c)[G3c1] / G3c[G3c1]
            pfm_we[Gwe1] = np.absolute(we - ewe)[Gwe1] / Gwe[Gwe1]
            PFM[0, r] = <float>cnt_e
            PFM[5, r] = <float>(time() - t3 + t2)
            if p_method == 'GMN2':
                print('C2:', np.mean(pfm_c2))
                PFM[4, r] = np.mean(pfm_c2)
            if p_method == 'GMN3':
                print('C3:', np.mean(pfm_c3))
                PFM[4, r] = np.mean(pfm_c3)
            if p_method == 'GMN23':
                print('C23:', np.mean(pfm_c2 + pfm_c3))
                PFM[4, r] = np.mean(pfm_c2 + pfm_c3)
            if p_method == 'GMN3w':
                print('C3w:', np.mean(pfm_c3 + pfm_we))
                PFM[4, r] = np.mean(pfm_c3 + pfm_we)
            if p_method == 'GMN23w':
                print('C23WE:', np.mean(pfm_c2 + pfm_c3 + pfm_we))
                PFM[4, r] = np.mean(pfm_c2 + pfm_c3 + pfm_we)
            if r >= 2 and (PFM[4, r - 1] - PFM[4, r]) <= p_tolerance:
                PFM[1, :] = pfm_c2
                PFM[2, :] = pfm_c3
                PFM[3, :] = pfm_we
                break
        rpt[rp] = <float>(time() - t3 + t2)
        print(time() - t3 + t2)
        print(int(np.sum(c2.astype(np.int32)) / 2) == int(np.sum(g_adj.astype(np.int32)) / 2))
        with open(OUP_GRAPH, 'w', newline='', encoding='UTF-8') as gph:
            gph.write(str(p_n) + ' ' + str(int(np.sum(c2.astype(np.int32)) / 2)) + ' ' + str(1) + '\n')
        with open(OUP_GRAPH, 'a', newline='', encoding='UTF-8') as gph:
            for i in range(p_n):
                if c2_v[i] != 0:
                    Gni_v = G_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    Gwni_v = G_wadj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    gni_v = g_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                    aw = np.vstack([np.asarray(Gni_v)[np.asarray(gni_v) == 1] + 1, np.asarray(Gwni_v)[np.asarray(gni_v) == 1]]).T.flatten().astype(np.int32)
                    np.savetxt(gph, aw, fmt='%d', newline=' ')
                gph.write('\n')
    np.save(OUP_RT, np.array([np.mean(rpt), np.std(rpt)]))