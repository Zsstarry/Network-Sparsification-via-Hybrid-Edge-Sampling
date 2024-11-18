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
cpdef expectation(
str p_inpnet,
Py_ssize_t p_n,
Py_ssize_t p_edg_num,
cnp.float64_t p_scaling,
str p_suffix):

    cdef:
        cnp.ndarray[cnp.int32_t, ndim=1] G_deg, G_wdeg, G_cudeg, G_adj, G_wadj
        cnp.ndarray[cnp.float32_t, ndim=1] e2c, e3c, ewe, G2c, G3c, Gwe
        # for memoryview
        cnp.float32_t[:] e2c_v, e3c_v, ewe_v, G2c_v, G3c_v, Gwe_v
        cnp.int32_t[:] G_deg_v, G_wdeg_v, G_cudeg_v, G_adj_v, G_wadj_v, mk_v
        Py_ssize_t i=0, edg_num=0
        cnp.float64_t sigma = 0, sigmat = 0, G_gcc = 0, e_gcc = 0, e_gcct = 0
        str INP_DEG = ''
        str INP_WDEG = ''
        str INP_CUDEG = ''
        str INP_ADJ = ''
        str INP_WADJ = ''
        str OUP_E23W = ''
        str OUP_RT = ''


    print(p_inpnet, p_n, p_edg_num, p_scaling, p_suffix)
    INP_DEG = f'{p_inpnet}_deg.npy'
    INP_WDEG = f'{p_inpnet}_wdeg.npy'
    INP_CUDEG = f'{p_inpnet}_cudeg.npy'
    INP_ADJ = f'{p_inpnet}_adj.npy'
    INP_WADJ = f'{p_inpnet}_wadj.npy'
    if p_suffix != 'No':
        OUP_E23W = f'{p_inpnet}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_e23w_G23w_CyO_{p_suffix}.npy'
        OUP_RT = f'{p_inpnet}_rt_FStageI_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_RT_23 = f'{p_inpnet}_rt_FStageI23_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
    else:
        OUP_E23W = f'{p_inpnet}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_e23w_G23w_CyO.npy'
        OUP_RT = f'{p_inpnet}_rt_FStageI_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_RT_23 = f'{p_inpnet}_rt_FStageI23_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'


    # node, node, similarity, confidence
    edg_num = np.load(f'{p_inpnet}_Udwp_1.npy').astype(np.float32).shape[0]
    if edg_num != p_edg_num:
        print("Input edge number wrong!")
        os._exit(0)


    # load
    G_deg = np.load(INP_DEG).astype(np.int32)
    G_cudeg = np.load(INP_CUDEG).astype(np.int32)
    G_adj = np.load(INP_ADJ).astype(np.int32)


    # the expected 2-cliques and 3-cliques (and wedges)
    # ===========================================================
    t1 = time()
    e2c = np.zeros(p_n, dtype=np.float32)
    e3c = np.zeros(p_n, dtype=np.float32)
    ewe = np.zeros(p_n, dtype=np.float32)
    G2c = np.zeros(p_n, dtype=np.float32)
    G3c = np.zeros(p_n, dtype=np.float32)
    Gwe = np.zeros(p_n, dtype=np.float32)
    e2c_v = e2c
    e3c_v = e3c
    ewe_v = ewe
    G2c_v = G2c
    G3c_v = G3c
    Gwe_v = Gwe
    G_deg_v = G_deg
    G_cudeg_v = G_cudeg
    G_adj_v = G_adj
    if 'No' in p_suffix:
        sigma = p_scaling * p_scaling * p_scaling
    elif '3Dpdt' in p_suffix:
        sigma = 0.5 * (p_scaling * p_scaling * p_scaling + p_scaling)
    elif 'M3Dpdt' in p_suffix:
        sigma = p_scaling
    elif 'GDCC' in p_suffix:
        print('continue')
    else:
        print(p_suffix + 'is not defined!')
    # 2-cliques
    for i in range(p_n):
        e2c_v[i] = <float>(G_deg_v[i] * p_scaling)
        G2c_v[i] = <float>(G_deg_v[i])
    print('2-cliques done!!!!!!!!!!!!')
    # 3-cliques
    mk_v = np.zeros(p_n, dtype=np.int32) - 1
    for i in range(p_n):
        compute_G3c(i,
                    G_deg_v,
                    G_cudeg_v,
                    G_adj_v,
                    mk_v,
                    G3c_v)
        e3c_v[i] = <float>(sigma * G3c_v[i])
    print('3-cliques done!!!!!!!!!!!!')
    t2 = time() - t1

        # testing =====================
    if 'GDCC' in p_suffix:
        G_gcc = 0
        for i in range(p_n):
            if G2c_v[i] > 1:
                G_gcc += <float>(2 * G3c_v[i] / (G2c_v[i] * (G2c_v[i] - 1)))
        for sigmat in np.linspace(p_scaling ** 3, p_scaling, 50):
            e_gcct = 0
            for i in range(p_n):
                e3c_v[i] = <float>(sigmat * G3c_v[i])
                if e2c_v[i] > 1:
                    e_gcct += <float>(2 * e3c_v[i] / (e2c_v[i] * (e2c_v[i] - 1)))
            if abs(e_gcct - G_gcc) <= abs(e_gcc - G_gcc):
                e_gcc = e_gcct
                sigma = sigmat
        print('Average:', 0.5 * (p_scaling * p_scaling * p_scaling + p_scaling), sigma)
        # wedges
        for i in range(p_n):
            if G_deg_v[i] > 1:
                ewe_v[i] = <float>(0.5 * p_scaling * p_scaling * G_deg_v[i] * (G_deg_v[i] - 1) - e3c_v[i])
                Gwe_v[i] = <float>(0.5 * G_deg_v[i] * (G_deg_v[i] - 1) - G3c_v[i])
        print('wedges done!!!!!!!!!!!!')
        # testing =====================
    else:
        # wedges
        for i in range(p_n):
            if G_deg_v[i] > 1:
                ewe_v[i] = <float>(0.5 * p_scaling * p_scaling * G_deg_v[i] * (G_deg_v[i] - 1) - sigma * G3c_v[i])
                Gwe_v[i] = <float>(0.5 * G_deg_v[i] * (G_deg_v[i] - 1) - G3c_v[i])
        print('wedges done!!!!!!!!!!!!')
    t3 = time() - t1
    np.save(OUP_E23W,
            np.array([e2c, e3c, ewe, G2c, G3c, Gwe]).T.astype(np.float32))
    np.save(OUP_RT, np.array([<float>t3], dtype=np.float32))
    np.save(OUP_RT_23, np.array([<float>t2], dtype=np.float32))


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True) 
@cy.profile(True)
cpdef inline void compute_G3c(
Py_ssize_t p_i,
cnp.int32_t[:] p_deg,
cnp.int32_t[:] p_cudeg,
cnp.int32_t[:] p_adj,
cnp.int32_t[:] p_mk,
cnp.float32_t[:] p_G3c) noexcept nogil:

    cdef:
        Py_ssize_t ii=0, jj=0, j=0, k=0
        cnp.int32_t[:] ni, nj
        cnp.float64_t Gc3=0

    if p_deg[p_i] > 1:
        ni = p_adj[p_cudeg[p_i]:p_cudeg[p_i + 1]]
        for ii in range(p_deg[p_i]):
            p_mk[ni[ii]] = ni[ii]
        Gc3 = 0
        for ii in range(p_deg[p_i] - 1):
            j = ni[ii]
            nj = p_adj[p_cudeg[j]:p_cudeg[j + 1]]
            for jj in range(p_deg[j]):
                k = nj[jj]
                if p_mk[k] != -1:
                    Gc3 += 1
            p_mk[j] = -1
        p_mk[ni[ii + 1]] = -1
        p_G3c[p_i] = <float>Gc3