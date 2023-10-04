# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
from os import system as ossys
from build.Network import net as NETWORK
from build.HGSTI import expectation as HGST_STAGEI
from build.HGSTII import sparsification as HGST_STAGEII


# Compilation ====================================
ossys("python Setup.py build_ext --build-lib build")


# Sparsification =================================
# parameters =====================================
# input network path
inpnet = './Example/10000_LFR_5025025250_01US'
# number of nodes
n = 10000
# number of edges
m = 252923
# sparsification ratio
scaling = 0.5
# by default, 0.01 for an earlier convergence
tolerance = 0.01
# '2' for degree, '3' for 'triangle', and 'w' for 'wedge'
# by default, both 'GMN23' and 'GMN23w' are fine
method = 'GMN23'
# output network path
oupnet = './Example/10000_LFR_5025025250_01US'
# the number of sparse subgraph to be created
# by default, creating just 1 sparse subgraph
repeat = 1
# different initialization strategies:
# 'No': the original GST in https://ieeexplore.ieee.org/document/10068651
# 'No_LDInit': initialized with 'LocalDegreeSparsifier' from NETWORKIT
# 'No_RESInit': initialized with 'RandomEdgeSparsifier' from NETWORKIT
# 'No_RRESInit': initialized with 'LocalSparsifier(RandomEdgeSparsifier)' from NETWORKIT
# by default, both 'No_RESInit' and 'No_RRESInit' are fine
suffix = 'No_RESInit'  #
# ================================================
# network preparation
NETWORK(inpnet,
        int(n),
        int(m))
# core
HGST_STAGEI(inpnet,
            int(n),
            int(m),
            scaling,
            suffix)
HGST_STAGEII(inpnet,
             int(n),
             int(m),
             scaling,
             tolerance,
             method,
             oupnet,
             int(repeat),
             suffix)
