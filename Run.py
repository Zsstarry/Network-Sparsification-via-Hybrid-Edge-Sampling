# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
from os import system as ossys


# Compilation ====================================
ossys("python Setup.py build_ext --build-lib build")


from build.Network import net as NETWORK
from build.OGSTI import expectation as OGST_STAGEI
from build.OGSTII import sparsification as OGST_STAGEII
from build.OGSTII_Repeat import sparsification as OGST_STAGEII_Repeat


# Sparsification parameters ======================
#  =====================================
# input network
inpnet = "./Example/10000_LFR_5025025250_01US"
# network nodes
n = 10000
# network edges
m = 252923
# sparsification ratio
scaling = 0.5
# by default, 0.01 for an earlier convergence
tolerance = 0.01
# '2' for degree, '3' for 'triangle', and 'w' for 'wedge';
# by default, using 'GMN23w' see from "Generic Network Sparsification via Hybrid Edge Sampling"
method = "GMN23w"
# output network
oupnet = "./Example/10000_LFR_5025025250_01US"
# the number of sparse subgraph to be created; by default, using 1
repeat = 1
# by default, the current best algorithmic variant;
# With improved initialization by using LocalSparsifier(RandomEdgeSparsifier) from NETWORKIT, and
# With constrainted update by preserving the largest connected component and the weighted average clustering coefficient
suffix = "No_RESInit_SWG"  #
# ================================================
# network preparation
NETWORK(inpnet, int(n), int(m))
# core
OGST_STAGEI(inpnet, int(n), int(m), scaling, suffix)
OGST_STAGEII_Repeat(
    inpnet, int(n), int(m), scaling, tolerance, method, oupnet, int(repeat), suffix
)
