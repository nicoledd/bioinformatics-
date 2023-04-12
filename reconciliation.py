#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 2021

@author: Palash Sashittal
"""

import gurobipy as gp
import numpy as np
import pandas as pd
import networkx as nx
import itertools

# parsimonious clone reconciliation problem
class solvePCR:

    def __init__(self, snv_mat, cna_mat, threads = 1, timelimit = None, verbose = True):
        self.snv_mat = snv_mat
        self.cna_mat = cna_mat
        self.threads = threads
        self.timelimit = timelimit
        self.verbose = verbose

        self.nsamples = self.snv_mat.shape[1]
        self.sol_clones = None
        self.sol_props = None

    def solve(self):

        nsamples = self.snv_mat.shape[1]
        assert nsamples == self.cna_mat.shape[1], 'SNV and CNA matrix sizes do not match up.'

        nsnv = self.snv_mat.shape[0]
        ncna = self.cna_mat.shape[0]

        model = gp.Model('solvePCR')
        x = model.addVars(nsnv, ncna, vtype=gp.GRB.BINARY, name='x')
        w = model.addVars(nsamples, nsnv, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'w')
        y = model.addVars(nsamples, nsnv, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'y')
        d_snv = model.addVars(nsamples, nsnv, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'delta_snv')
        d_cna = model.addVars(nsamples, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'delta_cna')

        xsum = gp.LinExpr()
        for i in range(nsnv):
            for j in range(ncna):
                xsum += x[i,j]
        model.addConstr(xsum == nsnv+ncna-1)

        # additional constraint -- must have at least one clone from each type
        snvSum = gp.LinExpr()
        for i in range(nsnv):
            for j in range(ncna):
                snvSum += x[i,j]
            model.addConstr(snvSum >= 1)
            snvSum.clear()
        cnaSum = gp.LinExpr()
        for j in range(ncna):
            for i in range(nsnv):
                cnaSum += x[i,j]
            model.addConstr(cnaSum >= 1)
            cnaSum.clear()

        # encode product w[i,j,k] = y[i,j,k] * x[j,k]
        for i in range(nsamples):
            for j in range(nsnv):
                for k in range(ncna):
                    model.addConstr(w[i,j,k] <= y[i,j,k])
                    model.addConstr(w[i,j,k] <= x[j,k])
                    model.addConstr(w[i,j,k] >= x[j,k] + y[i,j,k] - 1)

        # encode abundance constraint for snv with correction
        # TODO modify code
        '''
        for i in range(nsamples):
            for j in range(nsnv):
                sum = gp.LinExpr()
                for k in range(ncna):
                    sum += w[i,j,k]
                model.addConstr(sum == self.snv_mat[j,i])
        '''
        for i in range(nsamples):
            for j in range(nsnv):
                sum = gp.LinExpr()
                for k in range(ncna):
                    sum += w[i,j,k]
                model.addConstr(self.snv_mat[j,i] - sum <= d_snv[i,j])
                model.addConstr(sum - self.snv_mat[j,i] <= d_snv[i,j])

        # encode abundance constraint for cna with correction
        # TODO modify code
        '''
        for i in range(nsamples):
            for k in range(ncna):
                sum = gp.LinExpr()
                for j in range(nsnv):
                    sum += w[i,j,k]
                model.addConstr(sum == self.cna_mat[k,i])
        '''
        for i in range(nsamples):
            for k in range(ncna):
                sum = gp.LinExpr()
                for j in range(nsnv):
                    sum += w[i,j,k]
                model.addConstr(self.cna_mat[k,i] - sum <= d_cna[i,k])
                model.addConstr(sum - self.cna_mat[k,i] <= d_cna[i,k])

        # encode total abundance constraint
        for i in range(nsamples):
            sum = gp.LinExpr()
            for j in range(nsnv):
                for k in range(ncna):
                    sum += w[i,j,k]
            model.addConstr(sum == 1)

        # set objective function
        # TODO modify code

        '''
        obj_sum = gp.LinExpr()
        for j in range(nsnv):
            for k in range(ncna):
                obj_sum += x[j,k]
        model.setObjective(obj_sum, gp.GRB.MINIMIZE)
        '''

        obj_sum = gp.LinExpr()
        for i in range(nsamples):
            for j in range(nsnv):
                obj_sum += d_snv[i,j]
            for k in range(ncna):
                obj_sum += d_cna[i,k]
        model.setObjective(obj_sum, gp.GRB.MINIMIZE)

        model.setParam(gp.GRB.Param.Threads, self.threads)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            solx = model.getAttr('x', x)
            self.sol_clones = [key for key, val in solx.items() if val >= 0.5]
            self.sol_props = model.getAttr('x', w)

    def writeCloneFile(self, clone_file, snv_clones = None, cna_clones = None):
        clone_data = []
        for clone in self.sol_clones:
            #for sample in range(self.nsamples):
            if snv_clones:
                snv_clone = snv_clones[clone[0]]
            else:
                snv_clone = clone[0]
            if cna_clones:
                cna_clone = cna_clones[clone[1]]
            else:
                cna_clone = clone[1]

            clone_data.append([clone, snv_clone, cna_clone] + [self.sol_props[sample, clone[0], clone[1]] for sample in range(self.nsamples)])
        df_clone = pd.DataFrame(clone_data, columns=['clone', 'snv_clone', 'cna_clone'] + [f'sample_{idx}' for idx in range(self.nsamples)])
        df_clone.to_csv(clone_file, sep='\t', index=False)
















# minimum correction tree parsimonious clone reconciliation problem
class solveMCTPCR:

    # initializing function
    def __init__(self, snv_mat, cna_mat, snv_edges, cna_edges, threads = 1, timelimit = None, verbose = True):

        # variables to store the snv and cna proportions (given as params)
        self.snv_mat = snv_mat
        self.cna_mat = cna_mat

        # variables to store the final minimized objective value 
        # and predicted clones
        self.objVal = None
        self.objClones = None

        # variables to store the snv and cna trees in DiGraph format
        # (the edges were given as params)
        G = nx.DiGraph()
        G.add_edges_from(snv_edges)
        self.snv_dag = nx.algorithms.transitive_closure_dag(G)
        G.clear()
        G.add_edges_from(cna_edges)
        self.cna_dag = nx.algorithms.transitive_closure_dag(G)

        # variables to store number of threads and time limit
        self.threads = threads
        self.timelimit = timelimit
        self.verbose = verbose

        # variables to store number of samples, and number of snv and cna clone types
        self.nsamples = self.snv_mat.shape[1]
        self.nsnv = self.snv_mat.shape[0]
        self.ncna = self.cna_mat.shape[0]

        # variables that are used as props
        self.sol_clones = None
        self.sol_props = None

        # parent dictionaries for snv and cna trees
        # dict[child] = [parent A, parent B, ...]
        # but why is this necessary? (to identify root nodes?)
        # Why would a child have more than 1 parent?
        self.snv_parent_dict = {}
        for edge in snv_edges:
            child = edge[1]
            parent = edge[0]
            if child not in self.snv_parent_dict.keys():
                self.snv_parent_dict[child] = [parent]
            else:
                self.snv_parent_dict[child].append(parent)
        self.cna_parent_dict = {}
        for edge in cna_edges:
            child = edge[1]
            parent = edge[0]
            if child not in self.cna_parent_dict.keys():
                self.cna_parent_dict[child] = [parent]
            else:
                self.cna_parent_dict[child].append(parent)

        # variables to store snv and cna tree edges (given as params)
        self.snv_edges = snv_edges
        self.cna_edges = cna_edges

        # variables to identify the root nodes of the snv and cna trees
        for j in range(self.nsnv):
            if j not in self.snv_parent_dict.keys():
                self.snv_root = j
                break
        for k in range(self.ncna):
            if k not in self.cna_parent_dict.keys():
                self.cna_root = k
                break




    def solve(self):

        # variable to store the number of samples
        nsamples = self.snv_mat.shape[1]

        # the number of samples must be the same for the cna and the snv proportions
        assert nsamples == self.cna_mat.shape[1], 'SNV and CNA matrix sizes do not match up.'

        # variables to store the number of snv and cna clone types
        nsnv = self.snv_mat.shape[0]
        ncna = self.cna_mat.shape[0]

        # create linear programming model
        model = gp.Model('solveMCTPCR')

        # x - binary variables which encode whether each clone (in the final set) exists or not
        x = model.addVars(nsnv, ncna, vtype=gp.GRB.BINARY, name='x')

        # w, y - continuous variables
        w = model.addVars(nsamples, nsnv, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'w')
        y = model.addVars(nsamples, nsnv, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'y')

        # z - continuous variables which encode the PREDICTED edges in the cna and snv trees
        z_snv = model.addVars(nsnv-1, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'z_snv')
        z_cna = model.addVars(nsnv, ncna-1, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'z_cna')

        # d - continuous variables which encode the difference between input and predicted proportions
        d_snv = model.addVars(nsamples, nsnv, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'delta_snv')
        d_cna = model.addVars(nsamples, ncna, vtype = gp.GRB.CONTINUOUS, lb = 0, ub = 1, name = 'delta_cna')

        # encode product w[i,j,k] = y[i,j,k] * x[j,k]
        for i in range(nsamples):
            for j in range(nsnv):
                for k in range(ncna):
                    model.addConstr(w[i,j,k] <= y[i,j,k])
                    model.addConstr(w[i,j,k] <= x[j,k])
                    model.addConstr(w[i,j,k] >= x[j,k] + y[i,j,k] - 1)

        # encode abundance constraint for snv with correction
        for i in range(nsamples):
            for j in range(nsnv):
                sum = gp.LinExpr()
                for k in range(ncna):
                    sum += w[i,j,k]
                model.addConstr(self.snv_mat[j,i] - sum <= d_snv[i,j])
                model.addConstr(sum - self.snv_mat[j,i] <= d_snv[i,j])

        # encode abundance constraint for cna
        for i in range(nsamples):
            for k in range(ncna):
                sum = gp.LinExpr()
                for j in range(nsnv):
                    sum += w[i,j,k]
                model.addConstr(self.cna_mat[k,i] - sum <= d_cna[i,k])
                model.addConstr(sum - self.cna_mat[k,i] <= d_cna[i,k])

        # encode total abundance constraint
        for i in range(nsamples):
            sum = gp.LinExpr()
            for j in range(nsnv):
                for k in range(ncna):
                    sum += w[i,j,k]
            model.addConstr(sum == 1)

        '''
        # multiplication constraints: If both endpoints exist, edge also exists
        # 1) encode z_snv[j,k] = x[parent(j), k] * x[j,k]
        # 2) encode z_cna[j,k] = x[j, parent(k)] * x[j,k]
        for edge_idx, edge in enumerate(self.snv_edges):
            parent = edge[0]
            child = edge[1]
            for k in range(ncna):
                model.addConstr(z_snv[edge_idx, k] <= x[parent, k])
                model.addConstr(z_snv[edge_idx, k] <= x[child, k])
                model.addConstr(z_snv[edge_idx, k] >= x[parent, k] + x[child, k] - 1)
        for edge_idx, edge in enumerate(self.cna_edges):
            parent = edge[0]
            child = edge[1]
            for j in range(nsnv):
                model.addConstr(z_cna[j, edge_idx] <= x[j, parent])
                model.addConstr(z_cna[j, edge_idx] <= x[j, child])
                model.addConstr(z_cna[j, edge_idx] >= x[j, parent] + x[j, child] - 1)
        '''

        # summation of edges constraints: There exists only one edge of each type
        # encode sum_{k} z_snv[j, k] == 1
        # encode sum_{j} z_cna[j, k] == 1
        for edge_idx in range(nsnv-1):
            summ = gp.LinExpr()
            for k in range(ncna):
                summ += z_snv[edge_idx, k]
            model.addConstr(summ == 1)
        for edge_idx in range(ncna - 1):
            summ = gp.LinExpr()
            for j in range(nsnv):
                summ += z_cna[j, edge_idx]
            model.addConstr(summ == 1)

        # the non-edge constraint: every non-edge of T1 (and T2) must be absent in T
        # generate all possible edges of T1 so that allEdges =[(u,v) (u,v) ...]
        # filter out allEdges to turn it into nonEdges -- these shouldn't be in T!
        # initiate summ variable
        # for every cna clone, j = 0 to ncna:
            # for every edge u,v that is in nonEdges:
                # summ += z_snv[u,v,j]
        # add this constraint to the model: summ should equal zero
        # do the same thing for T2
        # IS THIS ALREADY IMPLIED THOUGH?? UGH HOW DOES Z WORKK


        # the clone existence constraint: a clone can only exist if its parent clone exists
        # the exception is (0,0)
        # encode x[j,k] <= x[parent(j), k] + x[j, parent(k)]
        for j in range(nsnv):
            for k in range(ncna):
                if j != 0 or k != 0:
                    if j in self.snv_parent_dict.keys() or k in self.cna_parent_dict.keys():
                        sum = gp.LinExpr()
                        if j in self.snv_parent_dict.keys():
                            sum += x[self.snv_parent_dict[j][0], k]
                        if k in self.cna_parent_dict.keys():
                            sum += x[j, self.cna_parent_dict[k][0]]
                        model.addConstr(x[j,k] <= sum)

        # set objective function
        # we want to minimize the sum of differences
        # between the input and output proportions
        obj_sum = gp.LinExpr()
        for i in range(nsamples):
            for j in range(nsnv):
                obj_sum += d_snv[i,j]
            for k in range(ncna):
                obj_sum += d_cna[i,k]
        model.setObjective(obj_sum, gp.GRB.MINIMIZE)

        # set the number of threads used
        model.setParam(gp.GRB.Param.Threads, self.threads)

        # run the model to obtain the prediction
        # store results in class variables
        model.optimize()
        if model.status == gp.GRB.OPTIMAL:
            solx = model.getAttr('x', x)
            self.sol_clones = [key for key, val in solx.items() if val >= 0.5]
            self.sol_props = model.getAttr('x', w)

        # store the computed objective value
        self.objVal = model.objVal



    def getObjVal(self):
        return self.objVal

    def writeCloneFile(self, clone_file, snv_clones = None, cna_clones = None, writeMode='w'):
        clone_data = []
        for clone in self.sol_clones:
            #for sample in range(self.nsamples):
            if snv_clones:
                snv_clone = snv_clones[clone[0]]
            else:
                snv_clone = clone[0]
            if cna_clones:
                cna_clone = cna_clones[clone[1]]
            else:
                cna_clone = clone[1]

            clone_data.append([clone, snv_clone, cna_clone] + [self.sol_props[sample, clone[0], clone[1]] for sample in range(self.nsamples)])
        df_clone = pd.DataFrame(clone_data, columns=['clone', 'snv_clone', 'cna_clone'] + [f'sample_{idx}' for idx in range(self.nsamples)])
        df_clone.to_csv(clone_file, sep='\t', index=False, mode=writeMode)

    def writeCloneTree(self, clone_tree_file, snv_clones = None, cna_clones = None, writeMode='w'):
        clone_edges = []
        for clone1, clone2 in itertools.permutations(self.sol_clones, 2):
            if snv_clones:
                snv_clone1 = snv_clones[clone1[0]]
                snv_clone2 = snv_clones[clone2[0]]
            else:
                snv_clone1 = clone1[0]
                snv_clone2 = clone2[0]

            if cna_clones:
                cna_clone1 = cna_clones[clone1[1]]
                cna_clone2 = cna_clones[clone2[1]]
            else:
                cna_clone1 = clone1[1]
                cna_clone2 = clone2[1]

            if clone1[0] == clone2[0]:
                if clone1[1] in self.cna_parent_dict.keys():
                    if clone2[1] in self.cna_parent_dict[clone1[1]]:
                        clone_edges.append(((snv_clone2, cna_clone2), (snv_clone1, cna_clone1)))

            if clone1[1] == clone2[1]:
                if clone1[0] in self.snv_parent_dict.keys():
                    if clone2[0] in self.snv_parent_dict[clone1[0]]:
                        clone_edges.append(((snv_clone2, cna_clone2), (snv_clone1, cna_clone1)))

        with open(clone_tree_file, writeMode) as output:
            output.write('tree')
            for clone_edge in clone_edges:
                output.write(f'{clone_edge[0]}\t{clone_edge[1]}\n')

    def writeDOT(self, dot_file, snv_clones = None, cna_clones = None):

        with open(dot_file, 'w') as output:

            output.write(f'digraph N {{\n')
            output.write(f"\toverlap=\"false\"\n")
            output.write(f"\trankdir=\"TB\"\n")

            idx_dict = {}
            idx = 0
            for clone in self.sol_clones:
                if snv_clones:
                    snv_clone = snv_clones[clone[0]]
                else:
                    snv_clone = clone[0]
                if cna_clones:
                    cna_clone = cna_clones[clone[1]]
                else:
                    cna_clone = clone[1]

                idx_dict[clone] = idx
                output.write(f'\t{idx} [label=\"{snv_clone}, {cna_clone}\", style=\"bold\"];\n')

                idx += 1

            for clone1, clone2 in itertools.permutations(self.sol_clones, 2):
                if snv_clones:
                    snv_clone1 = snv_clones[clone1[0]]
                    snv_clone2 = snv_clones[clone2[0]]
                else:
                    snv_clone1 = clone1[0]
                    snv_clone2 = clone2[0]

                if cna_clones:
                    cna_clone1 = cna_clones[clone1[1]]
                    cna_clone2 = cna_clones[clone2[1]]
                else:
                    cna_clone1 = clone1[1]
                    cna_clone2 = clone2[1]

                if clone1[0] == clone2[0]:
                    if clone1[1] in self.cna_parent_dict.keys():
                        if clone2[1] in self.cna_parent_dict[clone1[1]]:
                            output.write(f"\t{idx_dict[clone2]} -> {idx_dict[clone1]} [style=\"bold\"];\n")

                if clone1[1] == clone2[1]:
                    if clone1[0] in self.snv_parent_dict.keys():
                        if clone2[0] in self.snv_parent_dict[clone1[0]]:
                            output.write(f"\t{idx_dict[clone2]} -> {idx_dict[clone1]} [style=\"bold\"];\n")

            output.write(f'}}')
