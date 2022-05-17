import logging
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.spatial import distance_matrix
import configparser
import time
from tqdm import tqdm
# import os
from os import path,mkdir

# Todo:
#       1.    Implement KNet
#       1.1   Single Layer
# ✅    1.1.1 NOM
# ✅    1.1.2 EOM
#       1.2   Multi-Layer
#       1.2.1 Serial connections
#       1.2.2 Parallel connections
#       2.    Correctness
#       3.    Debug,Logging
#       4.    Show graphs
#       5.    Other Algorithms, KMeans, DBScan, AP, MeanShift
#       6.    Metrics

# single layer
# Normal Operational Mode (NOM)
# 1.  Construction phase
# 1.1 Every datapoint as pre-Examplar (PE)
# 1.2 Build pre-cluster(PC) composed of its K nearest neighbors
# 1.3 Assign score 
#
# 2.  Selection phase
# 2.1 Sort the score min to max
# 2.2 Insert current PE in M(ode) if None member in L
# 2.3 Insert all member of PE in L
#
# 3.  Assignment phase
# 3.1 Consider |M| PEs as examplars
# 3.2 Assign every point to its nearest examplar
# 3.3 Set the most center point as examplar until convergence
#
# Exact Operational Mode (EOM)
# 
# multi layers
# Serial and parallel connections
# function [idx, vals, Pnts_Scores] = knet(DN, k, varargin)
class KNet():                                           #'parallel'
    def __init__(self,k,exact=False,geo=0,multilayer='serial'):
        self.k=[k] if type(k)==int else k
        self.lable_ = []
        self.exact = exact
        self.geo=geo
        self.multilayer=multilayer
        self.dstep=300
        if exact:
            logging.info(f'Exact mode with exact {exact} clusters and Initial K = {k}')
        else:
            logging.info(f'Normal mode with Initial K = {k}')
        if len(k)>1:
            logging.info(f'Multi layers {k}')
        else:
            logging.info(f'Single layer {k[0]}')

    def computeDistanceMatrix(self):
        logging.info('Compute Dist matrix')
        self.distMat = distance_matrix(self.data,self.data)
        logging.debug(f'DistMat: \n{self.distMat}')
        
    def KNN(self,indexs,k,Dist,geo=0):
        tic = time.time()
        logging.info(f'KNN K={k} Geo={geo} Index: {len(indexs)}')
        pc_val = {}
        pc_ind = {}
        pc_sco = {}
        pc_sco_v = []
        pc_sco_t = {}
        for i in tqdm(range(len(indexs)),desc='Construction',unit='d'):
            sv = np.sort(Dist[i])
            si = np.argsort(Dist[i])
            pc_val[indexs[i]] = sv[:k]
            pc_ind[indexs[i]] = [indexs[j] for j in si[:k]]
            pc_sco[i] = np.mean(pc_val[indexs[i]])
            pc_sco_t[i] = np.mean(Dist[i])
            pc_sco_v.append(pc_sco[i])

        uniq_sco = np.unique(pc_sco_v)
        for s in uniq_sco:
            ind,=np.where(pc_sco_v==s)
            if len(ind)>1:
                logging.info(f'insteady @{ind} Num={len(ind)}')
                for i in ind:
                    pc_sco_v[i] = pc_sco_t[i]

        pc_sco_i_sort = np.argsort(pc_sco_v)
        pc_sco_i_sort = [indexs[j] for j in pc_sco_i_sort]
        logging.info(f'\tConstruction Time: {time.time()-tic:.2f}')
        return pc_ind,pc_sco_i_sort

    def Selection(self,indexs,pc_ind,pc_sco):
        tic = time.time()
        logging.info(f'Selection:\t{indexs}\tInd:\t{len(pc_ind)}\tPC Score:\t{len(pc_sco)}')
        modes = []
        points= []
        for i in tqdm(pc_sco,desc='Selection',unit='d'):
            isin = np.isin(pc_ind[i],points)
            if not np.any(isin):
                points = np.concatenate((points,pc_ind[i])).astype(int)
                modes = np.concatenate((modes,[i])).astype(int)
        logging.info(f'Modes\t{len(modes)}\tPoints\t{len(points)}')
        logging.info(f'\tSelection Time: {time.time()-tic:.2f}')

        return modes,points

    def Assignment(self,indexs,modes,distMat,maxIteration=100):
        tic = time.time()
        centers = modes
        if self.exact:
            centers = centers[:self.exact]
        convergence = False
        iteration = 1
        while not convergence and iteration<=maxIteration:
            clusters = {}
            points = {}
            for m in centers:
                clusters[m] = []
            for i in tqdm(indexs,desc=f'Assign {iteration}',unit='d'):
                ls_d=[distMat[i,j] for j in centers]
                nearest=np.argsort(ls_d)[0]
                points[i] = centers[nearest]

                clusters[points[i]] = np.concatenate((clusters[points[i]],[i])).astype(int)
            c = np.zeros(len(centers),dtype=int)

            for i,k in enumerate(clusters.keys()):
                sumDist = np.zeros(len(clusters[k]))
                ps = clusters[k].astype(int)
                for j in range(len(ps)):
                    # sumDist[j] = np.sum([self.distMat[ps[j],q] for q in ps])
                    sumDist[j] = np.sum([distMat[ps[j],q] for q in ps])

                c[i] = ps[np.argsort(sumDist)[0]]
            if np.all(np.isin(c,centers)):
                logging.info(f'No Changes! Convergence')
                convergence = True
            else:
                logging.info('Changes!')
                centers= c
                iteration+=1
        logging.debug(f'Modes:\n{centers}\nPoints:\n{points}\nClusters:\n{clusters}')
        logging.info(f'Modes: {centers}')
        logging.info(f'\tAssignment Time: {time.time()-tic:.2f}')
        return centers, points

    def singleLayerNOM(self):
        pc_ind,pc_sco = self.KNN(range(self.length),self.k[0],self.distMat)
        modes,points = self.Selection(range(self.length),pc_ind,pc_sco)
        _, points= self.Assignment(range(self.length), modes,self.distMat)
        self.labels_ = [points[i] for i in range(len(points))]

    def singleLayerEOM(self):
        curK=self.k[0]
        modes = []
        points =[]
        remainP = range(self.length)
        while len(modes) <self.exact and curK>1:
            logging.info(f'K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')
            # 1.  Construction phase
            Dist = np.zeros((len(remainP),len(remainP)))
            for i,c in enumerate(remainP):
                Dist[i] = [self.distMat[c,j] for j in remainP]

            pc_ind,pc_sco = self.KNN(remainP,curK,Dist)
            # 2.  Selection phase
            m,p = self.Selection(remainP,pc_ind,pc_sco)
            remainP = remainP[remainP!=p]
            modes = np.concatenate((modes,m)).astype(int)
            curK-=1
            logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')

        # 3.  Assignment phase
        _, points= self.Assignment(range(self.length), modes,self.distMat)

        self.labels_ = [points[i] for i in range(len(points))]

    def multilayerSerialNOM(self):
        logging.info(f'multilayer Serial NOM K={self.k}')
        # NOM
        # First Layer:
        # 1.1  Construction phase
        logging.info(f'\n{"-"*35}1st  Layer{"-"*35}')
        pc_ind_l1,pc_sco_l1 = self.KNN(range(self.length),self.k[0],self.distMat)

        # 1.2  Selection phase
        modes_l1,p_l1 = self.Selection(range(self.length),pc_ind_l1,pc_sco_l1)
        logging.info(f'Num of Modes: {len(modes_l1)}')

        # 1.3  Assignment phase
        centers_l1, points_l1 = self.Assignment(range(self.length), modes_l1,self.distMat)
        logging.info(f'Assignment points {len(points_l1)}')
        logging.info(f'{"-"*33}End 1st  Layer{"-"*33}')
        # logging.info(f'centers: {centers}')

        # Second Layer:
        logging.info(f'\n{"-"*35}2nd  Layer{"-"*35}')
        # 2.1  Construction phase
        Dist = np.zeros((len(centers_l1),len(centers_l1)))
        for i,c in enumerate(centers_l1):
            Dist[i] = [self.distMat[c,j] for j in centers_l1]
        # print(Dist)
        if self.geo>0:
            Dist =self.nlinmap(Dist,self.geo)
            logging.info('Geo Done!')

        pc_ind_l2,pc_sco_l2 = self.KNN(centers_l1,self.k[1],Dist)
        
        # 2.2  Selection phase
        modes_l2,p_l2 = self.Selection(p_l1,pc_ind_l2,pc_sco_l2)
        logging.info(f'Num of Modes: {len(modes_l2)}\n{modes_l2}')
        # 2.3  Assignment phase
        # index in centers_l1
        modes = [np.where(centers_l1==i)[0][0] for i in modes_l2]
        logging.info(f'Before Assignment modes in centers: {modes}')
        centers_l2, points_l2 = self.Assignment(range(len(centers_l1)), modes,Dist)
        logging.info(f'After Assignment modes {centers_l2}')
        centers_l2 = [centers_l1[i] for i in centers_l2]

        if self.geo>0:
            points_final = {}
            for k in points_l2.keys():
                points_final[centers_l1[k]] = centers_l1[points_l2[k]]    
        else:
            centers_l2, points_final = self.Assignment(range(self.length), centers_l2,self.distMat)
        logging.info(f'Assignment modes {centers_l2}')
        logging.info(f'Assignment points {len(points_final)}')
        logging.info(f'\n{"-"*33}End 2nd  Layer{"-"*33}')
        self.labels_ = [points_final[points_l1[i]] for i in range(self.length)]
           
    def multilayerSerialEOM(self):
        logging.info(f'multilayer Serial EOM K={self.k} Exact={self.exact}')
        # EOM
        # First Layer:
        # 1.1  Construction phase
        logging.info(f'\n{"-"*35}1st  Layer{"-"*35}')
        pc_ind_l1,pc_sco_l1 = self.KNN(range(self.length),self.k[0],self.distMat)

        # 1.2  Selection phase
        modes_l1,p_l1 = self.Selection(range(self.length),pc_ind_l1,pc_sco_l1)
        logging.info(f'First Layer: Num of Modes: {len(modes_l1)}')

        # 1.3  Assignment phase
        centers_l1, points_l1 = self.Assignment(range(self.length), modes_l1,self.distMat)
        logging.info(f'First Layer: Assignment points {len(points_l1)}')
        logging.info(f'\n{"-"*35}1st  Layer{"-"*35}')
        # logging.info(f'centers: {centers}')
        
        curK=self.k[1]
        modes = []
        points =[]
        remainP = centers_l1
        logging.info(f'\n{"-"*35}2nd  Layer{"-"*35}')
        # Second Layer:
        while len(modes)<self.exact and curK>1:
            # 2.1  Construction phase
            Dist = np.zeros((len(remainP),len(remainP)))
            for i,c in enumerate(remainP):
                Dist[i] = [self.distMat[c,j] for j in remainP]
            # print(Dist)
            if self.geo>0:
                Dist =self.nlinmap(Dist,self.geo)
                logging.info('Geo Done!')

            pc_ind_l2,pc_sco_l2 = self.KNN(remainP,curK,Dist)
            
            # 2.2  Selection phase
            modes_l2,p_l2 = self.Selection(remainP,pc_ind_l2,pc_sco_l2)
            logging.info(f'Num of Modes: {len(modes_l2)}')
            logging.info(f'Selection modes {modes_l2}')
            remainP = remainP[remainP!=p_l2]
            modes = np.concatenate((modes,modes_l2)).astype(int)
            curK-=1
            logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')

        # 407 870

        # 2.3  Assignment phase
        # index in centers_l1
        modes = [np.where(centers_l1==i)[0][0] for i in modes_l2]
        logging.info(f'Before Assignment modes in centers: {modes}')
        centers_l2, points_l2 = self.Assignment(range(len(centers_l1)), modes,Dist)
        logging.info(f'After Assignment modes {centers_l2}')
        centers_l2 = [centers_l1[i] for i in centers_l2]
        points_final = {}
        for k in points_l2.keys():
            points_final[centers_l1[k]] = centers_l1[points_l2[k]]
        logging.info(f'Second Layer: Assignment modes {centers_l2}')
        logging.info(f'Second Layer: Assignment points {len(points_final)}')

        logging.info(f'\n{"-"*35}2nd  Layer{"-"*35}')

        self.labels_ = [points_final[points_l1[i]] for i in range(self.length)]

    def multilayerParallelNOM(self):
        logging.info(f'multilayer Parallel NOM K={self.k}')
        # divided into multiple parallel knets, default 300 datas per knets
        fromIndex= 0
        parallel_id = 0
        centers_Layer1 = np.array([],dtype=int)
        points_Layer1={}
        logging.info(f'\n{"="*35}1st  Layer{"="*35}')
        while fromIndex<self.length:
            parallel_id+=1
            toIndex = fromIndex+self.dstep if fromIndex+self.dstep<self.length else self.length
            if self.length - toIndex <self.dstep:
                toIndex=self.length
            indexs = range(fromIndex,toIndex)
            
            # First Layer:
            logging.info(f'\n{"-"*35}[{parallel_id}] {fromIndex}-{toIndex}{"-"*35}')
            # 1.1  Construction phase
            data = self.data[indexs]
            Dist = distance_matrix(data,data)

            pc_ind_l1,pc_sco_l1 = self.KNN(indexs,self.k[0],Dist)

            # 1.2  Selection phase
            modes_l1,p_l1 = self.Selection(indexs,pc_ind_l1,pc_sco_l1)
            logging.info(f'Num of Modes: {len(modes_l1)}')

            # 3.  Assignment phase
            tic = time.time()
            modes_in_indexs = modes_l1 - fromIndex
            centers_l1, points_l1 = self.Assignment(range(len(indexs)), modes_in_indexs,Dist)
            # centers_l1, points_l1,clusters_l1 = self.Assignment(indexs, modes_l1,Dist)
            centers_l1 = centers_l1 + fromIndex
            ps = {}
            for p in points_l1.keys():
                ps[p+fromIndex] = points_l1[p] +fromIndex
            centers_Layer1 = np.concatenate((centers_Layer1,centers_l1))
            logging.info(f'Num of centers: {len(centers_l1)}\t total {len(centers_Layer1)}')
            points_Layer1.update(ps)
            logging.info(f'Assignment points {len(ps)}')
            
            fromIndex = toIndex
        logging.info(f'\n{"="*32}End 1st  Layer{"="*32}')

        logging.info(f'\n{"="*35}2nd  Layer{"="*35}')
        logging.info(f'Layer 2 K={self.k[1]} Centers_Layer1 = {centers_Layer1.shape} Points = {len(points_Layer1)}')
        
        
        
        # Second Layer:
        # 2.1  Construction phase
        # logging.info(centers_Layer1)
        data = self.data[centers_Layer1]
        Dist = distance_matrix(data,data)
        # Dist = np.zeros((len(centers_Layer1),len(centers_Layer1)))
        # for i,c in enumerate(centers_Layer1):
        #     Dist[i] = [self.distMat[c,j] for j in centers_Layer1]
        # print(Dist)
        if self.geo>0:
            Dist =self.nlinmap(Dist,self.geo)
            logging.info('Geo Done!')

        pc_ind_l2,pc_sco_l2 = self.KNN(centers_Layer1,self.k[1],Dist)
        
        # 2.2  Selection phase
        modes_l2,p_l2 = self.Selection(p_l1,pc_ind_l2,pc_sco_l2)
        logging.info(f'Second Layer: Num of Modes: {len(modes_l2)}')
        logging.info(f'Second Layer: Selection modes {modes_l2}')
        # 2.3  Assignment phase
        # index in centers_l1
        modes = [np.where(centers_Layer1==i)[0][0] for i in modes_l2]
        logging.info(f'Second Layer: Before Assignment modes in centers: {modes}')
        centers_l2, points_l2 = self.Assignment(range(len(centers_Layer1)), modes,Dist,maxIteration=1)
        logging.info(f'Second Layer: Assignment modes {centers_l2}')
        centers_l2 = [centers_Layer1[i] for i in centers_l2]

        points_final = {}
        for k in points_l2.keys():
            points_final[centers_Layer1[k]] = centers_Layer1[points_l2[k]]
        logging.info(f'Second Layer: Assignment modes {centers_l2}')
        logging.info(f'Second Layer: Assignment points {len(points_final)}')

        logging.info(f'\n{"="*35}2nd  Layer{"="*35}')
        self.labels_ = [points_final[points_Layer1[i]] for i in range(self.length)]

    def multilayerParallelEOM(self):
        logging.info(f'Multi layer parallel structer K={self.k}')
        # divided into multiple parallel knets, default 300 datas per knets
        fromIndex= 0
        parallel_id = 0
        centers_Layer1 = np.array([],dtype=int)
        points_Layer1={}
        while fromIndex<self.length:
            parallel_id+=1
            toIndex = fromIndex+self.dstep if fromIndex+self.dstep<self.length else self.length
            indexs = range(fromIndex,toIndex)
            fromIndex = toIndex
            # First Layer:
            # 1.1  Construction phase
            logging.info(f'\n{"-"*80}')
            tic = time.time()
            data = self.data[indexs]
            Dist = distance_matrix(data,data)
            # Dist = np.zeros((len(indexs),len(indexs)))
            # for i,c in enumerate(indexs):
            #     Dist[i] = [self.distMat[c,j] for j in indexs]
            logging.info(f'Len Dist {Dist.shape} Len Index {len(indexs)} Len Data {len(data)}')
            pc_ind_l1,pc_sco_l1 = self.KNN(indexs,self.k[0],Dist)

            # 1.2  Selection phase
            tic = time.time()
            modes_l1,p_l1 = self.Selection(indexs,pc_ind_l1,pc_sco_l1)
            
            logging.info(f'First Layer [{parallel_id}]: Num of Modes: {len(modes_l1)}')

            # 1.3  Assignment phase
            tic = time.time()
            modes_in_indexs = modes_l1 - fromIndex
            centers_l1, points_l1 = self.Assignment(range(len(indexs)), modes_in_indexs,Dist)
            centers_l1 = centers_l1 + fromIndex
            centers_Layer1 = np.concatenate((centers_Layer1,centers_l1))
            points_Layer1.update(points_l1)
            logging.info(f'First Layer [{parallel_id}]: Assignment points {len(points_l1)}')
            logging.info(f'First Layer [{parallel_id}]: Assignment modes {len(centers_l1)}')

            logging.info(f'\n{"-"*80}')

        logging.info(f'\n{"="*80}')
        logging.info(f'Layer 2 K={self.k[1]} Centers_Layer1 = {centers_Layer1.shape} Points = {len(points_Layer1)}')
        

        curK=self.k[1]
        modes = []
        points =[]
        remainP = centers_Layer1

        # Second Layer:
        while len(modes)<self.exact and curK>1:
            # 2.1  Construction phase
            tic = time.time()
            logging.info(remainP)
            data = self.data[remainP]
            Dist = distance_matrix(data,data)

            # Dist = np.zeros((len(remainP),len(remainP)))
            # for i,c in enumerate(remainP):
            #     Dist[i] = [self.distMat[c,j] for j in remainP]
            # print(Dist)
            if self.geo>0:
                Dist =self.nlinmap(Dist,self.geo)
                logging.info('Geo Done!')

            pc_ind_l2,pc_sco_l2 = self.KNN(remainP,curK,Dist)
            
            # 2.2  Selection phase
            tic = time.time()
            modes_l2,p_l2 = self.Selection(remainP,pc_ind_l2,pc_sco_l2)
            logging.info(f'Second Layer: Num of Modes: {len(modes_l2)}')
            logging.info(f'Second Layer: Selection modes {modes_l2}')
            logging.info(f'Second Layer: Num of Remain: {len(remainP)}')
            remainP = remainP[remainP!=modes_l2]
            modes = np.concatenate((modes,modes_l2)).astype(int)
            curK-=1
            logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')
            logging.info(f'modes= {modes}')

        # 2.3  Assignment phase
        tic = time.time()
        Dist = np.zeros((len(centers_Layer1),len(centers_Layer1)))
        for i,c in enumerate(centers_Layer1):
            Dist[i] = [self.distMat[c,j] for j in centers_Layer1]
        # print(Dist)
        if self.geo>0:
            Dist =self.nlinmap(Dist,self.geo)
            logging.info('Geo Done!')
        # index in centers_l1
        modes = [np.where(centers_Layer1==i)[0][0] for i in modes]
        logging.info(f'Second Layer: Before Assignment modes in centers: {modes}')
        logging.info(f'len centers_Layer1 = {len(centers_Layer1)}')
        centers_l2, points_l2 = self.Assignment(range(len(centers_Layer1)), modes,Dist,maxIteration=1)
        logging.info(f'Second Layer: Assignment modes {centers_l2}')
        centers_l2 = [centers_Layer1[i] for i in centers_l2]
        points_final = {}
        for k in points_l2.keys():
            points_final[centers_Layer1[k]] = centers_Layer1[points_l2[k]]
        logging.info(f'Second Layer: Assignment modes {centers_l2}')
        logging.info(f'Second Layer: Assignment points {len(points_final)}')

        logging.info(f'\n{"-"*80}')

        self.labels_ = [points_final[points_Layer1[i]] for i in range(self.length)]

    def fit_predict(self,data):
        logging.info('fit')
        self.data= data
        self.length=self.data.shape[0]
        # single layer        
        if len(self.k) ==1:
            self.computeDistanceMatrix()
            if not self.exact:
                self.singleLayerNOM()
            else:
                self.singleLayerEOM()
        # Multi layers serial structer
        elif self.multilayer=='serial':
            self.computeDistanceMatrix()
            if not self.exact:
                self.multilayerSerialNOM()
            else:
                self.multilayerSerialEOM()
        # Multi layer parallel structer
        elif self.multilayer=='parallel':
            if not self.exact:
                self.multilayerParallelNOM()
            else: # exact mode
                self.multilayerParallelEOM()
        return self.labels_


    def nlinmap(self,D,K):
        N = D.shape[0]
        INF = 1000*np.max(np.max(D))*N
        ind = np.argsort(D)
        for i in range(N):
            D[i,ind[i,1+K:]] = INF
        D = np.minimum(D,D.T)
        for i in tqdm(range(N),desc='nlinmap'):
            mat= np.matlib.repmat(D[:,i],N,1).T+np.matlib.repmat(D[i,:],N,1)
            D = np.minimum(D,mat)
        return D