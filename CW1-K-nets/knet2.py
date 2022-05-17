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
from sklearn.metrics.pairwise import euclidean_distances

class KNet():
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
    
    def fit(self,data):
        logging.info('fit')
        # single layer        
        if len(self.k) ==1:
            self.singleLayer(range(len(data)),distance_matrix(data,data))
            self.cluster_centers_ = np.array([data[i] for i in self.modes])
        # Multi layers serial structer
        elif self.multilayer=='serial':
            if not self.exact:
                self.multilayerSerialNOM(range(len(data)),distance_matrix(data,data))
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
            else:
                self.multilayerSerialEOM(range(len(data)),distance_matrix(data,data))
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
        # Multi layer parallel structer
        elif self.multilayer=='parallel':
            if not self.exact:
                self.multilayerParallelNOM(data)
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
            else: # exact mode
                self.multilayerParallelEOM(data)
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
        return self

    def predict(self,data):
        logging.info('predict')
        DistMat = distance_matrix(self.cluster_centers_,data)
        self.labels_ = np.zeros(len(data))
        for i in range(len(data)):
            self.labels_[i] = np.argsort(DistMat[i])
        return self.labels_

    def fit_predict(self,data):
        logging.info('fit_predict')
        # single layer        
        if len(self.k) ==1:
            self.singleLayer(range(len(data)),distance_matrix(data,data))
            self.cluster_centers_ = np.array([data[i] for i in self.modes])
        # Multi layers serial structer
        elif self.multilayer=='serial':
            if not self.exact:
                self.multilayerSerialNOM(range(len(data)),distance_matrix(data,data))
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
            else:
                self.multilayerSerialEOM(range(len(data)),distance_matrix(data,data))
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
        # Multi layer parallel structer
        elif self.multilayer=='parallel':
            if not self.exact:
                self.multilayerParallelNOM(data)
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
            else: # exact mode
                self.multilayerParallelEOM(data)
                self.cluster_centers_ = np.array([data[i] for i in self.modes])
        return self.labels_

    def singleLayer(self,indexs,DistMat):
        curK=self.k[0]
        modes = []
        points =[]
        remainP = indexs
        exact = self.exact if self.exact else 1
        while len(modes) <exact and curK>1:
            logging.info(f'K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')
            # Get the distance matrix
            Dist = np.zeros((len(remainP),len(remainP)))
            for i,c in enumerate(remainP):
                Dist[i] = [DistMat[c,j] for j in remainP]

            # 1.  Construction phase
            pc_ind,pc_sco = self.Construction(remainP,curK,Dist)
            # 2.  Selection phase
            m,p = self.Selection(remainP,pc_ind,pc_sco)
            remainP = remainP[remainP!=p]
            modes = np.concatenate((modes,m)).astype(int)
            curK-=1
            logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')

        # 3.  Assignment phase
        self.modes, points= self.Assignment(modes,DistMat)
        
        self.labels_ = [points[i] for i in range(len(points))]
        self.inertia_ = np.sum([DistMat[i,self.labels_[i]] for i in range(len(self.labels_))])
        return self.labels_, self.modes

    def multilayerSerialNOM(self,indexs,DistMat):
        logging.info(f'multilayer Serial NOM K={self.k}')
        logging.info(f'\n{"-"*35}1st  Layer{"-"*35}')
        pc_ind_l1,pc_sco_l1 = self.Construction(indexs,self.k[0],DistMat)
        modes_l1,p_l1 = self.Selection(indexs,pc_ind_l1,pc_sco_l1)
        centers_l1, points_l1 = self.Assignment(modes_l1,DistMat)
        logging.info(f'Assignment points {len(points_l1)}')
        logging.info(f'{"-"*33}End 1st  Layer{"-"*33}')

        # Second Layer:
        logging.info(f'\n{"-"*35}2nd  Layer{"-"*35}')
        # 2.1  Construction phase
        Dist = np.zeros((len(centers_l1),len(centers_l1)))
        for i,c in enumerate(centers_l1):
            Dist[i] = [DistMat[c,j] for j in centers_l1]
        # print(Dist)
        if self.geo>0:
            Dist =self.nlinmap(Dist,self.geo)
            logging.info('Geo Done!')

        pc_ind_l2,pc_sco_l2 = self.Construction(centers_l1,self.k[1],Dist)
        # 2.2  Selection phase
        modes_l2,p_l2 = self.Selection(p_l1,pc_ind_l2,pc_sco_l2)
        logging.info(f'Num of Modes: {len(modes_l2)}\n{modes_l2}')
        # 2.3  Assignment phase
        # index in centers_l1
        modes = [np.where(centers_l1==i)[0][0] for i in modes_l2]
        logging.info(f'Before Assignment modes in centers: {modes}')
        centers_l2, points_l2 = self.Assignment(modes,Dist)
        logging.info(f'After Assignment modes {centers_l2}')
        centers_l2 = [centers_l1[i] for i in centers_l2]

        logging.info(f'\n{"-"*33}End 2nd  Layer{"-"*33}')
        if self.geo>0:
            points_final = {}
            for k in points_l2.keys():
                points_final[centers_l1[k]] = centers_l1[points_l2[k]]
            self.modes = centers_l2
        else:
            self.modes, points_final = self.Assignment(centers_l2,DistMat)
        logging.info(f'Assignment modes {centers_l2}')
        logging.info(f'Assignment points {len(points_final)}')
        self.labels_ = [points_final[points_l1[i]] for i in indexs]
        self.inertia_ = np.sum([DistMat[i,self.labels_[i]] for i in range(len(self.labels_))])
        return self.labels_

    def multilayerSerialEOM(self,indexs,DistMat):
        logging.info(f'multilayer Serial EOM K={self.k}')
        logging.info(f'\n{"-"*35}1st  Layer{"-"*35}')
        pc_ind_l1,pc_sco_l1 = self.Construction(indexs,self.k[0],DistMat)
        modes_l1,p_l1 = self.Selection(indexs,pc_ind_l1,pc_sco_l1)
        centers_l1, points_l1 = self.Assignment(modes_l1,DistMat)
        logging.info(f'Assignment points {len(points_l1)}')
        logging.info(f'{"-"*33}End 1st  Layer{"-"*33}')

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
                Dist[i] = [DistMat[c,j] for j in remainP]
            # print(Dist)
            if self.geo>0:
                Dist =self.nlinmap(Dist,self.geo)
                logging.info('Geo Done!')

            pc_ind_l2,pc_sco_l2 = self.Construction(remainP,curK,Dist)
            
            # 2.2  Selection phase
            modes_l2,p_l2 = self.Selection(remainP,pc_ind_l2,pc_sco_l2)
            logging.info(f'Num of Modes: {len(modes_l2)}')
            logging.info(f'Selection modes {modes_l2}')
            remainP = remainP[remainP!=p_l2]
            modes = np.concatenate((modes,modes_l2)).astype(int)
            curK-=1
            logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')

        logging.info(f'after Assignment modes: {modes}') # 4
        # 2.3  Assignment phase
        # index in centers_l1
        modes = [np.where(centers_l1==i)[0][0] for i in modes]
        logging.info(f'Before Assignment modes in centers: {modes}') # 4
        centers_l2, points_l2 = self.Assignment(modes,Dist)
        logging.info(f'After Assignment modes {centers_l2}')  # 4
        centers_l2 = [centers_l1[i] for i in centers_l2]
        logging.info(f'relabel Assignment modes {centers_l2}') #297
        # points_final = {}
        # for k in points_l2.keys():
        #     points_final[centers_l1[k]] = centers_l1[points_l2[k]]
        # logging.info(f'Second Layer: Assignment modes {centers_l2}')
        # logging.info(f'Second Layer: Assignment points {len(points_final)}')

        logging.info(f'\n{"-"*35}2nd  Layer{"-"*35}')

        # self.labels_ = [points_final[points_l1[i]] for i in range(self.length)]
        # 
        # 778 1203 1795 2434 2905

        if self.geo>0:
            points_final = {}
            for k in points_l2.keys():
                points_final[centers_l1[k]] = centers_l1[points_l2[k]]
            self.modes = centers_l2
        else:
            self.modes, points_final = self.Assignment(centers_l2,DistMat)
        logging.info(f'Assignment modes {centers_l2}')
        logging.info(f'Assignment points {len(points_final)}')
        self.labels_ = [points_final[points_l1[i]] for i in indexs]
        self.inertia_ = np.sum([DistMat[i,self.labels_[i]] for i in range(len(self.labels_))])
        return self.labels_
    
    def multilayerParallelNOM(self,originData):
        logging.info(f'multilayer Parallel NOM K={self.k}')
        # divided into multiple parallel knets, default 300 datas per knets
        fromIndex= 0
        parallel_id = 0
        centers_Layer1 = np.array([],dtype=int)
        points_Layer1={}
        logging.info(f'\n{"="*35}1st  Layer{"="*35}')
        indexs = range(len(originData))
        dataSize = len(originData)-1
        while fromIndex<dataSize:
            parallel_id+=1
            toIndex = fromIndex+self.dstep if fromIndex+self.dstep<dataSize else dataSize
            if dataSize - toIndex <self.dstep:
                toIndex=dataSize
            idx = range(fromIndex,toIndex)
            
            # First Layer:
            logging.info(f'\n{"-"*35}[{parallel_id}] {fromIndex}-{toIndex}{"-"*35}')
            # 1.1  Construction phase
            data = originData[idx]
            Dist = distance_matrix(data,data)

            pc_ind_l1,pc_sco_l1 = self.Construction(idx,self.k[0],Dist)

            # 1.2  Selection phase
            modes_l1,p_l1 = self.Selection(idx,pc_ind_l1,pc_sco_l1)
            logging.info(f'Num of Modes: {len(modes_l1)}')

            # 3.  Assignment phase
            modes_in_indexs = modes_l1 - fromIndex
            centers_l1, points_l1 = self.Assignment(modes_in_indexs,Dist)
            # centers_l1, points_l1,clusters_l1 = self.Assignment(indexs, modes_l1,Dist)
            centers_l1 = centers_l1 + fromIndex
            ps = {}
            for p in points_l1.keys():
                ps[p+fromIndex] = points_l1[p] +fromIndex
            centers_Layer1 = np.concatenate((centers_Layer1,centers_l1))
            logging.info(f'Num of centers: {len(centers_l1)}\t total {len(centers_Layer1)}')
            points_Layer1.update(ps)
            logging.info(f'Assignment points {list(ps.values())}')
            
            fromIndex = toIndex
        logging.info(f'\n{"="*32}End 1st  Layer{"="*32}')

        logging.info(f'\n{"="*34}2nd  Layer{"="*34}')
        logging.info(f'Layer 2 K={self.k[1]} Centers_Layer1 = {centers_Layer1.shape} Points = {len(points_Layer1)}')
        
        # Second Layer:
        # 2.1  Construction phase
        data = originData[centers_Layer1]
        Dist = distance_matrix(data,data)
        if self.geo>0:
            Dist =self.nlinmap(Dist,self.geo)
            logging.info('Geo Done!')

        pc_ind_l2,pc_sco_l2 = self.Construction(range(len(centers_Layer1)),self.k[1],Dist)
        
        # 2.2  Selection phase
        modes_l2,p_l2 = self.Selection(range(len(centers_Layer1)),pc_ind_l2,pc_sco_l2)
        logging.info(f'Second Layer: Num of Modes: {len(modes_l2)}')
        logging.info(f'Second Layer: Selection modes {modes_l2}')
        # 2.3  Assignment phase
        modes_l2, _ = self.Assignment(modes_l2,Dist)
        logging.info(f'Second Layer: Assignment modes {modes_l2}')

        logging.info(f'\n{"="*34}End 2nd Layer{"="*34}')
        # final assignment phase
        # index in centers_l1
        self.modes = [centers_Layer1[i] for i in modes_l2]
        logging.info(f'Final: Before Assignment modes in centers: {self.modes}')
        data = originData[self.modes]
        Dist = distance_matrix(originData,data)
        logging.info(f'Final: Before assignment Dist Shape {Dist.shape}')
        self.labels_ = self.ParallelAssignment(self.modes,Dist)

        logging.info(f'Second Layer: Assignment labels {np.unique(self.labels_)}')


        # self.labels_ = [points_final[points_Layer1[i]] for i in indexs]
        # self.inertia_ = np.sum([distance_matrix([originData[i]],[originData[self.labels_[i]]])[0,0] for i in range(len(self.labels_))])
        return self.labels_

    def multilayerParallelEOM(self,originData):
        logging.info(f'Multi layer parallel EOM K={self.k}')
        # divided into multiple parallel knets, default 300 datas per knets
        fromIndex= 0
        parallel_id = 0
        centers_Layer1 = np.array([],dtype=int)
        points_Layer1={}
        logging.info(f'\n{"="*35}1st  Layer{"="*35}')
        indexs = range(len(originData))
        dataSize = len(originData)-1
        while fromIndex<dataSize:
            parallel_id+=1
            toIndex = fromIndex+self.dstep if fromIndex+self.dstep<dataSize else dataSize
            if dataSize - toIndex <self.dstep:
                toIndex=dataSize
            idx = range(fromIndex,toIndex)
            
            # First Layer:
            logging.info(f'\n{"-"*35}[{parallel_id}] {fromIndex}-{toIndex}{"-"*35}')
            # 1.1  Construction phase
            data = originData[idx]
            Dist = distance_matrix(data,data)

            pc_ind_l1,pc_sco_l1 = self.Construction(idx,self.k[0],Dist)

            # 1.2  Selection phase
            modes_l1,p_l1 = self.Selection(idx,pc_ind_l1,pc_sco_l1)
            logging.info(f'Num of Modes: {len(modes_l1)}')

            # 3.  Assignment phase
            modes_in_indexs = modes_l1 - fromIndex
            centers_l1, points_l1 = self.Assignment(modes_in_indexs,Dist)
            # centers_l1, points_l1,clusters_l1 = self.Assignment(indexs, modes_l1,Dist)
            centers_l1 = centers_l1 + fromIndex
            ps = {}
            for p in points_l1.keys():
                ps[p+fromIndex] = points_l1[p] +fromIndex
            centers_Layer1 = np.concatenate((centers_Layer1,centers_l1))
            logging.info(f'Num of centers: {len(centers_l1)}\t total {len(centers_Layer1)}')
            points_Layer1.update(ps)
            # logging.info(f'Assignment points {list(ps.values())}')
            
            fromIndex = toIndex
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
            logging.info(remainP)
            data = originData[remainP]
            Dist = distance_matrix(data,data)

            if self.geo>0:
                Dist =self.nlinmap(Dist,self.geo)
                logging.info('Geo Done!')

            pc_ind_l2,pc_sco_l2 = self.Construction(remainP,curK,Dist)
            
            # 2.2  Selection phase
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
        data = originData[centers_Layer1]
        Dist = distance_matrix(data,data)
        if self.geo>0:
            Dist =self.nlinmap(Dist,self.geo)
            logging.info('Geo Done!')
        # index in centers_l1
        modes = [np.where(centers_Layer1==i)[0][0] for i in modes]
        logging.info(f'Second Layer: Before Assignment modes in centers: {modes}')
        logging.info(f'len centers_Layer1 = {len(centers_Layer1)}')
        centers_l2, points_l2 = self.Assignment( modes,Dist)
        logging.info(f'Second Layer: Assignment modes {centers_l2}')
        
        self.modes = [centers_Layer1[i] for i in centers_l2]

        logging.info(f'Final: Before Assignment modes in centers: {self.modes}')
        data = originData[self.modes]
        Dist = distance_matrix(originData,data)
        logging.info(f'Final: Before assignment Dist Shape {Dist.shape}')
        self.labels_ = self.ParallelAssignment(self.modes,Dist)

        logging.info(f'\n{"-"*80}')
        logging.info(f'Second Layer: Assignment labels {np.unique(self.labels_)}')

    def ParallelAssignment(self,modes, dist):
        points = np.zeros(dist.shape[0],dtype=int)
        self.inertia_ = 0
        for i,d in enumerate(dist):
            nearestDist = np.sort(d)[0]
            self.inertia_ += nearestDist
            nearest=np.argsort(d)[0]
            points[i] = modes[nearest]
        return points

    def Construction(self,indexs,k,dist,geo=0):
        tic = time.time()
        logging.info(f'Construction K={k} Geo={geo} #Data: {len(indexs)}')
        pc_val = {}
        pc_ind = {}
        pc_sco = {}
        pc_sco_v = []
        pc_sco_t = {}
        for i in range(len(indexs)):
            sv = np.sort(dist[i])
            si = np.argsort(dist[i])
            pc_val[indexs[i]] = sv[:k]
            pc_ind[indexs[i]] = [indexs[j] for j in si[:k]]
            pc_sco[i] = np.mean(pc_val[indexs[i]])
            pc_sco_t[i] = np.mean(dist[i])
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
        for i in pc_sco:
            isin = np.isin(pc_ind[i],points)
            if not np.any(isin):
                points = np.concatenate((points,pc_ind[i])).astype(int)
                modes = np.concatenate((modes,[i])).astype(int)
        logging.info(f'Modes\t{len(modes)}\tPoints\t{len(points)}')
        logging.info(f'\tSelection Time: {time.time()-tic:.2f}')
        return modes,points

    def Assignment(self,modes,dist,maxIteration=100):
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
            for i in range(dist.shape[0]):
                ls_d=[dist[i,j] for j in centers]
                nearest=np.argsort(ls_d)[0]
                points[i] = centers[nearest]

                clusters[points[i]] = np.concatenate((clusters[points[i]],[i])).astype(int)
            c = np.zeros(len(centers),dtype=int)

            for i,k in enumerate(clusters.keys()):
                sumDist = np.zeros(len(clusters[k]))
                ps = clusters[k].astype(int)
                for j in range(len(ps)):
                    sumDist[j] = np.sum([dist[ps[j],q] for q in ps])

                c[i] = ps[np.argsort(sumDist)[0]]
            if np.all(np.isin(c,centers)):
                logging.info(f'No Changes! Convergence')
                convergence = True
            else:
                logging.info('Changes!')
                centers= c
                iteration+=1
        logging.debug(f'Modes:\n{centers}\nPoints:\n{points}\nClusters:\n{clusters}')
        logging.info(f'\tAssignment Time: {time.time()-tic:.2f}')
        return centers, points
   
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