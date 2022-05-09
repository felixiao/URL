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

def PlotData():
    fig,ax= plt.subplots(3,3)
    # load data31
    data = pd.read_csv('Artificial/art_data/data31.csv',header=None)
    ax[0,0].scatter(data[0],data[1],marker='.')
    ax[0,0].set_aspect(1)
    # load data7
    data = pd.read_csv('Artificial/art_data/data7.csv',header=None)
    ax[0,1].scatter(data[0],data[1],marker='.')
    ax[0,1].set_aspect(1)
    # load data4
    data = pd.read_csv('Artificial/art_data/data4.csv',header=None)
    ax[0,2].scatter(data[0],data[1],marker='.')
    ax[0,2].set_aspect(1)
    # load data3
    data = pd.read_csv('Artificial/art_data/data3.csv',header=None)
    ax[1,0].scatter(data[0],data[1],marker='.')
    ax[1,0].set_aspect(1)
    # load ncircles
    data = pd.read_csv('Artificial/art_data/ncircles.csv',header=None)
    ax[1,1].scatter(data[0],data[1],marker='.')
    ax[1,1].set_aspect(1)
    # load data50
    data = pd.read_csv('Artificial/art_data/data50.csv',header=None)
    ax[1,2].scatter(data[0],data[1],marker='.')
    ax[1,2].ticklabel_format(scilimits=[-5,4])
    ax[1,2].set_aspect(1)
    # load noisy_spiral
    data = pd.read_csv('Artificial/art_data/noisy_spiral.csv',header=None)
    ax[2,0].scatter(data[0],data[1],marker='.')
    ax[2,0].set_aspect(1)
    # load birch_sin
    data = pd.read_csv('Artificial/art_data/birch_sin.csv',header=None)
    ax[2,1].scatter(data[0],data[1],marker='.')
    ax[2,1].ticklabel_format(scilimits=[-5,4])
    ax[2,1].set_aspect(1)
    # load birch_grid
    data = pd.read_csv('Artificial/art_data/birch_grid.csv',header=None)
    ax[2,2].scatter(data[0],data[1],marker='.')
    ax[2,2].set_aspect(1)
    plt.tight_layout(pad=1)
    plt.show()
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
class KNet():
    def __init__(self,data,k,labelTrue = None,exact=False,geo=0,struct=False,metric='euclidean',resultPath='results/data/'):
        self.data=data
        self.length=self.data.shape[0]
        self.k=[k] if type(k)==int else k
        self.idx=None
        self.vals=None
        self.Pnts_Scores=None
        self.clusterIndex = np.zeros(self.length).astype(int)
        self.labelTrue = labelTrue
        self.exact = exact
        self.geo=geo
        self.modes = []
        self.list_points = []
        if not path.exists(resultPath):
            mkdir(resultPath)
        self.resultPath = resultPath
        logging.info(f'Load {self.resultPath.split("/")[-1]} Length= {self.length}')
        if exact:
            logging.info(f'Exact mode with exact {exact} clusters and Initial K = {k}')
        else:
            logging.info(f'Normal mode with Initial K = {k}')
        if len(k)>1:
            logging.info(f'Multi layers {k}')
        else:
            logging.info(f'Single layer {k[0]}')

    def computeDistanceMatrix(self):
        self.distMat = distance_matrix(self.data,self.data)
        logging.debug(f'DistMat: \n{self.distMat}')

    def KNN(self,indexs,k,Dist,geo=0):
        #distmat    [1,1]   [2,2]   [3,3]   [4,4]
        #[1,1]      0       √2      2       √6
        #[2,2]      √2      0       √2      2
        #[3,3]      2       √2      0       √2
        #[4,4]      √6      2       √2      0

        #KNN,K=2    [1,1]   [2,2]   [3,3]   [4,4]
        #0          1,√2    0,√2    1,√2    2,√2
        #1          3,2     2,√2    3,√2    1,2

        #Score      [1,1]   [2,2]   [3,3]   [4,4]
        #           1+√2/2  √2      √2      1+√2/2
        logging.info(f'KNN K={k} Geo={geo} Index: {len(indexs)}')
        pc_val = {}
        pc_ind = {}
        pc_sco = {}
        pc_sco_v = []
        pc_sco_t = {}
        # for i in range(self.length):
        for i in tqdm(range(len(indexs)),desc='Construction',unit='d'):
            # dist = [Dist[i,j] for j in indexs]
            # dist = self.distMat[i,:]
            # logging.info(f'[{i}] = index{indexs[i]} {len(Dist[i])}')
            sv = np.sort(Dist[i])
            si = np.argsort(Dist[i])
            pc_val[indexs[i]] = sv[:k]
            pc_ind[indexs[i]] = [indexs[j] for j in si[:k]]
            pc_sco[i] = np.mean(pc_val[indexs[i]])
            pc_sco_t[i] = np.mean(Dist[i])
            pc_sco_v.append(pc_sco[i])
        uniq_sco= np.unique(pc_sco_v)
        for s in uniq_sco:
            ind, = np.where(pc_sco_v==s)
            if len(ind)>1:
                for i in ind:
                    pc_sco_v[i] = pc_sco_t[i]
        pc_sco_i_sort = [indexs[j] for j in np.argsort(pc_sco_v)]
        # logging.info(f'KNN val:\n{pc_val}\nind:\n{pc_ind}\nsco:\n{pc_sco_i_sort}')
        return pc_val,pc_ind,pc_sco_i_sort

    def Selection(self,indexs,pc_ind,pc_sco):
        logging.debug(f'Selection:\n{indexs}\nInd:\n{pc_ind}\nPC Score:\n{pc_sco}')
        modes = []
        points= []
        for i in tqdm(pc_sco,desc='Selection',unit='d'):
            isin = np.isin(pc_ind[i],points)
            if not np.any(isin):
                points = np.concatenate((points,pc_ind[i])).astype(int)
                modes = np.concatenate((modes,[i])).astype(int)
        logging.debug(f'Modes\n{modes}\nPoints\n{points}')
        return modes,points

    def Assignment(self,indexs,modes,distMat):
        centers = modes
        if self.exact:
            centers = centers[:self.exact]
        convergence = False
        iteration = 1
        while not convergence:
            clusters = {}
            points = {}
            for m in centers:
                clusters[m] = []
            logging.debug(f'cluster:{clusters}')
            for i in tqdm(indexs,desc=f'Assign {iteration}',unit='d'):
                ls_d=[distMat[i,j] for j in centers]
                logging.debug(f'lsd: {ls_d}')
                nearest=np.argsort(ls_d)[0]
                logging.debug(f'near: {nearest}')
                points[i] = centers[nearest]
                logging.debug(f'Point[{i}] = {points[i]}')

                logging.debug(f'clusters[{points[i]}] = {clusters[points[i]]}')
                clusters[points[i]] = np.concatenate((clusters[points[i]],[i])).astype(int)
            c = np.zeros(len(centers),dtype=int)

            for i,k in enumerate(clusters.keys()):
                sumDist = np.zeros(len(clusters[k]))
                ps = clusters[k].astype(int)
                # logging.info(f'PS {ps}')
                for j in range(len(ps)):
                    sumDist[j] = np.sum([self.distMat[ps[j],q] for q in ps])

                c[i] = ps[np.argsort(sumDist)[0]]
                logging.debug(f'[{i}] center = {c[i]}')
            logging.debug(f'centers:\n{centers}\nc:\n{c}')
            if np.all(np.isin(c,centers)):
                logging.info(f'No Changes! Convergence, Iter = {iteration}')
                convergence = True
            else:
                logging.info('Changes!')
                centers= c
                iteration+=1
        logging.debug(f'Modes:\n{centers}\nPoints:\n{points}\nClusters:\n{clusters}')
        return centers, points,clusters

    def AssignmentGeo(self,indexs,modes,distMat,maxIteration=100):
        centers = modes
        if self.exact:
            centers = centers[:self.exact]
        convergence = False
        iteration = 1
        while not convergence and iteration < maxIteration:
            clusters = {}
            points = {}
            for m in centers:
                clusters[m] = []
            logging.debug(f'cluster:{clusters}')
            for i in tqdm(indexs,desc=f'Assign {iteration}',unit='d'):
                ls_d=[distMat[i,j] for j in centers]
                logging.debug(f'lsd: {ls_d}')
                nearest=np.argsort(ls_d)[0]
                logging.debug(f'near: {nearest}')
                points[i] = centers[nearest]
                logging.debug(f'Point[{i}] = {points[i]}')

                logging.debug(f'clusters[{points[i]}] = {clusters[points[i]]}')
                clusters[points[i]] = np.concatenate((clusters[points[i]],[i])).astype(int)
            c = np.zeros(len(centers),dtype=int)

            for i,k in enumerate(clusters.keys()):
                sumDist = np.zeros(len(clusters[k]))
                ps = clusters[k].astype(int)
                # logging.info(f'PS {ps}')
                for j in range(len(ps)):
                    sumDist[j] = np.sum([self.distMat[ps[j],q] for q in ps])

                c[i] = ps[np.argsort(sumDist)[0]]
                logging.debug(f'[{i}] center = {c[i]}')
            logging.debug(f'centers:\n{centers}\nc:\n{c}')

            if np.all(np.isin(c,centers)):
                logging.info(f'No Changes! Convergence, Iter = {iteration}')
                convergence = True
            else:
                logging.info('Changes!')
                centers= c
                if len(centers)<3:
                    logging.info(f'centers = {centers}')
                iteration+=1
        logging.debug(f'Modes:\n{centers}\nPoints:\n{points}\nClusters:\n{clusters}')
        return centers, points,clusters

    def fit(self):  
        logging.info('fit')
        self.computeDistanceMatrix()
        if len(self.k) ==1:
            # single layer
            if not self.exact:
                # NOM:
                # 1.  Construction phase
                tic = time.time()
                pc_val,pc_ind,pc_sco = self.KNN(range(self.length),self.k[0],self.distMat)
                logging.info(f'Construction Time: {time.time()-tic:.2f}')

                # 2.  Selection phase
                tic = time.time()
                modes,points = self.Selection(range(self.length),pc_ind,pc_sco)
                logging.info(f'Selection Time: {time.time()-tic:.2f}')
                # 3.  Assignment phase
                tic = time.time()
                centers, points,clusters = self.Assignment(range(self.length), modes,self.distMat)
                logging.info(f'Assignment Time: {time.time()-tic:.2f}')

                self.clusterIndex = [points[i] for i in range(len(points))]

            else:
                # EOM:
                curK=self.k[0]
                modes = []
                points =[]
                remainP = range(self.length)
                while len(modes) <self.exact and curK>1:
                    logging.info(f'K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')
                    # 1.  Construction phase
                    tic = time.time()
                    Dist = np.zeros((len(remainP),len(remainP)))
                    for i,c in enumerate(remainP):
                        Dist[i] = [self.distMat[c,j] for j in remainP]

                    pc_val,pc_ind,pc_sco = self.KNN(remainP,curK,Dist)
                    logging.info(f'Construction Time: {time.time()-tic:.2f}')
                    # 2.  Selection phase
                    tic = time.time()
                    m,p = self.Selection(remainP,pc_ind,pc_sco)
                    logging.info(f'Selection Time: {time.time()-tic:.2f}')
                    remainP = remainP[remainP!=p]
                    modes = np.concatenate((modes,m)).astype(int)
                    curK-=1
                    logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(modes)} ')
                # 3.  Assignment phase
                tic = time.time()
                centers, points,clusters = self.Assignment(range(self.length), modes,self.distMat)
                logging.info(f'Assignment Time: {time.time()-tic:.2f}')

                self.clusterIndex = [points[i] for i in range(len(points))]
        else:
            # Multi layers
            if not self.exact:
                # NOM
                # First Layer:
                # 1.1  Construction phase
                logging.info(f'\n{"-"*80}')
                tic = time.time()
                pc_val_l1,pc_ind_l1,pc_sco_l1 = self.KNN(range(self.length),self.k[0],self.distMat)
                logging.info(f'First Layer: Construction Time: {time.time()-tic:.2f}')

                # 1.2  Selection phase
                tic = time.time()
                modes_l1,p_l1 = self.Selection(range(self.length),pc_ind_l1,pc_sco_l1)
                
                logging.info(f'First Layer: Selection Time: {time.time()-tic:.2f}')
                logging.info(f'First Layer: Num of Modes: {len(modes_l1)}')
                # datasPx = [self.data[i,0] for i in modes]
                # datasPy = [self.data[i,1] for i in modes]
                # _,ax = plt.subplots(1,3,sharex=True,sharey=True)
                # ax[0].scatter(datasPx,datasPy,marker='.',c='r')
                # ax[0].set_title('FirstLayer')
                # 3.  Assignment phase
                tic = time.time()
                centers_l1, points_l1,clusters_l1 = self.Assignment(range(self.length), modes_l1,self.distMat)
                logging.info(f'First Layer: Assignment Time: {time.time()-tic:.2f}')
                logging.info(f'First Layer: Assignment points {len(points_l1)}')
                logging.info(f'\n{"-"*80}')
                # logging.info(f'centers: {centers}')
                # Second Layer:
                # 2.1  Construction phase
                tic = time.time()

                Dist = np.zeros((len(centers_l1),len(centers_l1)))
                for i,c in enumerate(centers_l1):
                    Dist[i] = [self.distMat[c,j] for j in centers_l1]
                # print(Dist)
                if self.geo>0:
                    Dist =self.nlinmap(Dist,geo)
                    logging.info('Geo Done!')

                pc_val_l2,pc_ind_l2,pc_sco_l2 = self.KNN(centers_l1,self.k[1],Dist)
                logging.info(f'Second Layer: Construction Time: {time.time()-tic:.2f}')
                
                # 2.2  Selection phase
                tic = time.time()
                modes_l2,p_l2 = self.Selection(p_l1,pc_ind_l2,pc_sco_l2)
                logging.info(f'Second Layer: Selection Time: {time.time()-tic:.2f}')
                logging.info(f'Second Layer: Num of Modes: {len(modes_l2)}')
                logging.info(f'Second Layer: Selection modes {modes_l2}')

                # 407 870

                # 2.3  Assignment phase
                tic = time.time()
                # index in centers_l1
                modes = [np.where(centers_l1==i)[0][0] for i in modes_l2]
                print(modes)
                centers_l2, points_l2,clusters_l2 = self.AssignmentGeo(range(len(centers_l1)), modes,Dist)
                logging.info(f'Second Layer: Assignment modes {centers_l2}')
                centers_l2 = [centers_l1[i] for i in centers_l2]
                points_final = {}
                for k in points_l2.keys():
                    points_final[centers_l1[k]] = centers_l1[points_l2[k]]
                logging.info(f'Second Layer: Assignment Time: {time.time()-tic:.2f}')
                logging.info(f'Second Layer: Assignment modes {centers_l2}')
                logging.info(f'Second Layer: Assignment points {len(points_final)}')

                logging.info(f'\n{"-"*80}')
                # 3 Final Assignment phase
                # tic = time.time()
                # modes_lf, points_lf,clusters_lf= self.Assignment(range(self.length), centers_l2,self.distMat)
                # logging.info(f'Final Assignment Time: {time.time()-tic:.2f}')

                self.clusterIndex = [points_final[points_l1[i]] for i in range(self.length)]

        logging.debug(np.unique(self.labelTrue))
        logging.debug(np.unique(self.clusterIndex))
        pred={}
        pred['True'] = self.labelTrue
        pred['Pred'] = self.clusterIndex
        pd.DataFrame(pred,columns=['True','Pred']).to_csv(path.join(self.resultPath,'pred.csv'),index=False)
    
        return self
    
    def nlinmap(self,D,K):
        N = D.shape[0]
        INF = 1000*np.max(np.max(D))*N
        ind = np.argsort(D)
        for i in range(N):
            D[i,ind[i,1+K:]] = INF
        # 1  2  3     1  4  7      1  2  3
        # 4  5  6     2  5  8  ->  2  5  6  
        # 7  8  9     3  6  9      3  6  9
        D = np.minimum(D,D.T)
    
        #               k=0                                 
        # 1 2 3 4 5     1 2 3 4 5   1 1 1 1 1    2 3 4 5 6     1 2 3 4 5
        # 2 3 4 5 1     1 2 3 4 5   2 2 2 2 2    3 4 5 6 7     2 3 4 5 1
        # 3 4 5 1 2  -> 1 2 3 4 5 + 3 3 3 3 3 =  4 5 6 7 8  -> 3 4 5 1 2
        # 4 5 1 2[3]    1 2 3 4[5]  4 4 4 4[4]   5 6 7 8[9]    4 5 1 2 3        1 2 3 4 3
        # 5 1 2 3 4     1 2 3 4 5   5 5 5 5 5    6 7 8 9 10    5 1 2 3 4        2 3 4 5 1
        #               k=1                                                 ->  3 4 5 1 2
        #               2 3 4 5 1   2 2 2 2 2    4 5 6 7 3     1 2 3 4 3        4 5 1 2 3
        #               2 3 4 5 1   3 3 3 3 3    5 6 7 8 4     2 3 4 5 1        3 1 2 3 4
        #               2 3 4 5 1 + 4 4 4 4 4 =  6 7 8 9 5 ->  3 4 5 1 2
        #               2 3 4 5 1   5 5 5 5 5    7 8 9 106     4 5 1 2 3
        #               2 3 4 5 1   1 1 1 1 1    3 4 5 6 1     3 1 2 3 4
        #               k=2
        #               3 4 5 1 2   3 3 3 3 3    6 7 8 4 5     1 2 3 4 5    ->  1 2 3 4 3
        #
        # D[i,j], D[0,j]+D[i,0], D[1,j]+D[i,1]
        # D[3,4]=3, 9          , 6

        for i in tqdm(range(N),desc='nlinmap'):
            mat= np.matlib.repmat(D[:,i],N,1).T+np.matlib.repmat(D[i,:],N,1)
            D = np.minimum(D,mat)
        return D

    def show(self):
        tic = time.time()
        logging.info('Show')
        _,ax= plt.subplots(1,2,sharex=True,sharey=True)
        # ax[0].scatter(self.data[:,0],self.data[:,1],marker='.')
        unique_labels = np.unique(self.labelTrue)

        unique_modes=np.unique(self.clusterIndex)
        logging.info(f'Label:{len(unique_labels)}  Pred:{len(unique_modes)}')
        logging.info(f'Label:{unique_labels}\nPred:{unique_modes}')
        np.random.seed(19680801)
        colors = np.random.rand(max(len(unique_labels),len(unique_modes)),3)
        
        co_true ={}
        cluster_label={}
        cluster_pred ={}
        for i,m in enumerate(unique_labels):
            co_true[m]=colors[i]
            cluster_label[m] = self.data[self.labelTrue==m]
        # logging.info(f'Cluster Label\n{cluster_label}')
        co_pred={}
        for i,m in enumerate(unique_modes):
            co_pred[m] = colors[i]
            cluster_pred[m] = self.data[self.clusterIndex==m]
        # logging.info(f'Cluster Pred\n{cluster_pred}')        
        # print(co)
        for k in cluster_label.keys():
            ax[0].scatter(cluster_label[k][:,0],cluster_label[k][:,1],marker='.',color=co_true[k])
        ax[0].set_aspect(1)
        ax[0].set_title('True')
        for k in cluster_pred.keys():
            ax[1].scatter(cluster_pred[k][:,0],cluster_pred[k][:,1],marker='.',color=co_pred[k])
        ax[1].set_aspect(1)
        ax[1].set_title('Pred')
        
        plt.tight_layout(pad=1)
        logging.info(f'Show Time: {time.time()-tic:.2f}')
        # plt.show()
        plt.savefig(path.join(self.resultPath,'result.png'))
            

if __name__ == '__main__':

    dataset = 'ncircles'

    config = configparser.ConfigParser()
    config.read('config.ini')
    logging.basicConfig(level=logging.INFO,format= '%(message)s',handlers=[
        logging.FileHandler(f'{config[dataset]["Log"]}',mode='a'),
        logging.StreamHandler()
    ],datefmt="%Y-%m-%d %H:%M:%S",force=True)

    data = pd.read_csv(config[dataset]["Data"],header=None)
    data_true=pd.read_csv(config[dataset]["Data_label"],header=None)[0].tolist()
    exact = int(config[dataset]["Exact"])
    Ks = config[dataset]["K"][1:-1].split(',')
    K = [int(k) for k in Ks]
    if exact==0:
        exact = False
    geo = int(config[dataset]["Geo"])
    tic = time.time()
    knet = KNet(np.array(data),k=K,labelTrue=data_true,resultPath=config[dataset]["Result"],exact=exact,geo=geo)
    knet.fit().show()
    logging.info(f'Time: {time.time()-tic:.2f}')