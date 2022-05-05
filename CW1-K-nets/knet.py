import logging
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
import configparser
import time

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
    def __init__(self,data,k,labelTrue = None,exact=False,geo=3,struct=False,metric='euclidean',predPath='pred.csv'):
        self.data=data
        self.length=self.data.shape[0]
        self.k=[k] if type(k)==int else k
        self.idx=None
        self.vals=None
        self.Pnts_Scores=None
        self.clusterIndex = np.zeros(self.length).astype(int)
        self.labelTrue = labelTrue
        self.exact = exact
        self.modes = []
        self.list_points = []
        self.predPath = predPath
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

    def KNN(self,indexs,k):
        logging.debug(f'KNN K={k} Index:\n {indexs}')
        pc_val = {}
        pc_ind = {}
        pc_sco = {}
        pc_sco_v = []
        # for i in range(self.length):
        for i in indexs:
            dist = [self.distMat[i,j] for j in indexs]
            # dist = self.distMat[i,:]
            sv = np.sort(dist)
            si = np.argsort(dist)
            pc_val[i] = sv[:k]
            pc_ind[i] = [indexs[j] for j in si[:k]]
            pc_sco[i] = np.mean(pc_val[i])
            pc_sco_v.append(np.mean(pc_val[i]))
        pc_sco_i_sort = [indexs[j] for j in np.argsort(pc_sco_v)]
        logging.debug(f'KNN val:\n{pc_val}\nind:\n{pc_ind}\nsco:\n{pc_sco_i_sort}')
        return pc_val,pc_ind,pc_sco_i_sort

    def Selection(self,indexs,pc_ind,pc_sco):
        logging.debug(f'Selection:\n{indexs}\nInd:\n{pc_ind}\nPC Score:\n{pc_sco}')
        modes = []
        points= []
        for i in pc_sco:
            isin = np.isin(pc_ind[i],points)
            if not np.any(isin):
                points = np.concatenate((points,pc_ind[i])).astype(int)
                modes = np.concatenate((modes,[i])).astype(int)
        logging.debug(f'Modes\n{modes}\nPoints\n{points}')
        return modes,points

    def Assignment(self,indexs,modes):
        centers = modes
        if self.exact:
            centers = centers[:self.exact]
        convergence = False
        while not convergence:
            clusters = {}
            points = {}
            for m in centers:
                clusters[m] = []
            logging.debug(f'cluster:{clusters}')
            for i in indexs:
                ls_d=[self.distMat[i,j] for j in centers]
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
                logging.info('No Changes! Convergence')
                convergence = True
            else:
                logging.info('Changes!')
                centers= c
        logging.debug(f'Modes:\n{centers}\nPoints:\n{points}\nClusters:\n{clusters}')
        return centers, points,clusters

    def construction(self,k):
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

        # logging.debug(f'DistMat\n{self.distMat}')
        
        self.pre_clusters_val = np.zeros((self.length,k))
        self.pre_clusters_ind = {}
        self.pre_clusters_sco = np.zeros(self.length)
        # in EOM only the remain points are selected for KNN
        for i in range(self.length):
            dist = self.distMat[i,:]
            sv = np.sort(dist)
            si = np.argsort(dist)
            num_p = 0
            self.pre_clusters_ind[i] = []
            for j,sortI in enumerate(si):
                if sortI not in self.list_points:
                    self.pre_clusters_ind[i] = np.concatenate((self.pre_clusters_ind[i],[sortI]))
                    self.pre_clusters_val[i][num_p] = sv[j]
                    num_p +=1
                    if num_p>=k:
                        break
                
            # self.pre_clusters_val[i] = sv[:k]
            # # self.pre_clusters_ind[i] = np.concatenate(([i],si[:k]))
            # self.pre_clusters_ind[i] = si[:k]
            self.pre_clusters_sco[i] = np.mean(self.pre_clusters_val[i])
        # logging.debug(f'KNN index\n{self.pre_clusters_ind}')
        # logging.debug(f'KNN value\n{self.pre_clusters_val}')
        # logging.debug(f'KNN score\n{self.pre_clusters_sco}')
    
    def selection(self):
        sv = np.sort(self.pre_clusters_sco)
        si = np.argsort(self.pre_clusters_sco)
        # logging.debug(f'Pre Cluster min to max\n{si}\n{sv}\n{self.pre_clusters_ind}')

        for i in si:
            isin = np.isin(self.pre_clusters_ind[i],self.list_points)
            # logging.debug(f'[{i}] {isin} {self.pre_clusters_ind[i]}')
            if not np.any(isin):
                # self.list_points = np.concatenate((self.list_points,[si[i]]))
                self.list_points = np.concatenate((self.list_points,self.pre_clusters_ind[i]))
                self.modes = np.concatenate((self.modes,[i]))
                # logging.debug(f'[{i}] Add Mode {self.pre_clusters_ind[i]}')
        
        # logging.debug(f'Modes\n{self.modes}')
        # logging.debug(f'List Points\n{self.list_points}')

    def assignment(self):
        if self.exact:
            self.modes=self.modes[:self.exact]
        self.nearInd = np.zeros((self.length,self.length))
        for i in range(self.length):
            self.nearInd[i]=np.argsort(self.distMat[i,:])
        # logging.debug(f'Nearest Index\n{self.nearInd}')
        logging.debug('Nearst Mode')
        
        convergence = False
        while not convergence:
            clusters = [[] for i in self.modes]

            for i in range(self.length):
                v = [self.nearInd[i].tolist().index(m) for m in self.modes]
                si= np.argsort(v)
                mo= self.modes[si[0]]
                # logging.debug(f'[{i}] {v} {si} {mo}')
                self.clusterIndex[i] = i if i in self.modes else int(mo)
                ind = self.modes.tolist().index(self.clusterIndex[i])
                clusters[ind].append(i)
            # logging.debug(self.clusterIndex)
            # logging.debug(f'Clusters\n{clusters}')
            # find the center point of the clusters
            center=np.zeros(len(self.modes))
            for i,c in enumerate(clusters):
                dataC = self.data[c,:]
                distMat = distance_matrix(dataC,dataC)
                sumDist = np.zeros(len(dataC))
                for j in range(len(dataC)):
                    sumDist[j] =np.sum(distMat[j,:])
                center[i] = c[np.argsort(sumDist)[0]]
            if np.all(np.isin(center,self.modes)):
                logging.debug('No Changes! Convergence')
                # logging.debug(center)
                convergence = True
            else:
                logging.debug('Changes!')
                self.modes= center
                # logging.debug(self.modes)

    def layerConstruction(self,k,data):
        distMat = distance_matrix(data,data)

        pre_clusters_val = np.zeros((len(data),k))
        pre_clusters_ind = {}
        pre_clusters_sco = np.zeros(len(data))
        for i in range(len(data)):
            dist = distMat[i,:]
            sv = np.sort(dist)
            si = np.argsort(dist)
            num_p = 0
            pre_clusters_ind[i] = []
            for j,sortI in enumerate(si):
                if sortI not in self.list_points:
                    pre_clusters_ind[i] = np.concatenate((pre_clusters_ind[i],[sortI]))
                    pre_clusters_val[i][num_p] = sv[j]
                    num_p +=1
                    if num_p>=k:
                        break
        pre_clusters_sco[i] = np.mean(pre_clusters_val[i])
        return pre_clusters_val,pre_clusters_ind,pre_clusters_sco,distMat
    #   |                          Layer 1                    |                     Layer 2             |
    #       DistMat 1 2 3 4 5   KNN 1 2 3 4 5   Score     Mode  DistMat(Layer2) KNN      Score   Mode   Assignment
    # 1 1      1                1                1   d              2 3 4         2 3 4    
    # 2 2      2                2                2   a      2     2             2         2 b    2 [4]   2 [4,1]
    # 3 3  ->  3            ->  3            ->  3   c ->   3 ->  3         ->  3      -> 3 a -> 3       3 [5]
    # 4 4      4                4                4   b      4     4             4         4 c
    # 5 5      5                5                5   e
    def layerSelection(self,ind,sco):
        sv = np.sort(sco)
        si = np.argsort(sco)
        # logging.debug(f'Pre Cluster min to max\n{si}\n{sv}\n{self.pre_clusters_ind}')

        for i in si:
            isin = np.isin(ind[i],self.list_points)
            # logging.debug(f'[{i}] {isin} {self.pre_clusters_ind[i]}')
            if not np.any(isin):
                # self.list_points = np.concatenate((self.list_points,[si[i]]))
                self.list_points = np.concatenate((self.list_points,ind[i]))
                self.modes = np.concatenate((self.modes,[i]))
    
    def layerAssignment(self,distMat):
        if self.exact:
            self.modes=self.modes[:self.exact]
        length = distMat.shape[0]
        nearInd = np.zeros((length,length))
        for i in range(length):
            nearInd[i]=np.argsort(distMat[i,:])
        # logging.debug(f'Nearest Index\n{self.nearInd}')
        logging.debug('Nearst Mode')
        
        convergence = False
        while not convergence:
            clusters = [[] for i in self.modes]

            for i in range(length):
                v = [nearInd[i].tolist().index(m) for m in self.modes]
                si= np.argsort(v)
                mo= self.modes[si[0]]
                # logging.debug(f'[{i}] {v} {si} {mo}')
                self.clusterIndex[i] = i if i in self.modes else int(mo)
                ind = self.modes.tolist().index(self.clusterIndex[i])
                clusters[ind].append(i)
            # logging.debug(self.clusterIndex)
            # logging.debug(f'Clusters\n{clusters}')
            # find the center point of the clusters
            center=np.zeros(len(self.modes))
            for i,c in enumerate(clusters):
                dataC = self.data[c,:]
                distMat = distance_matrix(dataC,dataC)
                sumDist = np.zeros(len(dataC))
                for j in range(len(dataC)):
                    sumDist[j] =np.sum(distMat[j,:])
                center[i] = c[np.argsort(sumDist)[0]]
            if np.all(np.isin(center,self.modes)):
                logging.debug('No Changes! Convergence')
                # logging.debug(center)
                convergence = True
            else:
                logging.debug('Changes!')
                self.modes= center
                # logging.debug(self.modes)


    def fit(self):  
        logging.debug('fit')
        self.computeDistanceMatrix()
        if len(self.k) ==1:
            # single layer
            if not self.exact:
                # NOM:
                # 1.  Construction phase
                tic = time.time()
                pc_val,pc_ind,pc_sco = self.KNN(range(self.length),self.k[0])
                logging.info(f'Construction Time: {time.time()-tic:.2f}')

                # 2.  Selection phase
                tic = time.time()
                modes,points = self.Selection(range(self.length),pc_ind,pc_sco)
                logging.info(f'Selection Time: {time.time()-tic:.2f}')
                # 3.  Assignment phase
                tic = time.time()
                centers, points,clusters = self.Assignment(range(self.length), modes)
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
                    pc_val,pc_ind,pc_sco = self.KNN(remainP,curK)
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
                centers, points,clusters = self.Assignment(range(self.length), modes)
                logging.info(f'Assignment Time: {time.time()-tic:.2f}')

                self.clusterIndex = [points[i] for i in range(len(points))]
        else:
            # Multi layers
            if not self.exact:
                # NOM
                # First Layer:
                # 1.1  Construction phase
                remainP = range(self.length)
                tic = time.time()
                pc_val,pc_ind,pc_sco = self.KNN(remainP,self.k[0])
                logging.info(f'First Layer: Construction Time: {time.time()-tic:.2f}')

                # 1.2  Selection phase
                tic = time.time()
                modes,p = self.Selection(remainP,pc_ind,pc_sco)
                logging.info(f'First Layer: Selection Time: {time.time()-tic:.2f}')

                # Second Layer:
                # 2.1  Construction phase
                tic = time.time()
                pc_val,pc_ind,pc_sco = self.KNN(p,self.k[1])
                logging.info(f'Second Layer: Construction Time: {time.time()-tic:.2f}')
                # 2.2  Selection phase
                tic = time.time()
                m,p = self.Selection(p,pc_ind,pc_sco)
                logging.info(f'Second Layer: Selection Time: {time.time()-tic:.2f}')

                # # 2.3 Assignment phase
                # tic = time.time()
                # centers, points,clusters = self.Assignment(p, m)
                # logging.info(f'Second Layer: Assignment Time: {time.time()-tic:.2f}')

                # 3 Final Assignment phase
                tic = time.time()
                centers, points,clusters = self.Assignment(range(self.length), m)
                logging.info(f'Second Layer: Assignment Time: {time.time()-tic:.2f}')

                self.clusterIndex = [points[i] for i in range(len(points))]
                # # 1.1  Construction phase
                # tic = time.time()
                # self.construction(self.k[0])
                # logging.info(f'First Layer: Construction Time: {time.time()-tic:.2f}')

                # # 1.2  Selection phase
                # tic = time.time()
                # self.selection()
                # logging.info(f'First Layer: Selection Time: {time.time()-tic:.2f}')

                # # Second Layer:
                # # 2.1  Construction phase
                # tic = time.time()
                # selectData = self.data[self.list_points.astype(int),:]
                # val,ind,sco,distMat = self.layerConstruction(self.k[1],selectData)
                # logging.info(f'Second Layer: Construction Time: {time.time()-tic:.2f}')
                # # 2.2  Selection phase
                # tic = time.time()
                # self.layerSelection(ind,sco)
                # logging.info(f'Second Layer: Selection Time: {time.time()-tic:.2f}')
                # # 2.3 Assignment phase
                # tic = time.time()
                # self.layerAssignment()
                # logging.info(f'Second Layer: Assignment Time: {time.time()-tic:.2f}')



        logging.debug(np.unique(self.labelTrue))
        logging.debug(np.unique(self.clusterIndex))
        pred={}
        pred['True'] = self.labelTrue
        pred['Pred'] = self.clusterIndex
        pd.DataFrame(pred,columns=['True','Pred']).to_csv(self.predPath,index=False)
        

        # # Check the range of k values
        # if sum(np.where(np.array(self.k)<=0,1,0))>0:
        #     print(f'Error the value of the resolution parameter cannot be smaller than 1. (K={self.k})\n')
        #     return None
        # # Process input
        # Dists,Neighbs,data,knetstruct,c,kstep,dstep,pidx,metric,maxiters,resolve,nlin = self.process_input()

        # initialize=1
        # Dists, Neighbs, K = self.check_nans(Dists, Neighbs)
        return self
        
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
        # for i in range(self.length):
        #     ax[0].scatter(self.data[i,0],self.data[i,1],marker='+' if i in self.labelTrue else '.',color=co_true[self.labelTrue[i]])
        #     ax[1].scatter(self.data[i,0],self.data[i,1],marker='+' if i in self.modes else '.',color=co_pred[self.clusterIndex[i]])
        plt.tight_layout(pad=1)
        logging.info(f'Show Time: {time.time()-tic:.2f}')
        plt.show()
            

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    logging.basicConfig(level=logging.INFO,format= '%(message)s',handlers=[
        logging.FileHandler(f'{config["PATH"]["Data"]}.log',mode='a'),
        logging.StreamHandler()
    ],datefmt="%Y-%m-%d %H:%M:%S",force=True)
    # PlotData()
    data = pd.read_csv(config["PATH"]["Data"],header=None)
    data_true=pd.read_csv(config["PATH"]["Data_label"],header=None)[0].tolist()
    # data = np.random.rand(1000,2)*5
    tic = time.time()
    knet = KNet(np.array(data),k=[3,15],labelTrue=data_true,predPath=config["PATH"]["Data_pred"])
    knet.fit().show()
    # knet = KNet(np.array([[1,1],[2,2],[3,3],[4,4],[5,5]]),k=[2])
    # knet.computeDistanceMatrix()
    # pc_val,pc_ind,pc_sco = knet.KNN([0,1,2,3,4],2)
    # modes,_ = knet.Selection([0,1,2,3,4],pc_ind,pc_sco)
    # knet.Assignment([0,1,2,3,4],modes)
    logging.info(f'Time: {time.time()-tic:.2f}')
    # KNet(data,[1,2]).fit().show()