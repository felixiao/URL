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
#       1.1.2 EOM
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

    # load data7
    data = pd.read_csv('Artificial/art_data/data7.csv',header=None)
    ax[0,1].scatter(data[0],data[1],marker='.')

    # load data4
    data = pd.read_csv('Artificial/art_data/data4.csv',header=None)
    ax[0,2].scatter(data[0],data[1],marker='.')

    # load data3
    data = pd.read_csv('Artificial/art_data/data3.csv',header=None)
    ax[1,0].scatter(data[0],data[1],marker='.')

    # load ncircles
    data = pd.read_csv('Artificial/art_data/ncircles.csv',header=None)
    ax[1,1].scatter(data[0],data[1],marker='.')

    # load data50
    data = pd.read_csv('Artificial/art_data/data50.csv',header=None)
    ax[1,2].scatter(data[0],data[1],marker='.')
    ax[1,2].ticklabel_format(scilimits=[-5,4])
    # load noisy_spiral
    data = pd.read_csv('Artificial/art_data/noisy_spiral.csv',header=None)
    ax[2,0].scatter(data[0],data[1],marker='.')

    # load birch_sin
    data = pd.read_csv('Artificial/art_data/birch_sin.csv',header=None)
    ax[2,1].scatter(data[0],data[1],marker='.')
    ax[2,1].ticklabel_format(scilimits=[-5,4])

    # load birch_grid
    data = pd.read_csv('Artificial/art_data/birch_grid.csv',header=None)
    ax[2,2].scatter(data[0],data[1],marker='.')

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
    def __init__(self,data,k,labelTrue = None,exact=False,geo=3,struct=False,metric='euclidean'):
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
        if exact:
            logging.info(f'Exact mode with exact {exact} clusters and Initial K = {k}')
        else:
            logging.info(f'Normal mode with Initial K = {k}')
    def computeDistanceMatrix(self):
        self.distMat = distance_matrix(self.data,self.data)
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


    def fit(self):  
        logging.debug('fit')
        self.computeDistanceMatrix()
        if not self.exact:
            # single layer NOM:
            # 1.  Construction phase
            tic = time.time()
            self.construction(self.k[0])
            logging.info(f'Construction Time: {time.time()-tic:.2f}')

            # 2.  Selection phase
            tic = time.time()
            self.selection()
            logging.info(f'Selection Time: {time.time()-tic:.2f}')
            # 3.  Assignment phase
            tic = time.time()
            self.assignment()
            logging.info(f'Assignment Time: {time.time()-tic:.2f}')
        else:
            # single layer EOM:
            curK=self.k[0]
            while len(self.modes) < self.exact and curK>1:
                logging.info(f'K = {curK}\tExact = {self.exact}\tModes = {len(self.modes)} ')
                # 1.  Construction phase
                self.construction(curK)
                # 2.  Selection phase
                self.selection()
                curK-=1
                logging.info(f'After K = {curK}\tExact = {self.exact}\tModes = {len(self.modes)} ')
            # 3.  Assignment phase
            self.assignment()


        logging.debug(np.unique(self.labelTrue))
        logging.debug(np.unique(self.clusterIndex))
        pred={}
        pred['True'] = self.labelTrue
        pred['Pred'] = self.clusterIndex
        pd.DataFrame(pred,columns=['True','Pred']).to_csv('Data4_pred.csv',index=False)
        

        # # Check the range of k values
        # if sum(np.where(np.array(self.k)<=0,1,0))>0:
        #     print(f'Error the value of the resolution parameter cannot be smaller than 1. (K={self.k})\n')
        #     return None
        # # Process input
        # Dists,Neighbs,data,knetstruct,c,kstep,dstep,pidx,metric,maxiters,resolve,nlin = self.process_input()

        # initialize=1
        # Dists, Neighbs, K = self.check_nans(Dists, Neighbs)
        return self
    
    def check_nans(self,Dists, Neighbs):
        # Check if there are nan values
        if (Neighbs is None or len(Neighbs)==0) and np.any(np.isnan(Dists)):
            print('Nan values detected resolving....\n')
            K = np.zeros(Dists.shape[1])
            nNeighbs = []
            nDists = []
            for i in range(Dists.shape[1]):
                # Get Non nan 
                no_nan_inds=np.where((np.where(np.isnan(a),0,1) + np.where(np.isinf(a),0,1))==2,1,0)*range(1,len(a)+1)
                no_nan_inds=no_nan_inds[no_nan_inds>0]-1
                arr = np.array(a)[no_nan_inds]
                nNeighbs[i] = no_nan_inds[np.argsort(arr)]
                nDists[i] = np.sort(arr)
                K[i] = len(no_nan_inds) if len(no_nan_inds)<self.k else self.k
            return nDists,nNeighbs,K
        elif Neighbs and len(Neighbs)>0:
            if self.k.shape[0]==self.k.shape[1]==1:
                K = self.k[0]*np.ones(len(Dists))
            return Dists,Neighbs,K
        else: return None, None,self.k

    def process_input(self,maxiters=100,exact=-1,resolve=1,geo=0,dstep=300,kstep=1,metric='euc',prior=None,struct=False):
        print('Process input')
        pidx = prior
        data=[]
        nlin=geo
        metric=metric
        # if the DN is a dictionary contains meta information 
        # then get the meta info and data
        if type(self.DN) == dict:
            pidx=self.DN['prior']
            meds=np.unique(pidx)
            metric=self.DN['metric']
            data=self.DN['data']
            # compute the dist matrix for data
            if data and len(data)>0:
                self.DN = self.distfun(data[meds,:], data[meds,:], metric, 0)
        
        if self.DN.shape[0]==2:
            Dists = self.DN[0]
            Neighbs = self.DN[1]
        else:
            Dists = self.DN
            Neighbs = None
        return Dists,Neighbs,data,struct,exact,kstep,dstep,pidx,metric,maxiters,resolve,nlin
    
    def distfun(self,X,C,dist,iter):
        n,p = X.shape
        D = np.zeros(n,C.shape[0])
        nclusts = C.shape[0]
        if dist == 'euc':
            for i in range(nclusts):
                D[:,i] = (X[:,0] - C[i,1])^2
                for j in range(1,p):
                    D[:,i] += (X[:,j] - C[i,j])^2
            D=D^0.5
        elif dist == 'seuclidean':
            for i in range(nclusts):
                D[:,i] = (X[:,0] - C[i,1])^2
                for j in range(1,p):
                    D[:,i] += (X[:,j] - C[i,j])^2
        elif dist == 'hamming':
            for i in range(nclusts):
                D[:,i] = abs(X[:,0] - C[i,1])
                for j in range(1,p):
                    D[:,i] += abs(X[:,j] - C[i,j])
                D[:,i] /= p
        elif dist == 'cityblock':
            for i in range(nclusts):
                D[:,i] = abs(X[:,0] - C[i,1])
                for j in range(1,p):
                    D[:,i] += abs(X[:,j] - C[i,j])
        return D
        
    def show(self):
        tic = time.time()
        logging.debug('Show')
        _,ax= plt.subplots(1,2,sharex=True,sharey=True)
        # ax[0].scatter(self.data[:,0],self.data[:,1],marker='.')
        unique_labels = np.unique(self.labelTrue)

        unique_modes=np.unique(self.clusterIndex)
        logging.debug(f'Label:{len(unique_labels)}  Pred:{len(unique_modes)}')
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
        for k in cluster_pred.keys():
            ax[1].scatter(cluster_pred[k][:,0],cluster_pred[k][:,1],marker='.',color=co_pred[k])
        ax[1].set_aspect(1)
        # for i in range(self.length):
        #     ax[0].scatter(self.data[i,0],self.data[i,1],marker='+' if i in self.labelTrue else '.',color=co_true[self.labelTrue[i]])
        #     ax[1].scatter(self.data[i,0],self.data[i,1],marker='+' if i in self.modes else '.',color=co_pred[self.clusterIndex[i]])
        plt.tight_layout(pad=1)
        logging.info(f'Show Time: {time.time()-tic:.2f}')
        plt.show()
            
def pdist(data):
    return distance_matrix(data,data)
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
    knet = KNet(np.array(data),k=150,labelTrue=data_true,exact=4)
    knet.fit().show()
    logging.info(f'Time: {time.time()-tic:.2f}')
    # KNet(data,[1,2]).fit().show()