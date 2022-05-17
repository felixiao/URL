from cv2 import log
from sklearn.cluster import *
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import configparser
import time
from knet2 import KNet
import logging
from sklearn.mixture import GaussianMixture
from sklearn.metrics import DistanceMetric
from sklearn.datasets import make_blobs, make_checkerboard, make_circles,make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import *
from os import path,mkdir
from itertools import cycle, islice

class Cluster():
    def __init__(self,config):
        self.clusters = {}
        self.metrics ={}
        self.config = config
        

    def fit_predict(self,dataname, data,label_true=None):
        self.data= data
        self.label_true_ = label_true
        dataconfig = self.config[dataname]
        self.loadFile = dataconfig.getboolean('Load')
        # estimate bandwidth for mean shift
        bandwidth = estimate_bandwidth(self.data, quantile=float(dataconfig["quantile"]))
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(self.data, n_neighbors=int(dataconfig["n_neighbors"]), include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        Ks          = dataconfig["K"][1:-1].split(',')
        K           = [int(k) for k in Ks]
        exact       = int(dataconfig["Exact"])
        exact       = False if exact==0 else exact
        geo         = int(dataconfig["Geo"])
        multilayer  = dataconfig["Multilayer"]
        n_clusters  = int(dataconfig["NClusters"])
        eps         = float(dataconfig["eps"])
        min_samples = int(dataconfig["min_samples"])
        xi          = float(dataconfig["xi"])
        damping     = float(dataconfig["damping"])
        preference  = int(dataconfig["preference"])
        min_cluster_size = float(dataconfig["min_cluster_size"])
        self.filename = f'K{K}-E{exact}-G{geo}-{multilayer if len(K)>1 else "single"}'

        if self.config['ClusterModel']['KNets'] =='1':
            self.clusters['KNets']              = KNet(k=K,exact=exact,geo=geo,multilayer=multilayer)
        if self.config['ClusterModel']['MeanShift']=='1':
            self.clusters['MeanShift']          = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        if self.config['ClusterModel']['KMeans']=='1':
            self.clusters['KMeans']             = KMeans(n_clusters=n_clusters)
        if self.config['ClusterModel']['MiniBatchKMeans']=='1':
            self.clusters['MiniBatchKMeans']    = MiniBatchKMeans(n_clusters=n_clusters)
        if self.config['ClusterModel']['Ward']  =='1':
            self.clusters['Ward']               = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
        if self.config['ClusterModel']['Spectral']=='1':
            self.clusters['Spectral']           = SpectralClustering(n_clusters=n_clusters,eigen_solver="arpack",affinity="nearest_neighbors")
        if self.config['ClusterModel']['DBSCAN']=='1':
            self.clusters['DBSCAN']             = DBSCAN(eps=eps)
        if self.config['ClusterModel']['OPTICS']=='1':
            self.clusters['OPTICS']             = OPTICS(min_samples=min_samples,xi=xi,min_cluster_size=min_cluster_size)
        if self.config['ClusterModel']['AP']    =='1':
            self.clusters['AP']                 = AffinityPropagation(damping=damping, preference=preference, random_state=0)
        if self.config['ClusterModel']['AvgLinkage']=='1':
            self.clusters['AvgLinkage']         = AgglomerativeClustering(n_clusters=n_clusters,linkage="average",affinity="cityblock",connectivity=connectivity)
        if self.config['ClusterModel']['Birch'] =='1': 
            self.clusters['Birch']              = Birch(n_clusters=n_clusters)
        if self.config['ClusterModel']['GaussianMixture']=='1':
            self.clusters['GaussianMixture']    = GaussianMixture(n_components=n_clusters, covariance_type="full")
    
        
        for c in self.clusters.keys():
            logging.info(f'\nRunning clustering with {c}')
            tic = time.time()
            label_pred = self.clusters[c].fit_predict(self.data)
            self.metrics[c] = self.getmetrics(self.data,self.label_true_,label_pred)
            if hasattr(self.clusters[c], "inertia_"):
                self.metrics[c]['MSE'] = f'{self.clusters[c].inertia_/len(self.data):.3f}'
            if c == 'GaussianMixture':
                self.gmm_label = label_pred
            self.metrics[c]['Time'] = f'{time.time()-tic:.2f}s'.lstrip("0")
            logging.info(f'Finish in {self.metrics[c]["Time"]}')

    def getmetrics(self,data,label_true,label_pred):
        if label_true is None:
            return None
        logging.info(f'Label True {np.unique(label_true)}\nLabel Pred {np.unique(label_pred)}')
        n_clusters = len(set(label_pred)) - (1 if -1 in label_pred else 0)
        mtrc ={}
        mtrc['#Samples']        = len(data)
        mtrc['#Cluster']        = n_clusters
        mtrc['Time']            = ''
        mtrc['MSE']             = 'n/a'
        mtrc['Homogeneity']     = f'{homogeneity_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['Completeness']    = f'{completeness_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['V-measure']       = f'{v_measure_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['FowlkesMallows']  = f'{fowlkes_mallows_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['RandIndex']       = f'{rand_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['Adj.RandIndex']   = f'{adjusted_rand_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['Adj.MutualInfo']  = f'{adjusted_mutual_info_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['Norm.MutualInfo'] = f'{normalized_mutual_info_score(label_true, label_pred)*100:.2f}%'.replace(".00",'')
        mtrc['SilhouetteCoefficient'] = f'{silhouette_score(data, label_pred)*100:.2f}%'.replace(".00",'') if len(np.unique(label_pred))>1 else 'n/a'
        mtrc['MutualInfo']      = f'{mutual_info_score(label_true, label_pred):.2f}'.replace(".00",'')

        logging.info("Number of samples:       %d" % len(data))
        logging.info("Number of clusters:      %d" % n_clusters)
        logging.info("Mean Squared Error:      %s" % mtrc['MSE'] )
        logging.info("Homogeneity:             %s" % mtrc['Homogeneity'])
        logging.info("Completeness:            %s" % mtrc['Completeness'])
        logging.info("V-measure:               %s" % mtrc['V-measure'])
        logging.info("Fowlkes Mallows:         %s" % mtrc['FowlkesMallows'])
        logging.info("Rand Index:              %s" % mtrc['RandIndex'])
        logging.info("Adjusted Rand Index:     %s" % mtrc['Adj.RandIndex'])
        logging.info("Mutual Information:      %s" % mtrc['MutualInfo'])
        logging.info("Adjusted Mutual Info:    %s" % mtrc['Adj.MutualInfo'])
        logging.info("Normalized Mutual Info:  %s" % mtrc['Norm.MutualInfo'])
        logging.info("Silhouette Coefficient:  %s" % mtrc['SilhouetteCoefficient'])
        return mtrc

    def plotcluster(self,axid,name,label,time='.00s',nmi='0%',centers=None,nrow=1,ncol=1):
        # if self.loadFile:
        #     self.data= StandardScaler().fit_transform(self.data)
        plt.subplot(nrow, ncol, axid)
        plt.title(name, size=18,pad=-1)
        colors = np.array(list(islice(cycle([
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",]),
                        int(max(label) + 1),)))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(self.data[:, 0], self.data[:, 1], s=10, color=colors[label])
        if centers is not None:
            # if self.loadFile:
            #     centers = StandardScaler().fit_transform(centers)
            logging.info(f'Centers: {centers}')
            plt.scatter(centers[:, 0], centers[:, 1], s=10, marker= 's',color="#000000")
        xmax = np.max(self.data[:,0])
        xmin = np.min(self.data[:,0])
        ymax = np.max(self.data[:,1])
        ymin = np.min(self.data[:,1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xticks(())
        plt.yticks(())
        plt.text(0.99,0.01,f'{time}\nNMI: {nmi}',transform=plt.gca().transAxes,size=15,horizontalalignment="right")

    def show(self,resultpath,displayConfig):
        nrow= int(displayConfig["Row"])
        unitsize = int(displayConfig["Unit"])
        left = float(displayConfig["left"])
        right = float(displayConfig["right"])
        top = float(displayConfig["top"])
        bottom = float(displayConfig["bottom"])
        wspace = float(displayConfig["wspace"])
        hspace = float(displayConfig["hspace"])
        ncol = int(displayConfig["Col"])
        plt.figure(figsize=(unitsize*ncol, unitsize*nrow))
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plot_num = 1
        logging.info(f'\nShowing ground truth')
        
        self.plotcluster(axid=plot_num,name='GroundTruth',label=self.label_true_,nrow=nrow,ncol=ncol)
        for c_name,c_model in self.clusters.items():
            plot_num+=1
            logging.info(f'Showing {c_name}')
            ct = None
            if hasattr(c_model, "cluster_centers_"):
                ct = c_model.cluster_centers_
            self.plotcluster(axid=plot_num,name=c_name,time=self.metrics[c_name]['Time'],
                            nmi=self.metrics[c_name]['Norm.MutualInfo'],centers=ct,
                            label=self.gmm_label if c_name == 'GaussianMixture' else c_model.labels_,nrow=nrow,ncol=ncol)

        metrics_df = pd.DataFrame(self.metrics,columns=self.metrics.keys())
        logging.info(metrics_df)
        # plt.show()
        plt.savefig(path.join(resultpath,f'{self.filename}-compare.png'))

class DatasetHandle():
    def __init__(self,config):
        self.config = config
    
    def getData(self,dataname):
        if self.config.getboolean(dataname,"Load"):
            path = self.config[dataname]["Data"]
            logging.info(f'Load Data from {path}')
            data = pd.read_csv(path,header=None).to_numpy()
            label= pd.read_csv(self.config[dataname]["Data_label"],header=None)[0].tolist()
            # data = StandardScaler().fit_transform(data)
            return data, label
        elif dataname == 'noisy_circles':
            data, label= make_circles(n_samples=int(self.config[dataname]["n_samples"]), 
                                    factor=float(self.config[dataname]["factor"]), 
                                    noise=float(self.config[dataname]["noise"]))
        elif dataname == 'noisy_moons':
            data, label= make_moons(n_samples=int(self.config[dataname]["n_samples"]),
                                    noise=float(self.config[dataname]["noise"]))
        elif dataname == 'blobs':
            data, label= make_blobs(n_samples=int(self.config[dataname]["n_samples"]),
                                    random_state=int(self.config[dataname]["random_state"]))
        elif dataname == 'varied':
            clu_stds = self.config[dataname]["cluster_std"][1:-1].split(',')
            clu_std = [float(k) for k in clu_stds]
            data, label= make_blobs(n_samples=int(self.config[dataname]["n_samples"]),
                                    cluster_std=clu_std,
                                    random_state=int(self.config[dataname]["random_state"]))
        elif dataname == 'aniso':
            X, label = make_blobs(n_samples=int(self.config[dataname]["n_samples"]),
                                random_state=int(self.config[dataname]["random_state"]))
            transformation = [[0.6, -0.6], [-0.4, 0.8]]
            data = np.dot(X, transformation)
        elif dataname == 'no_structure':
            data, label = np.random.rand(int(self.config[dataname]["n_samples"]), 2),None
        # normalize dataset for easier parameter selection
        data = StandardScaler().fit_transform(data)
        return data, label

if __name__ == '__main__':
    # logging utilities
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Logging utilities
    dataname = config['ClusterModel']['Dataset']
    logging.basicConfig(level=logging.INFO,format= '%(message)s',handlers=[
        logging.FileHandler(f'{config[dataname]["Log"]}',mode='w'),
        logging.StreamHandler()
    ],datefmt="%Y-%m-%d %H:%M:%S",force=True)
    # read data
    dataset = DatasetHandle(config)
    data,label = dataset.getData(dataname)
    # create cluster
    cluster = Cluster(config)
    cluster.fit_predict(dataname,data,label)
    if not path.exists(config[dataname]["ResultPath"]):
        mkdir(config[dataname]["ResultPath"])
    
    cluster.show(config[dataname]["ResultPath"],config["Display"])