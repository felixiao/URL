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


dataframe = pd.read_csv('data4_[3,50]_exact8_true.csv',header=None,index_col=False)
pd.DataFrame(dataframe.T[0].to_list()).to_csv('data4_true.csv',header=None,index=False)