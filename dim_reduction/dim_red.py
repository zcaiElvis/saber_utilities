import pandas as pd
import multiprocessing
from sklearn import preprocessing
from itertools import product
from sklearn.decomposition import PCA
import time


sag_read = pd.read_csv('try_sag_tetra_df.tsv', sep = '\t')
sag = sag_read.values

class Dim_Red:
    
    def __init__(self, train, test = [1,1]):
        self.train = train
        self.test = test
        (self.train, self.test) = self.data_prep()
        
        self.best_model = None
        self.models = []
        
    def data_prep(self):
        '''
        Add Customize Functions for preprocessing
        '''
        self.train = preprocessing.scale(self.train)
        self.test = preprocessing.scale(self.test)
        return (self.train, self.test)
    
    
    def pca(self, min_exp_var, workers = 1, comp_range = range(1,30)):
        '''
        pca multiprocessing caller
        '''
        pca_model = PCA(whiten = True)
        process = multiprocessing.Pool(1)
        self.models = process.starmap(self.getModel, product([pca_model], comp_range, [min_exp_var]))
        self.models = [x for x in models if x is not None]
        return self.models
    
    
    def getModel(self, model, comp, min_exp_var):
        '''
        worker for multiprocessing
        '''
        model.n_components = comp
        model.fit(self.train)
        exp_var = model.explained_variance_ratio_
        if sum(exp_var) > min_exp_var:
            return model
        return
    
    def getBestModel(self):
        '''
        get best model from pca
        '''
        min_comp_index = 0
        for m in range(0, len(self.models)):
            if self.models[m].n_components < self.models[min_comp_index].n_components:
                min_comp_index = m
                
        return self.models[min_comp_index]
            
    
    
if __name__ == "__main__":
            
    t = time.time()
    dimred = Dim_Red(sag)    
    models = dimred.pca_new(0.9, workers = 5)
    print(time.time()-t)