import scipy.io as sp
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn import preprocessing

def read_mat_file (filename):
    data = sp.loadmat(filename)
    trData = data['trData']
    trLabels = data['trLabels']
    teData = data['teData']
    teLabels = data['teLabels']
    idx = data['idx']
    return (trData, trLabels, teData, teLabels, idx)



def mvpa(trData, trLabels, teData, teLabels,c):
    model = linear_model.LogisticRegression(penalty = 'l2', C = c)
    model.fit(trData, trLabels[0])
    k = model.predict(teData)
    numhits = 0
    a,b=teLabels.shape
    return sum(sum(k == teLabels))/b
    



def fsg(trData, trLabels, teData, teLabels, idx,c=1):

    # trData = num_voxels - by- num_tr_samples
    # teData = num_voxels - by- num_te_samples
    # trLabels = num_tr_samples - by - 1
    # teLabels = num_te_samples - by - 1
    # idx = 1 - by- num_voxels (indicates the classifier each voxel belongs to) 

    # Please cite the following if you plan to use the code in your own work: 

    # * Ozay, M., Vural, F. (2012). A New Fuzzy Stacked Generalization Technique and Analysis of its Performance

    penaltyfunc = 'l2'
    
    
    num_samples_tr = len(trData.T)
    num_samples_te = len(teData.T)    
    
    teData_ = teData
    trData_ = trData
    idx = (idx.T)
    membership_tr = [[] for i in range(num_samples_tr)]
    membership_te = [[] for i in range(num_samples_te)]
    
    base_layer_performances = []
    num_clusters = 0
    trLabels = trLabels.reshape(1,len(trLabels)).T
    teLabels = teLabels.reshape(1,len(teLabels)).T

    for i in xrange(1,max(idx)+1):
        if i in idx:
            clustertr = []
            clusterte = []
            num_voxels_in_cluster = 0
            for k in range(len(trData_)):
                if idx[k] == [i]:
                    clustertr.append(trData_[k]) 
                    clusterte.append(teData_[k])
                    num_voxels_in_cluster += 1
            if num_voxels_in_cluster > 4:# and num_voxels_in_cluster < 10000:
                clustertr = np.asarray(clustertr).T
                clusterte = np.asarray(clusterte).T
        #    membership train
                pred = []
                for j in range(num_samples_tr):
                    test_sample = clustertr[j]
                    train_samples = np.delete(clustertr,j,axis = 0)
                    train_labels = np.delete(trLabels,j,axis = 0)
                    model = linear_model.LogisticRegression(penalty = penaltyfunc,C = c)
                    train_labels = train_labels.reshape([len(train_labels),])
                    model.fit(train_samples, train_labels)
                    k = model.predict_proba(test_sample.reshape(1,-1))
                    preds = model.predict(test_sample.reshape(1,-1))
                    membership_tr[j].extend((k).tolist()[0])
                    pred.append(preds)
                    
        #    membership test
                model = linear_model.LogisticRegression(penalty = penaltyfunc,C = c)
                asd = trLabels.reshape([len(trLabels),])
                model.fit(clustertr, asd)
                k = model.predict_proba(clusterte)
                numhits = 0
                for j in range(num_samples_te):
                    membership_te[j].extend(k[j])
                    if np.argmax(k[j])+1 == teLabels[j]:
                        numhits += 1
                base_layer_performances.append([float(numhits) * 100/num_samples_te])        
                num_clusters += 1
            
            
    model_meta = linear_model.LogisticRegression(penalty = penaltyfunc,C = c)
    asd = trLabels.reshape([len(trLabels),])
    model_meta.fit(membership_tr,asd)
    k = model_meta.predict(membership_te)
    numhits = 0
    for i in range(len(k)):
        if k[i] == teLabels[i]:
            numhits+=1
            
    performance = float(numhits) * 100 /num_samples_te     
    max_base = max(base_layer_performances)
    
    return [max_base, base_layer_performances,performance,membership_tr,membership_te]    



