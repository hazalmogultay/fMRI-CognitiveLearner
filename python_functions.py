import numpy as np
import scipy.io as sio
import os
import time
import time
import sklearn.feature_selection as feat_select
import sklearn.cross_validation as cross_val
from sklearn import linear_model
import voxel_selection as vs
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def fsg(info,trData, trLabels, teData, teLabels, idx,c, runlength = 30, select = 0,subj = 'subj0', fold = 'fold0'):
    penaltyfunc = 'l2'
    
    runlength = 35
    num_samples_tr = len(trData.T)
    num_samples_te = len(teData.T)    
    
    teData_ = teData
    trData_ = trData
    idx = (idx.T)
    numclasses  = np.unique(trLabels)
    numclasses = numclasses.shape[0]
    
    membership_tr = np.zeros([num_samples_tr,numclasses*max(idx)[0]])#[[] for i in range(num_samples_tr)]
    membership_te = np.zeros([num_samples_te,numclasses*max(idx)[0]])#[[] for i in range(num_samples_te)]
    
    base_layer_performances = []
    num_clusters = 0
    trLabels = trLabels.reshape(1,len(trLabels)).T
    teLabels = teLabels.reshape(1,len(teLabels)).T
    start_point = 1
    if info.base_layer_memberships > 0:
        start_point = 2
    if info.base_layer_memberships > 1:
    	start_point = 3
    for i in xrange(start_point,max(idx)+1):
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
                indi1 = 0
                indi2= runlength
                for j in range(num_samples_tr/runlength):
                    test_sample = clustertr[indi1:indi2,:]
                    train_samples = np.concatenate((clustertr[0:indi1,:],clustertr[indi2:,:]),axis=0)

                    train_labels = np.concatenate((trLabels[0:indi1,:],trLabels[indi2:,:]),axis=0)
                    model = linear_model.LogisticRegression(penalty = penaltyfunc,C = c)
                    train_labels = train_labels.reshape([len(train_labels),])
                    model.fit(train_samples, train_labels)
                    k = model.predict_proba(test_sample)
                    membership_tr[indi1:indi2,(i-1)*numclasses:(i-1)*numclasses+numclasses] = k
                    indi1 = indi2
                    indi2 = indi1+runlength
                    
        #    membership test
                model = linear_model.LogisticRegression(penalty = penaltyfunc,C = c)
                asd = trLabels.reshape([len(trLabels),])
                model.fit(clustertr, asd)
                k = model.predict_proba(clusterte)
                #print k.shape, membership_te.shape
                membership_te[:,(i-1)*numclasses:(i-1)*numclasses+numclasses] = k
                
                numhits = 0
                for j in range(num_samples_te):
                    #membership_te[j].extend(k[j])
                    if np.argmax(k[j])+1 == teLabels[j]:
                        numhits += 1
                base_layer_performances.append([float(numhits) * 100/num_samples_te])        
                num_clusters += 1
            
    #info.resultdir + subj + '_' + fold + '_' + 'cluster_' + str(info.cluster) + '_mesh_' + str(info.ismesh) +'_select_' + str(select) +'_memberships.mat'
    if info.base_layer_memberships > 1:
        fn_load = info.get_second_layer_memberships + subj + '_' + fold + '_' + 'cluster_' + str(info.cluster) + '_mesh_' + str(info.ismesh) + '_select_'+ str(select) + '_memberships.mat'
        mem_load = sio.loadmat(fn_load)
        mm_tr = np.array(mem_load['membership_tr']) 
        mm_te = np.array(mem_load['membership_te'])
        num_classes = np.unique(np.array(trLabels)).shape[0]
        mm_tr = mm_tr[:,2:num_classes]
        mm_te = mm_te[:,2:num_classes]
        membership_tr = np.concatenate((mm_tr, np.array(membership_tr)) ,axis = 1)
        membership_te = np.concatenate((mm_te, np.array(membership_te)) ,axis = 1)
    	
    if info.base_layer_memberships > 0:
        fn_load = info.get_base_layer_memberships + subj + '_' + fold + '_' + 'cluster_' + str(0) + '_mesh_' + str(info.ismesh) + '_select_'+ str(select) + '_memberships.mat'
        mem_load = sio.loadmat(fn_load)
        mm_tr = np.array(mem_load['membership_tr']) 
        mm_te = np.array(mem_load['membership_te'])
        num_classes = np.unique(np.array(trLabels)).shape[0]
        mm_tr = mm_tr[:,0:num_classes]
        mm_te = mm_te[:,0:num_classes]
        membership_tr = np.concatenate((mm_tr, np.array(membership_tr)) ,axis = 1)
        membership_te = np.concatenate((mm_te, np.array(membership_te)) ,axis = 1)
        
      
    model_meta = linear_model.LogisticRegression(penalty = penaltyfunc,C = c)
    asd = trLabels.reshape([len(trLabels),])
    model_meta.fit(membership_tr,asd)
    k = model_meta.predict(membership_te)
    '''
    model_meta = svm.SVC()
    asd = trLabels.reshape([len(trLabels),])
    model_meta.fit(membership_tr,asd)
    k = model_meta.predict(membership_te)
	'''  
    
    numhits = 0
    for i in range(len(k)):
        if k[i] == teLabels[i]:
            numhits+=1
            
    performance = float(numhits) * 100 /num_samples_te     
    max_base = max(base_layer_performances)
    
    return [max_base, base_layer_performances,performance,membership_tr,membership_te]    


def mvpa_second_layer(info,select = 0):

	'''
	sample_call
	for c in [0,1]:
		for m in [0,1,2,3]:
			for select in [0,1]:
				info.cluster = c
				info.ismesh = m
				mvpa_accuracies = call_mvpa(info,select)
				fn = info.resultdir + 'cluster_' + str(c) + '_mesh_' + str(m) + '_select_' + str(select) + '_mvpa.mat'
				sio.savemat(fn, {'accuracies' : mvpa_accuracies})
	'''
	all_subjects = os.listdir(info.datapath)
	all_subjects = np.sort(all_subjects)
	accuracies = np.zeros([len(all_subjects), 6])
	index1 = 0
	index2 = 0
	for subj in all_subjects:
		newdir = info.dir + subj + '/'
		all_folds = os.listdir(newdir)
		all_folds = np.sort(all_folds)
		index2 = 0
		for fold in all_folds:
			if info.cluster == 1:
				fn = newdir + fold + '/' + info.sl 

			else:
				fn = newdir + fold + '/' + info.slaal
			
			data = sio.loadmat(fn)

			trl = np.reshape(data['labels'],[len(data['labels']),])
			tel = np.reshape(data['labels_te'],[len(data['labels_te']),])

			if info.ismesh == 0:
				tr = data['data']
				te = data['data_te']
				trnew = np.zeros([tr.shape[0], len(trl)])
				for i in range(0,len(trl)):
					trnew[:,i] = tr[:,i*6+2]

				tenew = np.zeros([te.shape[0], len(tel)])
				for i in range(0,len(tel)):
					tenew[:,i] = te[:,i*6+2]
			elif info.ismesh > 1 and info.cluster == 0:
				p = (info.ismesh - 1) * 10
				fn1 = newdir + fold + '/p_'+ str(p) + '_' + info.slaal
				data_mesh = sio.loadmat(fn1)
				trnew = data_mesh['a_tr']
				tenew = data_mesh['a_te']
			elif info.ismesh > 1 and info.cluster == 1:
				p = (info.ismesh - 1) * 10
				fn1 = newdir + fold + '/p_'+ str(p) + '_' + info.sl
				data_mesh = sio.loadmat(fn1)
				trnew = data_mesh['a_tr']
				tenew = data_mesh['a_te']
			else:
				trnew = data['a_tr']
				tenew = data['a_te']

			trnew, tenew,f_scores, selected = weight_selection(trnew, tenew, trl,select)
			'''
			if select == 1 and trnew.shape[0] > trnew.shape[1] * 2:
				f_scores, p_val = feat_select.f_classif(trnew.T, trl)
				selected, ax = vs.find_important_features([f_scores], difference = 1, plotting = 0)
				indices = sorted(range(len(f_scores)),key=lambda x:f_scores[x])
				selected_ind = indices[int(selected):-1]
				trnew = [trnew[x] for x in selected_ind]
				tenew = [tenew[x] for x in selected_ind]
			'''

			trnew = np.array(trnew).T
			tenew = np.array(tenew).T

			

			

			model = linear_model.LogisticRegression(penalty = 'l2')
			model.fit(trnew, trl)
			k = model.predict(tenew)
			acc = sum(k==tel)*100/len(tel)
			accuracies[index1, index2] = acc
			index2 = index2 + 1
		index1 = index1 + 1 
	return accuracies


def mvpa_first_layer(info,select = 0):

	'''
	sample_call
	for c in [0,1]:
		for m in [0,1,2,3]:
			for select in [0,1]:
				info.cluster = c
				info.ismesh = m
				mvpa_accuracies = call_mvpa(info,select)
				fn = info.resultdir + 'cluster_' + str(c) + '_mesh_' + str(m) + '_select_' + str(select) + '_mvpa.mat'
				sio.savemat(fn, {'accuracies' : mvpa_accuracies})
	'''
	all_subjects = os.listdir(info.datapath)
	all_subjects = np.sort(all_subjects)
	accuracies = np.zeros([len(all_subjects), 6])
	index1 = 0
	index2 = 0
	for subj in all_subjects:
		newdir = info.dir + subj + '/'
		all_folds = os.listdir(newdir)
		all_folds = np.sort(all_folds)
		index2 = 0
		for fold in all_folds:
			fn = newdir + fold + '/' + info.fl

			data = sio.loadmat(fn)

			trl = np.reshape(data['labels'],[len(data['labels']),])
			tel = np.reshape(data['labels_te'],[len(data['labels_te']),])

			if info.ismesh == 0:
				tr = data['data']
				te = data['data_te']
				trnew = np.zeros([tr.shape[0], len(trl)])
				for i in range(0,len(tel)):
					trnew[:,i] = tr[:,i*6+2]

				tenew = np.zeros([te.shape[0], len(tel)])
				for i in range(0,len(tel)):
					tenew[:,i] = te[:,i*6+2]
			else:
				trnew = data['a_tr']
				tenew = data['a_te']

			trnew, tenew,f_scores, selected = weight_selection(trnew, tenew, trl,select)
			'''
			if select == 1 and trnew.shape[0] > trnew.shape[1] * 2:
				f_scores, p_val = feat_select.f_classif(trnew.T, trl)
				selected, ax = vs.find_important_features([f_scores], difference = 1, plotting = 0)
				indices = sorted(range(len(f_scores)),key=lambda x:f_scores[x])
				selected_ind = indices[int(selected):-1]
				trnew = [trnew[x] for x in selected_ind]
				tenew = [tenew[x] for x in selected_ind]
			'''

			trnew = np.array(trnew).T
			tenew = np.array(tenew).T

			

			

			model = linear_model.LogisticRegression(penalty = 'l2')
			model.fit(trnew, trl)
			k = model.predict(tenew)
			acc = sum(k==tel)*100/len(tel)
			accuracies[index1, index2] = acc
			index2 = index2 + 1
		index1 = index1 + 1 
	return accuracies

def mvpa_third_layer(info,select = 0):

	'''
	sample_call
	for c in [0,1]:
		for m in [0,1,2,3]:
			for select in [0,1]:
				info.cluster = c
				info.ismesh = m
				mvpa_accuracies = call_mvpa(info,select)
				fn = info.resultdir + 'cluster_' + str(c) + '_mesh_' + str(m) + '_select_' + str(select) + '_mvpa.mat'
				sio.savemat(fn, {'accuracies' : mvpa_accuracies})
	'''
	all_subjects = os.listdir(info.datapath)
	accuracies = np.zeros([len(all_subjects), 6])
	index1 = 0
	index2 = 0
	for subj in all_subjects:
		newdir = info.dir + subj + '/'
		all_folds = os.listdir(newdir)
		index2 = 0
		for fold in all_folds:
			if info.cluster == 1:
				fn = newdir + fold + '/' + info.tl 
			else:
				fn = newdir + fold + '/' + info.tlaal

			data = sio.loadmat(fn)

			trl = np.reshape(data['labels'],[len(data['labels']),])
			tel = np.reshape(data['labels_te'],[len(data['labels_te']),])
			#print info.ismesh, info.cluster
			if info.ismesh == 0:
				tr = data['data']
				te = data['data_te']
				trnew = np.zeros([tr.shape[0], len(trl)])
				for i in range(0,len(trl)):
					trnew[:,i] = tr[:,i*6+2]

				tenew = np.zeros([te.shape[0], len(tel)])
				for i in range(0,len(tel)):
					tenew[:,i] = te[:,i*6+2]
			elif info.ismesh > 1 and info.cluster == 0:
				p = (info.ismesh - 1) * 10
				fn1 = newdir + fold + '/p_'+ str(p) + '_' + info.tlaal
				data_mesh = sio.loadmat(fn1)
				trnew = data_mesh['a_tr']
				tenew = data_mesh['a_te']
			elif info.ismesh > 1 and info.cluster == 1:
				p = (info.ismesh - 1) * 10
				fn1 = newdir + fold + '/p_'+ str(p) + '_' + info.tl
				data_mesh = sio.loadmat(fn1)
				trnew = data_mesh['a_tr']
				tenew = data_mesh['a_te']
			else:
				#print info.ismesh, info.cluster
				trnew = data['a_tr']
				tenew = data['a_te']

			trnew, tenew,f_scores, selected = weight_selection(trnew, tenew, trl,select)
			'''
			if select == 1 and trnew.shape[0] > trnew.shape[1] * 2:
				f_scores, p_val = feat_select.f_classif(trnew.T, trl)
				selected, ax = vs.find_important_features([f_scores], difference = 1, plotting = 0)
				indices = sorted(range(len(f_scores)),key=lambda x:f_scores[x])
				selected_ind = indices[int(selected):-1]
				trnew = [trnew[x] for x in selected_ind]
				tenew = [tenew[x] for x in selected_ind]
			'''

			trnew = np.array(trnew).T
			tenew = np.array(tenew).T

			

			

			model = linear_model.LogisticRegression(penalty = 'l2')
			model.fit(trnew, trl)
			k = model.predict(tenew)
			acc = sum(k==tel)*100/len(tel)
			accuracies[index1, index2] = acc
			index2 = index2 + 1
		index1 = index1 + 1 
	return accuracies

def call_fsg(info, select = 0,levels_for_fsg = [1,2]):
	all_subjects = os.listdir(info.datapath)
	accuracies = np.zeros([len(all_subjects), 6])
	index1 = 0
	index2 = 0
	all_subjects =  np.sort(all_subjects)
	for subj in all_subjects:
		newdir = info.dir + subj + '/'
		all_folds = os.listdir(newdir)
		all_folds = np.sort(all_folds)
		index2 = 0
		for fold in all_folds:
			fn1 = info.get_first_layer + subj + '/' + fold + '/' + info.fl 
			if info.ismesh < 2:
				if info.cluster == 1:
					fn2 = newdir + fold + '/' + info.sl 
					fn3 = newdir + fold + '/' + info.tl
				else:
					fn2 = newdir + fold + '/' + info.slaal
					fn3 = newdir + fold + '/' + info.tlaal
			else:
				p = (info.ismesh - 1) * 10
				if info.cluster == 1:
					fn2 = newdir + fold + '/p_' + str(p) + '_' + info.sl 
					fn3 = newdir + fold + '/p_' + str(p) + '_' + info.tl
				else:
					fn2 = newdir + fold + '/p_' + str(p) + '_' + info.slaal
					fn3 = newdir + fold + '/p_' + str(p) + '_' + info.tlaal

			fn1 = newdir + fold + '/p_' + str(10) + '_' + info.slaal 
			fn2 = newdir + fold + '/second_layer_fft_temp_neig_.mat' 
			data1 = sio.loadmat(fn1)
			data2 = sio.loadmat(fn2)
			#data3 = sio.loadmat(fn3)

			trl = np.reshape(data1['labels'],[len(data1['labels']),])
			tel = np.reshape(data1['labels_te'],[len(data1['labels_te']),])

			if info.ismesh == 0:
				for level in levels_for_fsg:
					if level == 1 and info.base_layer_memberships == 0:
						tr = data1['data']
						te = data1['data_te']
						 
						trnew = np.zeros([tr.shape[0], len(trl)])
						for i in range(0,len(trl)):
							trnew[:,i] = tr[:,i*6+2]

						tenew = np.zeros([te.shape[0], len(tel)])
						for i in range(0,len(tel)):
							tenew[:,i] = te[:,i*6+2]

						trnew, tenew, f_scores, selected = weight_selection(trnew,tenew,trl, select)
						idx = np.ones([trnew.shape[0],1])
					elif level == 2:
						tr = data2['data']
						te = data2['data_te']
						 
						trnewtmp = np.zeros([tr.shape[0], len(trl)])
						for i in range(0,len(trl)):
							trnewtmp[:,i] = tr[:,i*6+2]
						
						tenewtmp = np.zeros([te.shape[0], len(tel)])
						for i in range(0,len(tel)):
							tenewtmp[:,i] = te[:,i*6+2]

						trnewtmp, tenewtmp, f_scores2, selected2 = weight_selection(trnewtmp,tenewtmp,trl, select)
						if info.base_layer_memberships == 0: 
							idx = np.concatenate((idx,level*np.ones([trnewtmp.shape[0],1])))
							trnew = np.concatenate((trnew, trnewtmp))
							tenew = np.concatenate((tenew, tenewtmp))
						else:
							idx = level*np.ones([trnewtmp.shape[0],1])
							trnew = trnewtmp
							tenew = tenewtmp

					elif level == 3:
						tr = data3['data']
						te = data3['data_te']
						 
						trnewtmp = np.zeros([tr.shape[0], len(trl)])
						for i in range(0,len(tel)):
							trnewtmp[:,i] = tr[:,i*6+2]
						
						tenewtmp = np.zeros([te.shape[0], len(tel)])
						for i in range(0,len(tel)):
							tenewtmp[:,i] = te[:,i*6+2]
						trnewtmp, tenewtmp, f_scores3, selected3 = weight_selection(trnewtmp,tenewtmp,trl, select)
						idx = np.concatenate((idx,level*np.ones([trnewtmp.shape[0],1])))
						trnew = np.concatenate((trnew, trnewtmp))
						tenew = np.concatenate((tenew, tenewtmp))

			else:
				for level in levels_for_fsg:
					if level == 1 and info.base_layer_memberships == 0:
						trnew = data1['a_tr']
						tenew = data1['a_te']
						trnew, tenew, f_scores, selected = weight_selection(trnew,tenew,trl, select)
						idx = np.ones([trnew.shape[0],1]) 
						
					elif level == 2:
						trnewtmp = data2['a_tr']
						tenewtmp = data2['a_te']
						trnewtmp, tenewtmp, f_scores2, selected2 = weight_selection(trnewtmp,tenewtmp,trl, select)
						if info.base_layer_memberships == 0:
							idx = np.concatenate((idx,level*np.ones([trnewtmp.shape[0],1]))) 
							trnew = np.concatenate((trnew, trnewtmp))
							tenew = np.concatenate((tenew, tenewtmp))
						else:
							idx = level*np.ones([trnewtmp.shape[0],1])
							trnew = trnewtmp
							tenew = tenewtmp
					elif level == 3:
						trnewtmp = data3['a_tr']
						tenewtmp = data3['a_te']
						trnewtmp, tenewtmp, f_scores3, selected3 = weight_selection(trnewtmp,tenewtmp,trl, select)
						idx = np.concatenate((idx,level*np.ones([trnewtmp.shape[0],1]))) 
						trnew = np.concatenate((trnew, trnewtmp))
						tenew = np.concatenate((tenew, tenewtmp))


			trnew = trnew
			tenew = tenew

			idx = idx.astype(int) #int(idx.reshape([len(idx),]))
			idx = idx.T
			max_base, base_layer_performances,performance,membership_tr,membership_te = fsg(info, trnew, trl, tenew, tel, idx, 1,select, subj  = subj, fold = fold)
			fn = info.resultdir + subj + '_' + fold + '_' + 'cluster_' + str(info.cluster) + '_mesh_' + str(info.ismesh) +'_select_' + str(select) +'_memberships.mat'
			
			sio.savemat(fn, {'max_base' : max_base, 'base_performances': base_layer_performances, 'membership_tr': membership_tr, 'membership_te': membership_te})
			accuracies[index1, index2] = performance
			#print fn, ' - ', performance
			index2 = index2 + 1
		index1 = index1 + 1 
	return accuracies



def call_fsg_with_nn_svm(info, select = 0):
	all_subjects = os.listdir(info.datapath)
	all_subjects = np.sort(all_subjects)
	accuracies_nn = np.zeros([len(all_subjects), 6])
	accuracies_svm = np.zeros([len(all_subjects), 6])
	index1 = 0
	index2 = 0
	for subj in all_subjects:
		newdir = info.dir + subj + '/'
		all_folds = os.listdir(newdir)
		all_folds = np.sort(all_folds)
		index2 = 0
		for fold in all_folds:
			if info.cluster == 0:
				fn1 = info.get_second_layer + subj + '/' + fold + '/' + info.slaal
			else:
				fn1 = info.get_second_layer + subj + '/' + fold + '/' + info.sl

			#print fn1

			data1 = sio.loadmat(fn1)

			trl = np.reshape(data1['labels'],[len(data1['labels']),])
			tel = np.reshape(data1['labels_te'],[len(data1['labels_te']),])
			
		    
		 	trl = trl.reshape(1,len(trl)).T
			tel = tel.reshape(1,len(tel)).T


			ll = np.unique(trl)
			lll = ll.shape
			max_class_num = lll[0]
			


			fn = info.resultdir + subj + '_' + fold + '_' + 'cluster_' + str(info.cluster) + '_mesh_' + str(info.ismesh) +'_select_' + str(select) +'_memberships.mat'
			data = sio.loadmat(fn)			
			
			mem_tr = data['membership_tr']
			mem_te = data['membership_te']
			model_meta_nn = MLPClassifier(hidden_layer_sizes = (8,), max_iter = 1000 , learning_rate_init = 0.01,solver = 'lbfgs')# solver = 'sgd', max_iter = 10000,learning_rate='adaptive',)#,activation = 'logistic',learning_rate='adaptive', max_iter = 5000)
			#model_meta = linear_model.LogisticRegression(penalty = 'l2',C = 1)
    		

			asd = trl.reshape([len(trl),])
			model_meta_nn.fit(mem_tr,asd)
			k = model_meta_nn.predict(mem_te)
			numhits = 0
			for i in range(len(k)):
				if k[i] == tel[i]:
					numhits+=1
		            
			performance = float(numhits) * 100 /len(tel)     
			accuracies_nn[index1, index2] = performance

			model_meta_svm = svm.SVC()
			model_meta_svm.fit(mem_tr,asd)
			k = model_meta_svm.predict(mem_te)
			numhits = 0
			for i in range(len(k)):
				if k[i] == tel[i]:
					numhits+=1
		            
			performance = float(numhits) * 100 /len(tel)     
			accuracies_svm[index1, index2] = performance
			
			#print fn, ' - ', performance
			index2 = index2 + 1
		index1 = index1 + 1 
	#print np.mean(accuracies)
	return accuracies_nn, accuracies_svm

def weight_selection(tr,te,trl, select):
	if select == 1 and tr.shape[0] > tr.shape[1] * 2:
		f_scores, p_val = feat_select.f_classif(tr.T, trl)
		selected, ax = vs.find_important_features([f_scores], difference = 1, plotting = 0)
		indices = sorted(range(len(f_scores)),key=lambda x:f_scores[x])
		selected_ind = indices[int(selected):-1]
		if len(selected_ind) < 2:
			selected, ax = vs.find_important_features([f_scores], difference = 0.1, plotting = 0)
			indices = sorted(range(len(f_scores)),key=lambda x:f_scores[x])
			selected_ind = indices[int(selected):-1]
		trnew = np.array([tr[x] for x in selected_ind])
		tenew = np.array([te[x] for x in selected_ind])
		#print tr.shape, trnew.shape
		return trnew, tenew, f_scores, selected
	else:
		return tr, te, [], []


CL_Acuracies = call_fsg(info,select)
