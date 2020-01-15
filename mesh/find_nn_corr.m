function [neighbor_index] = find_nn_corr(corrs, k_values, voxel_ind)
% data=data';
% test=test';




%% BEGIN kNN


    
    [val, ind] = sort(corrs(voxel_ind,:), 'descend');


    for k=1:length(k_values)
	neighbor_index{1,k} = ind(2:k_values+1);
	
	
    end
end
