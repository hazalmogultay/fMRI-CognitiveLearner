%%[a_tr, a_te, tr_error, te_error]=temporal_ridge_fc_tr_te(corr(data'),p,data',data_te',10,6);

%%
function [a_tr, a_te, tr_error, te_error]=temporal_fc(corrs,k_values,tr_all,te_all,lambda, duration)

% pos: voxel_number by 3 matrix 
% k_values: the number of neighbor voxels ( p_values in the algorithm )
% tr_all, te_all: train and test data (N by voxel_number matrix, N:sample size
%%

p=k_values;

a_tr=[];
a_te=[];
te_error =[];
tr_error =[];

for j = 1:(size(tr_all,1)/duration)
    x=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration,:);
    corrs = corr(x);
    a_tr =[a_tr; reshape(corrs,[1,numel(corrs)])];
end

for j = 1:(size(te_all,1)/duration)
    x=te_all(duration*(j-1) + 1 : duration*(j-1) +duration,:);
    corrs = corr(x);
    a_te =[a_te; reshape(corrs,[1,numel(corrs)])];
end


end
   