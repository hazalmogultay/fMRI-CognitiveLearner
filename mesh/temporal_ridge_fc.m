

%%
function [a_tr]=temporal_ridge_fc(corrs,k_values,tr_all,lambda,duration)

% pos: voxel_number by 3 matrix 
% k_values: the number of neighbor voxels ( p_values in the algorithm )
% tr_all, te_all: train and test data (N by voxel_number matrix, N:sample size
%%

a_tr=[];
%tr_error =[];
size(tr_all,2)
for i=1:size(tr_all,2) % if size(tr_all)~=size(te_all), change here
    
   [neighbor_index] = find_nn_corr(corrs, k_values, i);

    a1=[];
    %error = [];
    for j = 1:(size(tr_all,1)/duration)
        y=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration,i);
        X=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration , neighbor_index{1,1});
%         M = eye(size(X,2));
%         M(1,1) = 1;
%         theta = pinv(X'*X+lambda.*M)*X'*y;
%         
        theta = ridge(y,X,lambda);
        %err = sum((X*theta - y).*(X*theta - y));
        tempp = zeros(1,size(tr_all,2));
        tempp(neighbor_index{1,1}) = theta';
        a1 = [a1;tempp];
        %error = [error err];
    end
    %tr_error = [tr_error;error];

    a_tr=[a_tr a1];
    
end


   