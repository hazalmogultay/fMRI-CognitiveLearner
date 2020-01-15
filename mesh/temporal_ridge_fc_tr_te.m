

%%
function [a_tr, a_te, tr_error, te_error]=temporal_ridge_fc_tr_te(corrs,k_values,tr_all,te_all,lambda,duration)

% pos: voxel_number by 3 matrix 
% k_values: the number of neighbor voxels ( p_values in the algorithm )
% tr_all, te_all: train and test data (N by voxel_number matrix, N:sample size
%%

p=k_values;

a_tr=[];
a_te=[];
te_error =[];
tr_error =[];

for i=1:size(tr_all,2) % if size(tr_all)~=size(te_all), change here

    %temp=1:size(pos,1);


    [neighbor_index] = find_nn_corr(corrs, k_values, i);

  %  d=setdiff(temp,neighbor_index{1,1});


    a1=[];
    error = [];
    for j = 1:(size(tr_all,1)/duration)
        y=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration,i);
        X=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration , neighbor_index{1,1});
        %lambda = 0.5;
        M = eye(size(X,2));
        M(1,1) = 1;
        theta = pinv(X'*X+lambda.*M)*X'*y;
        err = sum((X*theta - y).*(X*theta - y));
        a1 = [a1;theta'];
        error = [error err];
    end
    tr_error = [tr_error;error];

    
    a_tr=[a_tr a1];

   % d=setdiff(temp,neighbor_index{1,1});

   a1=[];
   error = [];
    for j = 1:(size(te_all,1)/duration)
        y=te_all(duration*(j-1) + 1 : duration*(j-1) +duration,i);
        X=te_all(duration*(j-1) + 1 : duration*(j-1) +duration , neighbor_index{1,1});
        %lambda = 0.5;
        M = eye(size(X,2));
        M(1,1) = 1;
        theta = pinv(X'*X+lambda.*M)*X'*y;
        err = sum((X*theta - y).*(X*theta - y));
        a1 = [a1;theta'];
        error = [error err];
    end
    te_error = [te_error;error];
    %tr_error = [tr_error g];

    
    a_te=[a_te a1];
    %te_error = [te_error g];


    clear x;
end
   