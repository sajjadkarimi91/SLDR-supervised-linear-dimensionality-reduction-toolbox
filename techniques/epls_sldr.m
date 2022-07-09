function [para, Z] = epls_sldr(X, labels, dim, lambda)

% Partial least squares (PLS) has a long history & PLS constructs the basis of a linear subspace iteratively
% This function considers the extended version of the PLS algorithm by combining the supervised objective and
% the unsupervised objective in PCA

% Xu, Shaojie, Joel Vaughan, Jie Chen, Agus Sudjianto, and Vijayan Nair.
% "Supervised Linear Dimension-Reduction Methods: Review, Extensions, and Comparisons."
% arXiv preprint arXiv:2109.04244 (2021).

%[para,W] = epls_sldr(X, labels, dim) , where dim values by default is Number of classes: C
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C)
%            dim has to be from 1<=dim<=d
%    lambda:  0 is equivalent to PLS and +inf is PCA
% Output:
%    para:   output structure of lda model for input of test_sldr.m function
%    Z:      n x dim matrix of dimensionality reduced features


classes_labels = unique(labels);
num_classes = length(classes_labels);

if(nargin==2)
    dim= min(num_classes,max(1,size(X,2)-1));
end

if nargin <4
    lambda = 1;
end


% recentering original feature
mb = mean(X,'omitnan');
X = X - mb;


[n ,d]= size(X);
y = zeros(n*num_classes,1);
Xk = zeros(n*num_classes,d);

strt = 0;
for k1 = 1:num_classes
    for k2=k1+1:num_classes
        ind_pos = find(labels==classes_labels(k1));
        ind_neg = find(labels==classes_labels(k2));
        this_dur = length(ind_pos) + length(ind_neg);

        Xk(strt+1:strt+this_dur,:) = X([ind_pos;ind_neg],:);
        y(strt+1:strt+this_dur) = [ones(length(ind_pos),1);-ones(length(ind_neg),1)];
        strt = strt + this_dur;
    end
end

y = y(1:strt);
Xk = Xk(1:strt,:);

% Xk = X;
% y = labels- mean(labels);

W = zeros(d,dim);
for k= 1:dim

    % SB = Xk'*(y*y'+lambda*eye(n*num_classes))*Xk; % this form is memory unefficeint
    SB = (Xk'*y)*(y'*Xk) + lambda*(Xk')*Xk;

    % Perform eigendecomposition of Sb
    [M1, eigen_vals] = eig(SB);
    [~, ind] = sort(diag(eigen_vals), 'descend');
    uk = M1(:,ind(1));
    zk = Xk*uk;
    Xk = Xk - zk*(uk');
    y = y- ((y'*zk)/(zk'*zk)) *zk;
    W(:,k) = uk;

end

% Z has the dimentional reduced data sample X.
Z = X*W;

para.W = W;
para.mb = mb;
para.model = 'epls';

end