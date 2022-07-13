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

for k1 = 1:min(num_classes,dim)
    strt = 0;
    y{k1} = zeros(n*num_classes,1);
    Xk{k1} = zeros(n*num_classes,d);

    k2_vals = 1:num_classes;
    k2_vals(k1)=[];
    for k2 = k2_vals
        ind_pos = find(labels==classes_labels(k1));
        ind_neg = find(labels==classes_labels(k2));
        this_dur = length(ind_pos) + length(ind_neg);

        Xk{k1}(strt+1:strt+this_dur,:) = X([ind_pos;ind_neg],:);
        y{k1}(strt+1:strt+this_dur) = [ones(length(ind_pos),1);-ones(length(ind_neg),1)];
        strt = strt + this_dur;
    end
    y{k1} = y{k1}(1:strt);
    Xk{k1} = Xk{k1}(1:strt,:);
    y{k1} = y{k1}- mean(y{k1});
end


W = zeros(d,dim);
k=0;
while k <= dim

    for k1 = 1:min(num_classes,dim)
        k=k+1;
        if k>dim
            break
        end

        Xkk =Xk{k1};
        yk = y{k1};

        % SB = Xk'*(y*y'+lambda*eye(n*num_classes))*Xk; % this form is memory unefficeint
        SB = (Xkk'*yk)*(yk'*Xkk) + lambda*(Xkk')*Xkk;

        % Perform eigendecomposition of Sb
        [M1, eigen_vals] = eig(SB);
        [~, ind] = sort(diag(eigen_vals), 'descend');
        uk = M1(:,ind(1));
        zk = Xkk*uk;
        Xk{k1} = Xkk - zk*(uk');
        y{k1} = yk- ((yk'*zk)/(zk'*zk)) *zk;
        W(:,k) = uk;

    end
end

% Z has the dimentional reduced data sample X.
Z = X*W;

para.W = W;
para.mb = mb;
para.model = 'epls';

end