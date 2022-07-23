function [para, Z] = sda_sldr(X, labels, dim, lamda_in, eps_val_in)

% Stochastic discriminant analysis (SDA) matches similarities between points in the projection space with those in a response space
% Juuti, Mika, Francesco Corona, and Juha Karhunen.
% Stochastic discriminant analysis for linear supervised dimension reduction.
% Neurocomputing 291 (2018): 136-150.

%[para,W] = sda_sldr(X, labels, dim) , where dim values by default is Number of classes: C
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    labels: n --- dimensional vector of class labels
%    dim:    ----- dimensionality of reduced space (default:C)
%            dim has to be from 1<=dim<=d
% Output:
%    para:   output structure of sda model for input of test_sldr.m function
%    Z:      n x dim matrix of dimensionality reduced features

global dummy_labels X_org count_itr lamda

classes_labels = unique(labels);
num_classes = length(classes_labels);

if(nargin==2)
    dim= min(num_classes,max(1,size(X,2)-1));
end

if nargin<4
    lamda = 0.01;
else
    lamda = lamda_in;
end

if nargin<5
    eps_val = 10^-5;
else
    eps_val = eps_val_in;
end



count_itr=0;

dummy_labels = eps_val + zeros(n,num_classes);
for k = 1:num_classes
    dummy_labels(labels ==classes_labels(k),k) = 1;
end



% recentering original feature
mb = mean(X,'omitnan');
X = X - mb;
X_org = X;

[coeff, score, ~, ~, ~, mu] = pca(X,'NumComponents',dim);
% Z = (yhat-mu)*coeff;


W = coeff;
x0 = W(:); % vectorize matrix

options = optimoptions('fminunc','MaxFunctionEvaluations',5*10^2,'Algorithm','trust-region','SpecifyObjectiveGradient',true);
% options = optimoptions('fmincon','MaxFunctionEvaluations',5*10^4,'SpecifyObjectiveGradient',true,'StepTolerance',10^-8,'ConstraintTolerance',10^-8);

W = fminunc(@sda_cost,x0,options);


% selecting dim eigenvectors associated with the dim largest eigenvalues
[V,D] = eig(S_chernoff);
D = real(diag(D));
V = real(V);
[~,sort_index]=sort(D,'descend');
W =  V(:,sort_index(1:dim));


% Z has the dimentional reduced data sample X.
Z = X*W;

para.W = W;
para.mb = mb;
para.model = 'sda';

end

%% cost function and its total gradient
% For large-scale data sets, we try to write memory-efficient code instead of fast code


function [f, g] = sda_cost(w_vec)

global dummy_labels X_org count_itr lamda

count_itr = count_itr+1;

f = 0;
f = f + lamda*sum(w_vec.^2);

[K,d] = size(X_org);
W = reshape(w_vec,d,[]);
Z = X_org*W;

P_sigma = 0;
Q_sigma = 0;
for i = 1:K

    D2i = sum((Z(i,:)-Z).^2,2);
    Qbi = 1./(1+D2i);

    [~,ind_class] = max(dummy_labels(i,:));
    Pi = dummy_labels(:,ind_class);
    Q_sigma = Q_sigma + sum(Qbi);
    P_sigma = P_sigma + sum(Pi);
end

for i = 1:K

    D2i = sum((Z(i,:)-Z).^2,2);
    Qi = (1/Q_sigma)./(1+D2i);
    Qbi = Q_sigma*Qi;

    [~,ind_class] = max(dummy_labels(i,:));
    Pi = dummy_labels(:,ind_class)/P_sigma;

    Ti = X_org(i,:) - X_org;


end


end

