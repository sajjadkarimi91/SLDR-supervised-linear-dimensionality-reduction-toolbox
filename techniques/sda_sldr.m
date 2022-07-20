function [para, Z] = sda_sldr(X, labels, dim, lamda_in)

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


classes_labels = unique(labels);
num_classes = length(classes_labels);

global dummy_labels X_org count_itr lamda

count_itr=0;

dummy_labels = zeros(n,num_classes);
for k = 1:num_classes
    dummy_labels(labels ==classes_labels(k),k) = 1;
end

if(nargin==2)
    dim= min(num_classes,max(1,size(X,2)-1));
end

if nargin<4
    lamda = 0.01;
else
    lamda = lamda_in;
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

function [f, g] = sda_cost(w_vec)

global dummy_labels X_org count_itr lamda
count_itr = count_itr+1;

% w_enb = 0.000001;
% w_dr = 0.00001;
sn0 = 0.001;
h_sl0 = sn0;
h_fee = 0.00005;

end

