function  Z = test_sldr(X, para)

% Z = test_sldr(X, para) 
% Input:
%    X:      n x d matrix of original feature samples
%            d --- dimensionality of original features
%            n --- the number of samples
%    para:   structure of the model 
% Output:
%    Z:      n x dim matrix of dimensionality reduced features


W = para.W;
mb = para.mb;

% recentering original feature
X = X - mb;
Z = X*W;



% if strcmp(para.model , 'lda') || strcmp(para.model , 'hlda')
% 
%     W = para.W;
%     mb = para.mb;
%     X = X - mb;
%     Z = X*W;
% 
% elseif strcmp(para.model , 'mmda')
% 
%     W = para.W;
%     mb = para.mb;
%     Sw_sqrtinv = para.Sw_sqrtinv;
% 
%     X = X - mb;
%     X = X * Sw_sqrtinv ;
%     Z = X*W;
% 
% end