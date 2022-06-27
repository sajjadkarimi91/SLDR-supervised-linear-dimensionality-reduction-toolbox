function  Z = test_lda(X, para)

%       [Z,W]=FDA(X,Y) , where r values by default is Number of classes
%       minus 1: C-1
% Input:
%    X:      d x n matrix of original samples
%            d --- dimensionality of original samples
%            n --- the number of samples
%    Y:      n --- dimensional vector of class labels
%    r:      ----- dimensionality of reduced space (default:C-1)
%            r has to be from 1<=r<=C-1, where C is # of lables "classes"
% Output:
%    W:      d x r transformation matrix (Z=T'*X)
%    Z:      r x n matrix of dimensionality reduced samples

if strcmp(para.model , 'flda') || strcmp(para.model , 'hflda')

    W = para.W;
    mb = para.mb;
    X = X - mb;
    Z = X*W;

elseif strcmp(para.model , 'mmda')

    W = para.W;
    mb = para.mb;
    Sw_sqrtinv = para.Sw_sqrtinv;

    X = X - mb;
    X = X * Sw_sqrtinv ;
    Z = X*W;

end