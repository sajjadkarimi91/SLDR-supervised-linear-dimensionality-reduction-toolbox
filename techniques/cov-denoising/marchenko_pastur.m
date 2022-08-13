function sigma_trg_den = marchenko_pastur(data_in)

AssetCovar = data_in'*data_in/size(data_in,1);
sigma_trg = (AssetCovar+AssetCovar')/2;

[U,S,V] = svd(sigma_trg);
lambda = diag(S);

% fitting Marchenko-Pastur distribution and find lambda-max for denoising
lambda_max = fit_MarchenkoPastur(data_in,lambda);
% lambda_max = 1.5*lambda_max;

% denoising using mean of noisy lambdas
noise_ind = find(lambda<=lambda_max);
lambda_j = (1/length(noise_ind))*sum(lambda(noise_ind));
lambda_den = lambda;
lambda_den(noise_ind) = lambda_j;

% sigma_den = W*diag(lambda_den)*W';
sigma_trg_den = U*diag(lambda_den)*V';