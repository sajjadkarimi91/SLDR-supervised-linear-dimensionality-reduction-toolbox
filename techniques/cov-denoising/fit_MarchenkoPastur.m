function lambda_max = fit_MarchenkoPastur(Data,lambda)


s = std(Data(:),'omitnan');
r = size(Data,2)/size(Data,1);

% Estimating Probability Density Function
% making sigma and ratio variable for best fitting
counter = 0;

for s_var = 1:50
    for r_var = 1:30
        counter = counter + 1;
        
        % choosing in [sigma/5 2*sigma] and [ratio/3 20*ratio]
        s_mp = (5+s_var)/30 * s;
        r_mp = r_var *r/10;
        
        %Estimating MP likelihood
        pdf_vals = marchenkopastur_pdf(lambda,s_mp,r_mp);
        pdf_vals(pdf_vals==0) =[];
        
        % saving parameters for choosing next
        ll_vals(counter) = sum(log(pdf_vals));
        L(counter) = length(pdf_vals);
        params(counter,1) = s_var;
        params(counter,2) = r_var;
    end
end

% choosing parameters with max log-likelihood sums
[~,ind_max] = max(ll_vals.*(L>1/3*length(lambda)));
s_var = params(ind_max,1) ;
r_var = params(ind_max,2) ;

% retrieving parameters 
s_mp = (5+s_var)/30 * s;
r_mp = r_var *r/10;

% adjusting plot intervals
n = 100;
lambda_p = s_mp^(2) * (1+10*sqrt(r_mp))^2;  % max lambda
lambda_m = s_mp^(2) * (1-2*sqrt(r_mp))^2;   % min lambda
space_lambda = linspace(lambda_m,lambda_p,n);
pdf_vals = marchenkopastur_pdf(space_lambda,s_mp,r_mp);
pdf_vals = pdf_vals/sum(pdf_vals);   % normalizng


% final normalizing
[f,lambda_h] = hist(lambda,space_lambda);
lambda_max = s_mp^2 * (1+sqrt(r_mp))^2;    % upper range for normalizing
lambda_min = s_mp^2 * (1-sqrt(r_mp))^2;    % lower range for normalizing
lamda_range = lambda;
lamda_range(lamda_range>lambda_max)=[];
lamda_range(lamda_range<lambda_min)=[];
f=f/sum(f) * length(lambda)/length(lamda_range);



