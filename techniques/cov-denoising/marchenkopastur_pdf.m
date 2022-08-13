function pdf_vals = marchenkopastur_pdf(lambda,s,c)

lambda_p = s^2 * (1+sqrt(c))^2;
lambda_m = s^2 * (1-sqrt(c))^2;

if lambda_m>0 && lambda_p < lambda_m*(min(length(lambda),100))
    
    pdf_vals = (1./(2*pi*lambda*c*s^(2))).*sqrt((lambda_p-lambda).*(lambda-lambda_m));
    pdf_vals(pdf_vals<0) =0;
    pdf_vals(lambda>lambda_p) = 0;
    pdf_vals(lambda<lambda_m) = 0;
    
else
    pdf_vals = 0*lambda;
end