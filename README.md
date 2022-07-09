# SLDR
A MATLAB toolbox for supervised linear dimension reduction (SLDR) including LDA, HLDA, MMDA and WHMMDA

Codes for the following papers:

1. Heteroscedastic Max–Min distance analysis for dimensionality reduction
2. Heteroscedastic max-min distance analysis 
3. Max-min distance analysis by using sequential SDP relaxation for dimension reduction 
4. Linear dimensionality reduction via a heteroscedastic extension of LDA: the Chernoff criterion
5. Supervised Linear Dimension-Reduction Methods: Review, Extensions, and Comparisons

## 1. Introduction.

This package includes the prototype MATLAB codes for supervised linear dimension reduction (SLDR).

The implemented methodes include: 

  1. Linear discriminant analysis (LDA)
  2. Heteroscedastic extension of LDA (HLDA)       
  3. Max-min distance analysis (MMDA) 
  4. Heteroscedastic extension of MMDA (WHMMDA) 
  5. Extended partial least squares (EPLS)  
     


## 2. Usage & Dependency.

## Dependency:
     CVX MATLAB toolbox form http://web.cvxr.com/cvx/cvx-w64.zip

## Usage:
Run and check "demo_run_methods.m" and you'll see the below results

Note: EPLS is a regression-based method, so its classification performance is poor

![results](/demo.jpg)
