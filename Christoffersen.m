function [LRcc,pvaluecc]= Christoffersen (LR,LR_indep)   
%Christoffersen test or conditional Coverage Test
%Before running this function,  run kupiec.m and independence.m functions 

%LR: statistic of unconditional coverage test (kupiec)
%LR_indep: statistic of independence test 
%pvauecc: p-value of conditional coverage test in percentage
%LRcc: statistic of conditional coverage test

LRcc=LR+LR_indep;
pv=1-chi2cdf(LRcc,2); 
pvaluecc=100*pv;
end

