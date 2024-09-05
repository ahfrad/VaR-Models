function [LR,pvalue,MAPE,exc]= kupiec(Ret,VaR_short,VaR_long,cl,n1)    
%Kupiec test or Unconditional Coverage test

%cl: confidence level/ [0.99 0.975 0.95 0.90]
%n1: estimation window
%n2: out-of-sample size (n2=1110 days)
%Ret: log-returns (entire dataset)/ It should be a row vector



%outputs:
         %pvalue 
                 
         %pvalue=[short position (first row)     
         %        long position]  (second row)
         %%% pvalues are in percentage
         
         %MAPE=[ShortPosition LongPosition Total]
         %MAPE: Mean Absolute Percentage Error
         
         %exc: number of exceptions [short position (first row)     
         %                           long position]  (second row)

r=Ret(n1+1:length(Ret))';    
N=length(r);
k=length(cl);
pvalue=zeros(2,k);
LR=zeros(2,k);
p1=zeros(1,k);
p2=zeros(1,k);
exc=zeros(2,k);
alpha=1-cl;

for j=1:k
    
    x1= sum(r>VaR_short(:,j));  %number of exceptions for short position
    x2= sum(r< VaR_long(:,j));   %number of exceptions for long position
    x=[x1 x2];
    p=[x1/N x2/N];  %empirical shortfall probability(exception ratio)
    
   % a=alpha(j).^x.*(1-alpha(j)).^(N-x);      
   % b= p.^x.*(1-p).^(N-x);
   % LR=-2*log(a./b);
    
   
    LR(:,j)=-2*(x.*log(alpha(j)./p)+(N-x).*log((1-alpha(j))./(1-p)));

    pv=1-chi2cdf(LR(:,j),1);
    pvalue(:,j)=100*pv'; 
    
    p1(1,j)=p(1); %saves exception ratio of short position to calculate MAPE
    p2(1,j)=p(2);  %saves exception ratio of short position to calculate MAPE
    exc(1,j)=x1;  %saves number of exceptions for short position
    exc(2,j)=x2;   %saves number of exceptions for long position
end

Mape1= 1/k*sum(abs(p1-alpha)./alpha); % short position
Mape2= 1/k*sum(abs(p2-alpha)./alpha);   % long position
Mape=.5*(Mape1+Mape2);

MAPE=[Mape1 Mape2 Mape];  %[short-positon  long-position   toatal] 

end
