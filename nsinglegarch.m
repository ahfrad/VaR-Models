
function [LogLF,coeffs,sigmaforcast]=nsinglegarch(x0,Ret,n1,n2,d) 
% This function estimates parameters, conditional volatility and LLF for Normal Single-Regime GARCH model

%x0: initial value/size(x0)=(1,5)
%cl: confidence level/ [0.99 0.975 0.95 0.90]
%coeffs: estimated coefficients
%d: forecasting window (d=5)
%LogLF: loglikelihood function
%n1: estimation window
%n2: out-of-sample size (n2=1110 days)
%Ret: log-returns (entire dataset)/ It should be a row vector
%sigmaforcast: estimated conditional volatility 

sigmapred=zeros(1,n2);  %size of the out-sample
coeffs=zeros(n2/d,5);
LogLF=zeros(n2/d,1);
j = n2/d - 1;
for i=0:j   
    r=Ret(1+d*i:n1+d*i);    %%r: pre-sample % update the parameters based on forecasting window
    
    %x0=size(1,5) x0=[mean(Ret) .2 .3 .25 .4];
    [coefficients, LLF] = nsingleasymfit (x0,r) %estimate the parameters via MLE
    coeffs(i+1,:)=coefficients;
    LogLF(i+1)=LLF;
    
    innovation=Ret(1+d*i:n1+d+d*i)-coefficients(1);   %innovation=epsilon
    innovation=innovation';
    
    a00 = coefficients(2);  %a0=wj  j=1,2 number of regimes
    a11 = coefficients(3);     %alpha j
    betaa = coefficients(4);   %beta j
    asymm = coefficients(5);
    
    T=length(r);
    sigma = zeros(1,T+d);
    
    if i==0
        sigma(:,1) = std(r);
        for t = 2:T+d
            sigma(:,t) = a00+a11*(abs(innovation(t-1,1))-asymm*innovation(t-1,1))+betaa*sigma(:,t-1);
        end
    else 
        sigma(:,1) =a00+a11*(abs(innovation0)-asymm*innovation0)+betaa*sigma0;
        for t = 2:T+d
         sigma(:,t) = a00+a11*(abs(innovation(t-1,1))-asymm*innovation(t-1,1))+betaa*sigma(:,t-1); %GARCH equation
    
        end
    end
    sigmapred(:,1+d*i:d+d*i)=sigma(:,T+1:T+d);
    innovation0=innovation(d,1);
    sigma0=sigma(:,d);
end
sigmaforcast=sigmapred';
end

function [coefficients, LLF] = nsingleasymfit (x0,r) 

nParameters = length (x0);
Optimization  =  optimset('fmincon');

if isinf(optimget(Optimization, 'MaxSQPIter'))
   Optimization  =  optimset(Optimization, 'MaxSQPIter', 1000 *nParameters);
end

global GARCH_TOLERANCE
GARCH_TOLERANCE  =  2*optimget(Optimization , 'TolCon', 1e-7);

Optimization  =  optimset(Optimization , 'MaxFunEvals' , 200*nParameters , ...
                                         'MaxIter'     , 1000             , ...
                                         'TolFun'      , 1e-6            , ...
                                         'TolCon'      , 1e-7            , ...
                                         'TolX'        , 1e-6            ) ;

   Optimization  =  optimset(Optimization , 'Display'     , 'iter');
   Optimization  =  optimset(Optimization , 'Diagnostics' , 'on');

Optimization  =  optimset(Optimization , 'LargeScale'  , 'off');

%
% Initialize a GLOBAL variable for convenience and code clarity.
global GARCH_TOLERANCE
GARCH_TOLERANCE  =  2*optimget(Optimization , 'TolCon', 1e-7);

objectiveFunction  =  @nsingleasym;

A  =  [0 0 1 1 0];        %a1+beta<1%  length(A)= length(x0) 
b  =  1  -  GARCH_TOLERANCE;  %x0=[mu a0 a1 beta asym]
%A=[];
%b=[];
LB =[-5; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE;-1]; 
UB =[5;1;1;1;1];

[coefficients, LLF   , ...
 exitFlag    , output, lambda] =  fmincon(objectiveFunction     , x0          , ...
                                          A                     , b           , ...
                                          []                    , []          , ...
                                          LB                    , UB          , ...
                                          []                    ,Optimization , ...
                                          r  );
end 

function [loglik,est] = nsingleasym(theta,r)

%here we have one regime then there is only one sigma.(h has one column)
%we've no lam becaus of there is only one regime.
mu = theta(1);
eps = r-mu;
eps=eps';
a0 = theta(2);
a1 = theta(3);
beta = theta(4);
asym = theta(5);


T = length(r);
h = zeros(T,1);

h(1) = std(r);

for t = 2:T
    h(t,1)  = a0 + a1*(abs(eps(t-1))-asym*eps(t-1)) + beta*h(t-1,1);
end


eta = eps(2:end)./h(2:end);
f = log(density(eta)./h(2:end));
loglik = -sum(f);
%%%%%
loglik(~isfinite(loglik))  =  1.0e+20;

est = [mu,a0,a1,beta,asym];
end


function f = density(z)

f = normpdf(z);
end




