
function [LogLF,coeffs,sigmaforcast]=snmsgarch(x0,Ret,n1,n2,d)
% This function estimates parameters, conditional volatility and LLF for Skew-Normal Markov-Switchin GARCH model

%x0: initial value/size(x0)=(1,11)
%cl: confidence level/ [0.99 0.975 0.95 0.90]
%coeffs: estimated coefficients
%d: forecasting window (d=5)
%LogLF: loglikelihood function
%n1: estimation window
%n2: out-of-sample size (n2=1110 days)
%Ret: log-returns (entire dataset)/ It should be a row vector
%sigmaforcast: estimated conditional volatility 

                                                                       
sigmapred=zeros(2,n2); 
coeffs=zeros(n2/d,11);
LogLF=zeros(n2/d,1);
j=n2/d-1;
for i=0:j   
    r=Ret(1+d*i:n1+d*i);    %r: pre-sample % update the parameters based on forecasting window
    
    %size(x0)=(1,11)  
    [coefficients, LLF] = snswitchasymfit (x0,r); %estimate the parameters via MLE
    coeffs(i+1,:)=coefficients;
    LogLF(i+1)=LLF;
    
    innovation=Ret(1+d*i:n1+d+d*i)-coefficients(1);   %innovation=epsilon
    innovation=innovation';
    
    a00 = coefficients(2:3)';  %a0=wj  j=1,2 number of regimes
    a11 = [coefficients(4);coefficients(5)];     %alpha j
    betaa = diag([coefficients(6),coefficients(7)]);   %beta j
    pp = coefficients(8);  %p11
    qq = coefficients(9);  %p22
    
    gamm1 = coefficients(10);   %gama1:skewness 
    deltaa = sign(gamm1).*sqrt(pi*(gamm1.^2).^(1/3))./sqrt(2*(gamm1.^2).^(1/3)+2^(1/3)*(4-pi)^(2/3));
    alphaa = sign(deltaa).*sqrt(deltaa.^2./(1-deltaa.^2)); 
    asymm = coefficients(11);
    
    T=length(r);
    sigma = zeros(2,T+d);
    
    if i==0
        sigma(:,1) = repmat(std(innovation(1:n1)),2,1)./sqrt(1-2/pi*deltaa.^2);
        for t = 2:T+d
            sigma(:,t) = a00+a11*(abs(innovation(t-1,1))-asymm*innovation(t-1,1))+betaa*sigma(:,t-1);
        end
    else 
        sigma(:,1) =a00+a11*(abs(innovation0)-asymm*innovation0)+betaa*sigma0;
        for t = 2:T+d
         sigma(:,t) = a00+a11*(abs(innovation(t-1,1))-asymm*innovation(t-1,1))+betaa*sigma(:,t-1); 
    
        end
    end
    sigmapred(:,1+d*i:d+d*i)=sigma(:,T+1:T+d);
    innovation0=innovation(d,1);
    sigma0=sigma(:,d);
end
sigmaforcast=sigmapred';
end

function [coefficients, LLF] = snswitchasymfit (x0,r) 

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


objectiveFunction  =  @snswitchasym;

A  =  [0 0 0 1 0 1 0 0 0 0 0;0 0 0 0 1 0 1 0 0 0 0];        %a1+beta<1%  length(A)= length(x0) 
b  =  [1  -  GARCH_TOLERANCE;1  -  GARCH_TOLERANCE];  %x0=[mu a01 a02 a11 a12 beta1 beta2 p q gama1 asym]
%A=[];
%b=[];
LB =[-5; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE;-.99;-1];
UB =[5;1;1;1;1;1;1;1;1;.99;1];

[coefficients, LLF   , ...
 exitFlag    , output, lambda] =  fmincon(objectiveFunction     , x0          , ...
                                          A                     , b           , ...
                                          []                    , []          , ...
                                          LB                    , UB          , ...
                                          []                    ,Optimization , ...
                                          r  );
end  

function [loglik,L,est] = snswitchasym(theta,r)

mu = theta(1);
eps = r-mu;
eps=eps';
T = length(eps);
a0 = theta(2:3)';
a1 = theta(4:5)';
beta = diag(theta(6:7));
p =theta(8);
q =theta(9);
P = [p,1-q;1-p,q];
pinf = [(1-q)/(2-p-q);(1-p)/(2-p-q)];

gam1 = theta(10);
delta = sign(gam1).*sqrt(pi*(gam1.^2).^(1/3))./sqrt(2*(gam1.^2).^(1/3)+2^(1/3)*(4-pi)^(2/3));
alpha = sign(delta).*sqrt(delta.^2./(1-delta.^2));

asym = theta(11);

ksi = zeros(2,T+1);
xsi = zeros(2,T);
h = zeros(2,T);
loglik = zeros(T,1);
h(:,1) = repmat(std(eps),2,1)./sqrt(1-2/pi*delta.^2);
ksi(:,2) = pinf;

for t = 2:T
    h(:,t) = a0+a1*(abs(eps(t-1,1))-asym*eps(t-1,1))+beta*h(:,t-1);
    LL(1,1)=ksi(1,t)*skewnormpdf(eps(t,1),h(1,t),alpha);
    LL(2,1)=ksi(2,t)*skewnormpdf(eps(t,1),h(2,t),alpha);
    xsi(:,t) = LL/sum(LL);
    ksi(:,t+1)=P*xsi(:,t);
    loglik(t,1)=log(ones(1,2)*LL);
end
if p>q
    est = [mu,a0',a1',diag(beta)',asym,gam1,p,q];
else
    est = [mu,a0(2),a0(1),a1(2),a1(1),beta(2,2),beta(1,1),asym,gam1,q,p];
end

L = -ones(1,T-1)*loglik(2:T)/T;
loglik=L*T;
%%%%
loglik(~isfinite(loglik))  =  1.0e+20;
end

function f = skewnormpdf(eps,sigma,alpha)

delta = alpha./sqrt(1+alpha.^2);
m = delta*sqrt(2/pi);
f = 2./sigma.*normpdf(eps./sigma+m).*normcdf(alpha.*(eps./sigma+m));
end
