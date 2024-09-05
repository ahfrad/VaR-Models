function VaR= varpredicts (coeffs,sigmaforcast,cl,n2,d)   
%This function predicts VaR based on estimated parameters.
% Estimate VaR for short position with cl
% Estimate VaR for long position with '1-cl'

%cl: confidence level/ [0.99 0.975 0.95 0.90]
%coeffs: estimated coefficients
%d: forecasting window (d=5)
%n1: estimation window
%n2: out-of-sample size (n2=1110 days)
%sigmaforcast: estimated conditional volatility 


k=length(cl);  
VaR=zeros(n2,k);
MU=coeffs(:,1);
MU=reshape(repmat(MU,1,d)',n2,1);  

p=coeffs(:,8);
p=reshape(repmat(p,1,d)',n2,1);
q=coeffs(:,9);
q=reshape(repmat(q,1,d)',n2,1);
PI1=(1-q)./(2-p-q);   
PI2=(1-p)./(2-p-q);
PI=[PI1 PI2];        %PI: stationary distribution

gam1 = coeffs(:,10);
delta = sign(gam1).*sqrt(pi*(gam1.^2).^(1/3))./sqrt(2*(gam1.^2).^(1/3)+2^(1/3)*(4-pi)^(2/3));
delta=reshape(repmat(delta,1,d)',n2,1);
alpha = sign(delta).*sqrt(delta.^2./(1-delta.^2));
m=delta.*sqrt(2/pi);

for j=1:k
    
    for i=1:n2
     
        sigmaJ=sigmaforcast(i,:);
        x0=sum(PI(i,:).*SN([cl(j) cl(j)],[MU(i)-sigmaJ(1).*m(i)  MU(i)-sigmaJ(2).*m(i)],sigmaJ,[alpha(i) alpha(i)]));
    
        res= @(VaRx)(sum(PI(i,:).*SN([VaRx VaRx],[MU(i)-sigmaJ(1).*m(i)  MU(i)-sigmaJ(2).*m(i)],sigmaJ,[alpha(i) alpha(i)]),2)-cl(j));
    
        VaRn=fzero(res,x0); 
        VaR(i,j)=VaRn;
        
   end

end
end

function p = SN(a,b,c,d)
%This function helps to execute both regime's calculations together in lines 24,26
F1 = SNcdf(a(1),b(1),c(1),d(1));
F2=SNcdf(a(2),b(2),c(2),d(2));
p=[F1 F2];
end

function cdf= SNcdf(x,l,s,alfa)
% This function computes the cdf of the Skew-Normal distribution
%l: location parameter
%s: scale parameter
%alfa: degree of skewness

cdf=normcdf((x-l)/s)-2*owenfunction((x-l)/s,alfa);
end

function T = owenfunction(z,a)


f =@(y) exp(-.5*z.^2.*(1+y.^2))./(1+y.^2)/(2*pi);   
T=quadl(f,0,a,1e-10,[]);
%T = integral(f, 0, a, 'AbsTol', 1e-10);
end 