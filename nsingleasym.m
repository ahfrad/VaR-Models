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




