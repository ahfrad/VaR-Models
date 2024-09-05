function [LR_indep,pvalue_indep, exc_indep] = independence(Ret, VaR_short, VaR_long, cl, n1)
% Independence test for VaR Backtesting
% Inputs:
% %Ret: log-returns (entire dataset)/ It should be a row vector
% VaR_short: matrix of VaR thresholds for short positions
% VaR_long: matrix of VaR thresholds for long positions
%cl: confidence level/ [0.99 0.975 0.95 0.90]
%n1: estimation window
% Outputs:
% pvalue_indep - p-values for the independence test
% exc_indep - number of exceptions for independence test

r = Ret(n1+1:end)'; % Returns for backtesting period
N = length(r);
k = length(cl);
pvalue_indep = zeros(2, k);
exc_indep = zeros(2, k);
alpha = 1 - cl;

for j = 1:k
    % Generate violation indicators
    breaches_short = r > VaR_short(:, j); % violations for short positions (corrected)
    breaches_long = r < VaR_long(:, j); % violations for long positions
    
    % Ensure vectors are column vectors
    breaches_short = breaches_short(:);
    breaches_long = breaches_long(:);
    
    % Compute for both short and long positions
    for pos = 1:2
        if pos == 1
            breaches = breaches_short;
        else
            breaches = breaches_long;
        end
        
        % Compute the transition matrix
        N00 = sum(breaches(1:end-1) == 0 & breaches(2:end) == 0);
        N01 = sum(breaches(1:end-1) == 0 & breaches(2:end) == 1);
        N10 = sum(breaches(1:end-1) == 1 & breaches(2:end) == 0);
        N11 = sum(breaches(1:end-1) == 1 & breaches(2:end) == 1);
        
        % Transition probabilities
        p01 = N01 / (N00 + N01);
        p11 = N11 / (N10 + N11);
        
        % Overall violation rate
        p = (N01 + N11) / (N00 + N01 + N10 + N11);
        
        % Likelihood ratio test statistic for independence
        L0 = (1 - p)^(N00 + N10) * p^(N01 + N11);
        Lobs = (1 - p01)^N00 * p01^N01 * (1 - p11)^N10 * p11^N11;
        LR_indep(pos, j) = -2 * log(L0 / Lobs);
        
        % p-value in percentage for independence
        pvalue_indep(pos, j) = 100 * (1 - chi2cdf(LR_indep(pos, j), 1));
                    
        % Save number of exceptions
        exc_indep(pos, j) = sum(breaches);
    end
end
end