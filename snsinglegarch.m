function [LogLF, coeffs, sigmaforcast] = snsinglegarch(x0, Ret, n1, n2, d)
% This function estimates parameters, conditional volatility and LLF for Skew-Normal Single-Regime GARCH model

%x0: initial value/size(x0)=(1,6)
%cl: confidence level/ [0.99 0.975 0.95 0.90]
%coeffs: estimated coefficients
%d: forecasting window (d=5)
%LogLF: loglikelihood function
%n1: estimation window
%n2: out-of-sample size (n2=1110 days)
%Ret: log-returns (entire dataset)/ It should be a row vector
%sigmaforcast: estimated conditional volatility 

    sigmapred = zeros(1, n2);  
    coeffs = zeros(n2/d, 6);
    LogLF = zeros(n2/d, 1);
    j = n2/d - 1;

    for i = 0:j
        r = Ret(1+d*i:n1+d*i);  % r: pre-sample % update the parameters based on forecasting window

        [coefficients, LLF] = snsingleasymfit(x0, r); % estimate the parameters via MLE
        coeffs(i+1, :) = coefficients;
        LogLF(i+1) = LLF;

        innovation = Ret(1+d*i:n1+d+d*i) - coefficients(1); % innovation = epsilon
        innovation = innovation';

        a00 = coefficients(2);  % a0
        a11 = coefficients(3);  % alpha
        betaa = coefficients(4);  % beta
        gamm1 = coefficients(5);
        deltaa = sign(gamm1) * sqrt(pi * (gamm1^2)^(1/3)) / sqrt(2 * (gamm1^2)^(1/3) + 2^(1/3) * (4 - pi)^(2/3));
        alphaa = sign(deltaa) * sqrt(deltaa^2 / (1 - deltaa^2));
        asymm = coefficients(6);
        m = sqrt(2 / pi) * deltaa;
        T = length(r);
        sigma = zeros(1, T + d);

        if i == 0
            sigma(:, 1) = std(r) / sqrt(1 - m^2);
            for t = 2:T+d
                sigma(:, t) = a00 + a11 * (abs(innovation(t-1, 1)) - asymm * innovation(t-1, 1)) + betaa * sigma(:, t-1);
            end
        else
            sigma(:, 1) = a00 + a11 * (abs(innovation0) - asymm * innovation0) + betaa * sigma0;
            for t = 2:T+d
                sigma(:, t) = a00 + a11 * (abs(innovation(t-1, 1)) - asymm * innovation(t-1, 1)) + betaa * sigma(:, t-1);
            end
        end
        sigmapred(:, 1+d*i:d+d*i) = sigma(:, T+1:T+d);
        innovation0 = innovation(d, 1);
        sigma0 = sigma(:, d);
    end
    sigmaforcast = sigmapred';
end

function [coefficients, LLF] = snsingleasymfit(x0, r)
    nParameters = length(x0);
    Optimization = optimset('fmincon');

    if isinf(optimget(Optimization, 'MaxSQPIter'))
        Optimization = optimset(Optimization, 'MaxSQPIter', 1000 * nParameters);
    end

    global GARCH_TOLERANCE
    GARCH_TOLERANCE  =  2*optimget(Optimization , 'TolCon', 1e-7);

    Optimization = optimset(Optimization, ...
                            'MaxFunEvals', 200 * nParameters, ...
                            'MaxIter', 1000, ...
                            'TolFun', 1e-6, ...
                            'TolCon', 1e-7, ...
                            'TolX', 1e-6, ...
                            'Display', 'iter', ...
                            'Diagnostics', 'on', ...
                            'LargeScale', 'off');
    Optimization  =  optimset(Optimization , 'LargeScale'  , 'off');


    objectiveFunction = @snsingleasym;

    A = [0 0 1 1 0 0];  % a1 + beta < 1
    b = 1 - GARCH_TOLERANCE;
    LB = [-5; GARCH_TOLERANCE; GARCH_TOLERANCE; GARCH_TOLERANCE; -0.99; -1];
    UB = [5; 0.99; 0.99; 0.99; 0.99; 1];

    [coefficients, LLF, exitFlag, output, lambda] = fmincon(objectiveFunction, x0, A, b, [], [], LB, UB, [], Optimization, r);
end

function [loglik, L, est, F] = snsingleasym(theta, r)
    mu = theta(1);
    a0 = theta(2);
    a1 = theta(3);
    beta = theta(4);
    gam1 = theta(5);

    
    delta = sign(gam1) * sqrt(pi * (gam1^2)^(1/3)) / sqrt(2 * (gam1^2)^(1/3) + 2^(1/3) * (4 - pi)^(2/3));
    alpha = sign(delta) * sqrt(delta^2 / (1 - delta^2));
    m = sqrt(2 / pi) * delta;

    asym = theta(6);
    T = length(r);
    h = zeros(T, 1);
    eps = r - mu;
    eps = eps';
    h(1) = std(r) / sqrt(1 - m^2);

    for t = 2:T
        h(t, 1) = a0 + a1 * (abs(eps(t-1)) - asym * eps(t-1)) + beta * h(t-1, 1);
    end

    eta = eps(2:end) ./ h(2:end);
    f = density(eta, alpha) ./ h(2:end);
    L = -sum(log(f)) / T;
    loglik = L * T;
    loglik(~isfinite(loglik)) = 1.0e+20;
    est = [mu, a0, a1, beta, asym, alpha];

    xi = eta(end-19:end) + sqrt(2 / pi) * delta;
    for t = 1:length(xi)
        F(t, 1) = skewnormcdf1(xi(t), alpha);
    end
end

function f = density(z, alpha)
    % pdf of SN(alpha) (z is a random variable like x)
    delta = alpha / sqrt(1 + alpha^2);
    m = sqrt(2 / pi) * delta;  % m: if z~SN(alpha) ==> mu=E(z)=sqrt(2/pi)*delta
    f = 2 * normpdf(z + m) .* normcdf(alpha * (z + m));  % pdf of SN
end

function f = owenfunction(x, z)
    f = exp(-0.5 * z.^2 .* (1 + x.^2)) ./ (1 + x.^2) / (2 * pi);
end

function F = skewnormcdf1(z, alpha)
    try
        T = quadl(@owenfunction, 0, alpha, 1e-12, [], z);
    catch ME
        disp('Error in integration:');
        disp(ME.message);
        T = 0;  
    end
    F = normcdf(z) - 2 * T;
end
