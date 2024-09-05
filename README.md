Step one: Parameters of models and conditional volatility are estimated by:

'snmsgarch.m':Skew-Normal Markov-Switching GARCH

'nmsgarch.m': Normal Markov-Switching GARCH

'snsinglegarch.m': Skew-Normal Single-Regime GARCH

'nsinglegarch.m': Normal Single-Regime GARCH

Step two: VaR is estimated by 'varpredicts.m' through rolling window method

Step three: VaR results is backtested by:

'kupiec.m': unconditional coverage test

'independence.m': independence test

'christofferson.m': conditional coverage test 

NOTE: All results for each model including estimated parameters, conditional volatility, VaRs and VaR backtesting  are available for download in the file 'results.zip'. 

To download results, open 'results.zip' and click on 'View raw'



