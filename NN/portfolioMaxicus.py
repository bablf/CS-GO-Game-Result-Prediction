import math
import numpy as np

from tqdm import tqdm
from random import random, uniform
from prettytable import PrettyTable
from scipy.optimize import minimize


"""
TODO:
- Formel ist für spiele ausgelegt, bei denen nur auf ein Team gewettet wird
Input:
:predictions:   Liste von predictions für jedes Match. 2N
:odds:          Liste von Odds für jedes Match. 2N

"""

def objective(B, P, O):
    return float(-(sum([(p * o - 1) * b for p, o, b in zip(P, O, B)]) /\
    math.sqrt(sum([(1-p) * p * b**2 * o**2 for p, o, b in zip(P, O, B)]))))

def expec(B, P, O):
    return sum([(p * o - 1) * b for p, o, b in zip(P, O, B) if b > 0.01 ])

def profit2einsatz(profit, einsatz):
    return profit/einsatz

def maxSharpe(P, O, beg_bet):
    """
    P = Predictions len(P) = 2 * NUMB_Matches
    O = Odds         -------- "" ------------
    """

    # Constraint für bets: alle Einsätze zusammen müssen kleiner 100 sein
    #constr = [{'type': 'ineq', 'fun': lambda x: sum(x) - 1}]
              #{'type': 'eq', 'fun': constb2} ]
    bnds = [(0, 1000) for i in range(len(P))]
    bets = [beg_bet for i in range(len(P))]
    sol = minimize(objective, x0=bets, args=(P,O,), tol= 1e-10, method='SLSQP', bounds=bnds)#, constraints=constr )
    #print(sol.x, len(sol.x))

    return expec(sol.x, P, O), sol , objective(sol.x,P, O)
"""
Lessons Learned aus Beispiel:

Beggining bets sind teilweise irrelevant, da:
Maximized Sharpe Ratio immer ca -0.5826443188276685 ist.
Sind aber wichtig, wenn man den Profit maximieren möchte

Profit zu Einsatz bleibt meist recht gleich: um die 0.5272967004677384

durch tol sehr klein machen, wird garantiert, dass nur auf ein Spiel getippt wird
ABER IST ES SINNVOLL AUF BEIDES ZU TIPPEN, UM PROFITVERLUST ZU VERMEIDEN?
==> Weniger Gewinn ==> ezBetticus besser machen, dann kein Problem mehr.

bounds werden nicht gebrochen, wohingegegen constraints gebrochen werden können

=> Nur der Profit verändert sich.   1000€ setzen    => 500€ Profit.
                                    15€ setzen      => 8€  Profit.


"""

def calc_portfolio(preds, odds, teams):
    bestSharpe = -10
    bestProfit = 0
    thresholdes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    P, O, T = [],[],[]
    for beg_bet in tqdm(range(1, 101)):
        for threshold in thresholdes:
            for p, o, t in zip(preds, odds, teams):
                if abs(p - 0.5) > threshold:
                    P.append(p)
                    O.append(o)
                    T.append(t)
            if len(O) == len(P) and len(O) > 0:
                profit, sol, sharpe = maxSharpe(P, O,beg_bet)
                #print(profit, sharpe)
                if profit > bestProfit:
                #if sharpe > bestSharpe: # if maximized even more:
                    # print("==========")
                    # print(threshold)
                    # x = [round(y,7)for y in sol.x]
                    # print("Bets:", x)
                    # print("best beginning bet:", beg_bet)
                    # print("Einsatz:" , sum(sol.x))
                    # print("Expected Profit: ", profit)
                    # print("Maximized Sharpe Ratio:", objective(sol.x, P, O))
                    bestSharpe = sharpe
                    bestProfit = profit
                    bestSol = sol
                    bestP = P
                    bestO = O
                    bestT = T
                    bestbeg_bet = beg_bet
            P, O = [], []

    # print("=== How to bet on the following games ===")
    bestSol.x = [round(y,2) for y in bestSol.x]
    # print("Bets:", bestSol.x)
    # print("best beginning bet:", bestbeg_bet)
    # print("Einsatz:" , sum(bestSol.x))
    # print("Expected Profit:", expec(bestSol.x,bestP, bestO))
    # print(profit2einsatz(bestProfit, sum(bestSol.x)))
    # print("Maximized Sharpe Ratio:", objective(bestSol.x,bestP, bestO))

    return bestSol.x, expec(bestSol.x, bestP, bestO), bestP, bestO, bestT
