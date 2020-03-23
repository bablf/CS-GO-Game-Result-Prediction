from scipy.optimize import minimize
from random import random, uniform
import math
import numpy as np
"""
TODO:
- Formel ist für spiele ausgelegt, bei denen nur auf ein Team gewettet wird
==> Wie sage ich das minimize?

Input:
:predictions:   Liste von predictions für jedes Match. 2N
:odds:          Liste von Odds für jedes Match. 2N

"""

def objective(B, P, O):
    # exp = 0
    # var = 0
    # print(B)
    # for (p1, p2), (o1,o2), (b1,b2) in zip(P,O,B):
    #     exp += (p1 * o1 -1) * b1 + (p2 * o2 -1) * b2
    #     var += ((1-p1) * p1 * b1**2 * o1**2) + ((1-p2) * p2 * b2**2 * o2**2)
    # return - exp / math.sqrt(var)
    return - (sum([(p * o - 1) * b for p, o, b in zip(P, O, B)]) /\
    math.sqrt(sum([(1-p) * p * b**2 * o**2 for p, o, b in zip(P, O, B)])))

def expec(B):
    return sum([(p * o - 1) * b for p, o, b in zip(P, O, B)])


def constb1(B):
    for i in range(0, len(B),2):
        b1, b2 = B[i:i+2]
        if b1 == 0.0:
            return b2 + b1 + b2
        else:
            return 42

def constb2(B):
    for i in range(0, len(B), 2):
        b1, b2 = B[i:i+2]
        print(b1, b2)
        if b2 > 0.0:
            return b2 + b1 - b2


"""def objective(P,O,B):
    return - (sum([(p * o - 1) * b for p, o, b in zip(P, O, B)])) /\
     math.sqrt(sum([(1-p) * p * b**2 * o**2 for p, o, b in zip(P, O, B)]))
"""


def maxSharpe(P, O):
    """
    P = Predictions len(P) = 2 * NUMB_Matches
    O = Odds         -------- "" ------------

    """

    # Constraint für prediction: nur wetten wenn > 50% oder
    # Constraint für bets: alle Einsätze zusammen müssen kleiner 100 sein
    constr = [{'type': 'ineq', 'fun': lambda x: sum(x) - 100}]
              #{'type': 'eq', 'fun': constb2} ]
    bnds = [(0, 20) for i in range(len(P))]
    bets = [100 for i in range(len(P))]
    #B = [1,2,3,4,5,6,7,8,9,10]
    sol = minimize(objective, x0=bets, args=(P,O,), method='SLSQP', bounds=bnds, constraints=constr)
    print(sol)
    print("Bets:",sol.x)
    print("Pred:", P)
    print("Odds:", O)
    print("Maximized Sharpe Ratio:", objective(sol.x,P, O))
    # TODO: nochmal schauen wie Profit und Einsatz zusammenhängen!
    print("Einsatz:" , sum(sol.x))
    print("Expected Profit:", expec(sol.x))
    return expec(sol.x), sol , objective(sol.x,P, O)






if __name__ == "__main__" :

#    O = [(uniform(1.1, 5.9), uniform(1.1, 5.9)) for _ in range(0, 10)]
#    P = [(random(),random()) for _ in range(0, 10)]
    O = [(uniform(1.1, 5.9)) for _ in range(0, 10)]
    P = [(random()) for _ in range(0, 10)]

    bestSharpe = -100
    preds, odds = [], []
    thresholdes = [0.0, 0.1, 0.2, 0.3]
    # for p, o in zip(P,O):
    #     #if abs(p - 0.5) > threshold :
    #     preds.append(p)
    #     odds.append(o)
    #print(len(odds))
    #if len(odds) == len(preds):

    profit, sol, sharpe = maxSharpe(P, O)
    for i in range(0, len(sol.x), 2):
        b1, b2 = sol.x[i:i+2]
        print(b1,b2)
    # for threshold in thresholdes:
    #     for p, o in zip(P,O):
    #         if abs(p - 0.5) > threshold :
    #             preds.append(p)
    #             odds.append(o)
    #     print(len(odds))
    #     if len(odds) == len(preds):
    #         profit, sol, sharpe = maxSharpe(preds, odds)
    #         if sharpe > bestSharpe:
    #             bestProfit = profit
    #             bestSol = sol
    #     preds, odds = [], []
    #
    #
    # print(bestSol)
    # print("Bets:", bestSol.x)
    # print("Einsatz:" , sum(sol.x))
    # print("Expected Profit:", expec(sol.x))
