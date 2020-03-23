from scipy.optimize import minimize
from random import random, uniform
import math
import numpy as np
from prettytable import PrettyTable



"""
TODO:
- Formel ist für spiele ausgelegt, bei denen nur auf ein Team gewettet wird
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
    return -(sum([(p * o - 1) * b for p, o, b in zip(P, O, B)]) /\
    math.sqrt(sum([(1-p) * p * b**2 * o**2 for p, o, b in zip(P, O, B)])))

def expec(B):
    return sum([(p * o - 1) * b for p, o, b in zip(P, O, B) if b > 1.0 ])


def maxSharpe(P, O):
    """
    P = Predictions len(P) = 2 * NUMB_Matches
    O = Odds         -------- "" ------------
    """

    # Constraint für bets: alle Einsätze zusammen müssen kleiner 100 sein
    #constr = [{'type': 'ineq', 'fun': lambda x: sum(x) - 100}]
              #{'type': 'eq', 'fun': constb2} ]
    bnds = [(0, 500) for i in range(len(P))]
    bets = [10 for i in range(len(P))]

    sol = minimize(objective, x0=bets, args=(P,O,),tol = 1E-4,method='SLSQP', bounds=bnds )
    #print(sol.x, len(sol.x))
    # for i in range(0, len(sol.x)):
    #      b1 = sol.x[i]
    #      print(b1.round(2))
    # # print("Pred:", P)
    # print("Odds:", O)
    # print("Maximized Sharpe Ratio:", objective(sol.x,P, O))
    # # TODO: nochmal schauen wie Profit und Einsatz zusammenhängen!
    # print("Einsatz:" , sum(sol.x))
    # print("Expected Profit:", expec(sol.x))
    return expec(sol.x), sol , objective(sol.x,P, O)






if __name__ == "__main__" :
    table = PrettyTable()
    table.field_names = ["Home Team", "Away Team", "Bet on", "Bet size"]
    O = [1.1, 2.5, 3.0, 1.7, 4.0, 1.2, 1.01, 1.9, 2.8, 1.4]
    #P = [("TeamA",0.4), ("Team B",0.6), ("Team C",0.7), ("Team D",0.4),("Team E", 0.2), ("Team F",0.7),\
    #("Team G",0.8), ("Team H",0.51), ("Team I",0.75), ("Team J",0.3)]
    P = [0.4,0.6,0.7,0.4,0.2, 0.7, 0.8, 0.51, 0.75, 0.3]
    bestSharpe = 10
    preds, odds = [], []
    thresholdes = [0.0, 0.1, 0.2, 0.3]

    for threshold in thresholdes:
        for p, o in zip(P,O):
            if abs(p - 0.5) > threshold:
                preds.append(p)
                odds.append(o)
        if len(odds) == len(preds) and len(odds) > 0:
            profit, sol, sharpe = maxSharpe(preds, odds)
            print(profit, sharpe)
            if sharpe < bestSharpe:
                bestProfit = profit
                bestSol = sol
        preds, odds = [], []


    table.add_row([ , , , ])

    print("=== How to bet on the following games ===")
    print("Expected Profit: ", bestProfit)

    # print("Bets:", bestSol.x)
    # print("Einsatz:" , sum(sol.x))
    # print("Expected Profit:", expec(sol.x))
