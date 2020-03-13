# csgo_betting_bot

# TODO: While waiting for better data
1. Is Batchtraining necessary/helpful for performance or accuracy?
2. Loss function Nachhilfe auf YT/stackoverflow
3. Skalieren der Daten anschauen
4. (Portfolio Maximization (opt) anschauen)

## Überlegung zu NN:

Ideen:
- MSE auf "Sum" oder "Mean" setzen
- Team Based Model implementieren. Vielleicht bei uns bessere Ergebnisse.

### Bisherige Werte:
#### Model 1
- Lernrate auf 0.001 gesetzt
- train-test auf 66%-33% gesetzt.
- Dropout nach jedem Layer eingefügt.
- accuracy: 62% nach 2 Epochen
- loss: immernoch nicht verändert.

#### Model 2:
- Lernrate 0.001
- tanh anstelle von sigmoid:
- accuracy von 63% nach einer
- 64% nach fünf epoche.
- Loss immernoch unverändert.
