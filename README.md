# csgo_betting_bot

## Überlegungen zu Data Format: 

### Daten 4 Dimensional machen:
* über erste Dimension iterieren. ==> keine Batchsize: (2, 5, x) 2 Teams, a 5 Spieler plus ihre x Feature

### Daten so lassen: 
* Können die Batch Dimension als mehrere Spiele sehen. kann auch funktionieren. 

==> Muss man vl Vergleichen

## Überlegung zu NN: 

* Wie im Paper nachbauen (viel aufwand, undurchsichtig, Leitung sehr gut )
* Dense Layer wie im Artikel (einfach, schnell, Leistung?)
* Bi-LSTM oder Dense für beide Teams und über dense layer zusammenführen. 
