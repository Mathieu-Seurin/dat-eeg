Avant toute chose :
=========

Base de données :
-------

Disponible ici : (2 fichiers .mat)
https://drive.google.com/folderview?id=0B6LRgmFU1UzkOG56NWkwaGVNQW8&usp=sharing

À placer dans le dossier Data/ dans code/

Bibliothéques nécessaire :
------
Sklearn 0.17 
Matplotlib
Numpy
Scipy




Explication
========
Dans le dossier code, vous avez 5 fichiers python :

- learnData (l'ensemble de mes outils pour l'apprentissage, création de modèles, sélection ect...)
- manipulateData (Outils pour manipuler les bases de données, reformater, filtrer ...)
- signalManipulation (Outils pour la manipulation des signaux, plot, STFT etc...)
- testSignal (Batterie de test pour plot des jolis signaux)
- wholeProcess (Fichiers pour lancer l'apprentissage des modèles)


Quelque exemples d'executables :
==========

Signal
----
python3 testSignal sin 
\# Pour afficher les différentes STFT du signal

python3 testSignal filter
\# Exemple de signal filtré, d'abord sur un sinus puis sur un vrai signal EEG

python3 testSignal p300
\# Exemple de p300 avec sa FFT et quelques STFT


Learning
-------

python wholeProcess.py AF lin    \# Subject A, filter, linear model

python wholeProcess.py test nonLin    \# Dummy Data, non Linear model

python wholeProcess.py ABF elastic    \#Subject A and B (attention, avec AB cela peut manger toute votre RAM et provoquer des "légers" ralentissements de votre ordinateur)



