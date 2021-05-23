# Short-Time Fourier-Transform (STFT)

Interaktives Notizbuch zur Kurzzeit-Fourier-Transformation in Signale und Systeme.
Neben dem interaktiven Notizbuch kann die Arbeit auch über einen Webbrowser dargestellt werden. Dafür ist eine HTML Datei beigefügt.

<br> Autor: Andreas Weber
<br> Matrikelnummer: 1540399
<br> Dozent: Prof. Dr. Janko Dietzsch
<br> Datum: 28.05.2021

# Inhalt

1. Motivation
2. Fehlen der Zeitauflösung in der Fourier-Transformation
3. Definition der Kurzzeit-Fourier-Transformation
   1. Zeitkontinuierliche STFT
   2. Zeitdiskrete STFT
4. Fensterung eines Signals
5. Zeit-Frequenz-Auflösung
   1. Unschärferelation
   2. Darstellung
      1. Beispiel 1: Sinussignal mit zwei verschiedenen Frequenzen
      2. Beispiel 2: Chirp Signal
      3. Beispiel 3: Musik als Signal
6. Literatur

# Notwendige Packages

- matplotlib
- numpy
- seaborn
- scipy
- sounddevice
- librosa
- ipywidgets

Verwendete Python Version: Python 3.8.5
