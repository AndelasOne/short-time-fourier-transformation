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

# Binder

Auf dem Binder Server kann auch ein Großteil der Arbeit angeschaut werden. Allerdings kann die Python Bibliothek "sounddevice" und "librosa" nicht ohne Weiteres in dem Docker Container mit dem Standardimage installiert werden.

https://mybinder.org/v2/gh/AndelasOne/SigSys_STFT_1540399/66f9922602c33a106ae263a952599aaa8582ef7f
