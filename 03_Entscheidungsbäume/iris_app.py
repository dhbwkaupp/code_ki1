import tkinter as tk
import numpy as np
import pickle
import sklearn

import tkinter as tk

# Funktion, die beim Klick auf den Button ausgeführt wird
def on_button_click():
    # Werte einlesen und in Feld schreiben
    messwerte = []
    messwerte.append( float(entry1.get()) )
    messwerte.append( float(entry2.get()) )
    messwerte.append( float(entry3.get()) )
    messwerte.append( float(entry4.get()) )
    messwert2d = []
    messwert2d.append(messwerte)
    messwerte = np.array(messwert2d)

    # Vorhersage machen und als Text ausgeben
    prediction = loaded_model.predict(messwerte)
    results = ["Iris setosa", "Iris virginica", "Iris versicolor"]
    text = results[prediction[0]-1]
    eText.set(text)


# Hauptfenster erstellen
root = tk.Tk()
root.title("Iris klassifizieren")

# Labels und Textfelder erstellen
label1 = tk.Label(root, text="Länge Kelchblatt:")
label1.grid(row=0, column=0, padx=10, pady=5)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1, padx=10, pady=5)

label2 = tk.Label(root, text="Breite Kelchblatt:")
label2.grid(row=1, column=0, padx=10, pady=5)
entry2 = tk.Entry(root)
entry2.grid(row=1, column=1, padx=10, pady=5)

label3 = tk.Label(root, text="Länge Kronblatt:")
label3.grid(row=2, column=0, padx=10, pady=5)
entry3 = tk.Entry(root)
entry3.grid(row=2, column=1, padx=10, pady=5)

label4 = tk.Label(root, text="Breite Kronblatt:")
label4.grid(row=3, column=0, padx=10, pady=5)
entry4 = tk.Entry(root)
entry4.grid(row=3, column=1, padx=10, pady=5)

label5 = tk.Label(root, text="Vorhersage:")
label5.grid(row=4, column=0, padx=10, pady=5)
eText = tk.StringVar()
entry5 = tk.Entry(root, state='readonly', textvariable = eText)
entry5.grid(row=4, column=1, padx=10, pady=5)

# Button erstellen
button = tk.Button(root, text="Klicken", command=on_button_click)
button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Modell laden
file_name = "modell.pickle"
loaded_model = pickle.load(open(file_name, "rb"))

# Hauptschleife starten
root.mainloop()
