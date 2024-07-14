import librosa
import numpy as np
import sounddevice as sd
import wavio
import tkinter as tk
import tensorflow
from tkinter import ttk

def dispositivos_disponibles():
    dispositivos = sd.query_devices()
    microfonos = []
    for dispositivo in dispositivos:
        if dispositivo['max_output_channels'] == 0:
            microfonos.append(dispositivo['name'])

    return microfonos

def record():
    nombre_dispositivo = device_var.get()
    fs = 96000
    duracion_grabacion = 1
    
    print('Grabación iniciada')
    grabacion = sd.rec(int(duracion_grabacion * fs), samplerate=fs, channels=1, device=nombre_dispositivo)
    sd.wait()
    print('Grabación terminada, guardada como:', end=" ")

    # Guarda la grabación como "record.wav"
    wavio.write(path_record, grabacion, fs, sampwidth=2)

    print('record.wav')

def play():
    wav = wavio.read(path_record)
    data = wav.data
    fs = wav.rate
    sd.play(data, fs)

def preprocess_audio(file_path,max_len):
    # Cargar el audio
    signal, sr = librosa.load(file_path,sr=96000)
    # Realizar preénfasis
    #filter_audio = librosa.effects.preemphasis(signal)
    # Extraer MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    # Realizar padding o recorte
    if mfcc.shape[1] < max_len:
        num_zeros = max_len - mfcc.shape[1]
        padded_mfcc = np.pad(mfcc, ((0, 0), (0, num_zeros)), mode='constant', constant_values=0)
        return padded_mfcc
    else:
        mfcc = mfcc[:, :max_len]
        return mfcc

def prediccion(model):
    score = ''
    padding = 500
    preprocessed_audio = preprocess_audio(path_record,padding)
    preprocessed_audio = preprocessed_audio.reshape(1, preprocessed_audio.shape[0], preprocessed_audio.shape[1])  # Convertir a formato (1, features, tiempo)
    prediction = model.predict(preprocessed_audio)

    print(f"bed : {prediction[0][0] * 100}%")
    print(f"cat : {prediction[0][1] * 100}%")
    print(f"happy : {prediction[0][2] * 100}%")

    predicted_label_encoded = np.argmax(prediction, axis=1)[0]
    palabra_predicha = ''

    if predicted_label_encoded == 0:
        palabra_predicha = 'bed'
    elif predicted_label_encoded == 1:
        palabra_predicha = 'cat'
    elif predicted_label_encoded == 2:
        palabra_predicha = 'happy'

    print(f"\nLa palabra predicha es: {palabra_predicha}")
    score = max(prediction[0]) * 100

    return(palabra_predicha,score)

def processing():
    global text_box

    model = tensorflow.keras.models.load_model(path_model)    
    predict, score = prediccion(model)

    if text_box is None:
        text_box = tk.Label(root, font=("Arial", 12)) 
        text_box.pack()

    text_box.config(text=f'Palabra predicha: "{predict}" con un {score:.2f}% de precision')

#GUI=============================================================================================
path_model = 'IA_MFCCs/GUI/modelo.h5'
path_record = 'IA_MFCCs/GUI/record.wav'
text_box = None

root = tk.Tk()
root.geometry("400x300")
root.title("Proyecto Rec Voz")

# Cuadro desplegable en la parte superior
devices = dispositivos_disponibles()
device_var = tk.StringVar()
device_var.set(devices[0])

device_dropdown = ttk.OptionMenu(root, device_var, *devices)
device_dropdown.config(width=399)
device_dropdown.pack()

# Botones en el medio
frame = tk.Frame(root)
record_button = tk.Button(frame, text="⚫", bg="red", fg="white", padx=10, pady=10, command=record)
play_button = tk.Button(frame, text="▶", bg="white", fg="black", padx=10, pady=10, command=play)
processing_button = tk.Button(frame, text="Procesar", bg="white", fg="black", padx=10, pady=10, command=processing)
record_button.grid(row=0, column=0, padx=10, pady=10)
play_button.grid(row=0, column=1, padx=10, pady=10)
processing_button.grid(row=0, column=2, padx=10, pady=10)
frame.pack()

root.mainloop()
