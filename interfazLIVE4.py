#se cambia lo del puerto fijo en com5 y 7 
# EMG-FES Real-Time GUI with Classification & Heatmap Overlay
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import serial, time, threading, csv
import numpy as np
import tensorflow as tf
import scipy.signal as signal
import os

# ===================== CONFIG =====================
fs = 750
ventana_ms = 500
samples_por_ventana = int((ventana_ms / 1000) * fs)
canales = 5
nombres_clases = ["Índice", "Middle", "Anular", "Meñique", "Nada"]
modelo_path = "modelo72nomover.h5"

# ================= FILTRADO ======================
def notch_filter(data, freq, fs):
    b, a = signal.iirnotch(freq, 30, fs)
    return signal.filtfilt(b, a, data)

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = signal.butter(4, [lowcut, highcut], btype='band', fs=fs)
    return signal.filtfilt(b, a, data)

def process_buffer(buffer):
    buffer = np.array(buffer, dtype=np.float32)
    for i in range(canales):
        buffer[:, i] -= np.mean(buffer[:, i])
        buffer[:, i] = notch_filter(buffer[:, i], 60, fs)
        buffer[:, i] = notch_filter(buffer[:, i], 50, fs)
        buffer[:, i] = bandpass_filter(buffer[:, i], 20, 374, fs)
    return buffer

# ================= INTERFAZ ======================
class EMGFESApp:
    def __init__(self, master):
        self.master = master
        master.title("EMG-FES Interfaz con Heatmap")
        master.geometry("600x700")
        self.running = False
        self.session_data = []

        # Modelo
        self.modelo = tf.keras.models.load_model(modelo_path)

        # Imagen base
        base = Image.open("mano.png").resize((512, 512))
        self.img_base = ImageTk.PhotoImage(base)

        self.canvas = tk.Canvas(master, width=512, height=512)
        self.canvas.pack(pady=10)
        self.canvas.create_image(0, 0, anchor='nw', image=self.img_base)

        # Overlay por dedo
        self.heat_overlays = {
            "Índice": self.canvas.create_oval(160, 60, 200, 100, fill='', outline=''),
            "Middle": self.canvas.create_oval(220, 50, 260, 90, fill='', outline=''),
            "Anular": self.canvas.create_oval(280, 60, 320, 100, fill='', outline=''),
            "Meñique": self.canvas.create_oval(340, 90, 380, 130, fill='', outline='')
        }

        self.label_mov = tk.Label(master, text="Esperando...", font=("Arial", 16))
        self.label_mov.pack(pady=5)

        self.start_btn = tk.Button(master, text="Iniciar Sesión", command=self.iniciar_sesion)
        self.start_btn.pack()

        self.stop_btn = tk.Button(master, text="Detener", command=self.detener_sesion, state='disabled')
        self.stop_btn.pack()

        self.save_btn = tk.Button(master, text="Guardar Sesión", command=self.guardar_sesion, state='disabled')
        self.save_btn.pack()

        self.load_btn = tk.Button(master, text="Reproducir Sesión", command=self.reproducir_sesion)
        self.load_btn.pack()

        # Solicitar puertos al usuario
        puerto_emg = simpledialog.askstring("Puerto EMG", "Ingresa el puerto COM para EMG (ej. COM5):")
        puerto_fes = simpledialog.askstring("Puerto FES", "Ingresa el puerto COM para FES (ej. COM7):")

        try:
            self.ser_emg = serial.Serial(puerto_emg, 2000000, timeout=1)
            self.ser_fes = serial.Serial(puerto_fes, 9600, timeout=1)
            time.sleep(2)
        except:
            messagebox.showerror("Error", "No se pudo abrir uno de los puertos.")
            master.destroy()

    def actualizar_heatmap(self, dedo_activo):
        for dedo, shape in self.heat_overlays.items():
            color = 'red' if dedo == dedo_activo else ''
            self.canvas.itemconfig(shape, fill=color, outline=color)

    def clasificar(self):
        buffer = []
        while self.running:
            try:
                linea = self.ser_emg.readline().decode('utf-8').strip()
                valores = list(map(int, linea.split(",")))
                if len(valores) == canales:
                    buffer.append(valores)
                if len(buffer) >= samples_por_ventana:
                    ventana = np.array(buffer[-samples_por_ventana:])
                    ventana_filtrada = process_buffer(ventana)
                    entrada = ventana_filtrada.reshape(1, samples_por_ventana, canales)
                    pred = self.modelo.predict(entrada, verbose=0)
                    clase = np.argmax(pred)
                    comando = nombres_clases[clase]

                    self.label_mov.config(text=f"🟢 {comando}")
                    self.actualizar_heatmap(comando)

                    if clase in [0, 1, 2, 3]:
                        self.ser_fes.write(f"{clase}\n".encode())

                    self.session_data.append([time.time(), *valores, comando])
                    buffer = []
            except:
                pass

    def iniciar_sesion(self):
        if not self.running:
            respuesta = messagebox.askyesno("Iniciar", "¿Deseas comenzar la sesión de lectura?")
            if not respuesta:
                return
            self.running = True
            self.hilo = threading.Thread(target=self.clasificar)
            self.hilo.start()
            self.label_mov.config(text="🟡 Grabando...")
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.save_btn.config(state='disabled')
            self.load_btn.config(state='disabled')

    def detener_sesion(self):
        if self.running:
            self.running = False
            self.label_mov.config(text="⏹️ Sesión detenida")
            messagebox.showinfo("Detenido", "La sesión ha sido detenida.")
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.save_btn.config(state='normal')
            self.load_btn.config(state='normal')

    def guardar_sesion(self):
        if not self.session_data:
            messagebox.showwarning("Sin datos", "Primero graba una sesión")
            return
        archivo = filedialog.asksaveasfilename(defaultextension=".csv")
        if archivo:
            with open(archivo, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'A0', 'A1', 'A2', 'A3', 'A4', 'Movimiento'])
                writer.writerows(self.session_data)
            messagebox.showinfo("Guardado", f"Sesión guardada en: {archivo}")
            self.session_data = []
            self.label_mov.config(text="✅ Sesión guardada")

    def reproducir_sesion(self):
        if self.running:
            messagebox.showwarning("En curso", "Detén la sesión actual antes de reproducir.")
            return
        archivo = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if archivo:
            respuesta = messagebox.askyesno("Reproducir", "¿Deseas iniciar la reproducción de esta sesión?")
            if not respuesta:
                return
            self.label_mov.config(text="🎬 Reproduciendo...")
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            self.load_btn.config(state='disabled')

            with open(archivo, 'r') as f:
                reader = csv.DictReader(f)
                for fila in reader:
                    movimiento = fila['Movimiento']
                    self.label_mov.config(text=f"🎬 {movimiento}")
                    self.actualizar_heatmap(movimiento)
                    time.sleep(0.3)

            messagebox.showinfo("Fin", "La reproducción ha finalizado.")
            self.label_mov.config(text="✅ Reproducción terminada")
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            self.load_btn.config(state='normal')

# Ejecutar
if __name__ == "__main__":
    root = tk.Tk()
    app = EMGFESApp(root)
    root.mainloop()
