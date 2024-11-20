import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import threading


# Función para calcular la distancia euclidiana
def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

# Inicialización de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Variables globales
cap = None
running = False
historial_letras = []

# Crear ventana principal
root = tk.Tk()
root.title("Reconocimiento de Letras - Lenguaje de Señas")
root.resizable(True, True)  # Permitir que la ventana sea redimensionable

# Aplicar estilo ttk
style = ttk.Style(root)
style.theme_use('clam')  # Puedes probar otros temas como 'alt', 'default', 'classic'

# Marco principal
main_frame = ttk.Frame(root)
main_frame.pack(fill='both', expand=True)

# Configurar las filas y columnas del grid
main_frame.columnconfigure(0, weight=3)  # Columna para la cámara
main_frame.columnconfigure(1, weight=1)  # Columna para el historial
main_frame.rowconfigure(0, weight=1)     # Fila para la cámara y el historial
main_frame.rowconfigure(1, weight=0)     # Fila para los botones

# Frame para la cámara
video_frame = ttk.LabelFrame(main_frame, text="Cámara")
video_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

# Label para mostrar el video
video_label = ttk.Label(video_frame)
video_label.pack(expand=True)

# Frame para el historial
historial_frame = ttk.LabelFrame(main_frame, text="Historial de Letras")
historial_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

# Lista para mostrar el historial
historial_listbox = tk.Listbox(historial_frame)
historial_listbox.pack(fill='both', expand=True, padx=5, pady=5)

# Botón para eliminar el historial
clear_button = ttk.Button(historial_frame, text="Eliminar Historial", command=lambda: historial_listbox.delete(0, tk.END))
clear_button.pack(pady=5)

# Frame para los botones de la cámara
buttons_frame = ttk.Frame(main_frame)
buttons_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

# Variables para los botones
start_button = None
stop_button = None

def iniciar_camara():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        running = True
        threading.Thread(target=procesar_video, daemon=True).start()

def detener_camara():
    global cap, running
    running = False
    if cap is not None:
        cap.release()
    # Limpiar la imagen en video_label
    video_label.config(image='')

def procesar_video():
    global cap, running
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2) as hands:
        
        while running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Preprocesamiento de la imagen
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Procesamiento de la imagen
            image_height, image_width, _ = image.shape
            letra_detectada = None

            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks):
                    for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        # Obtener las coordenadas clave de los dedos
                        thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                                    int(hand_landmarks.landmark[4].y * image_height))
                        thumb_ip = (int(hand_landmarks.landmark[3].x * image_width),
                                    int(hand_landmarks.landmark[3].y * image_height))
                        thumb_mcp = (int(hand_landmarks.landmark[2].x * image_width),
                                    int(hand_landmarks.landmark[2].y * image_height))
                        thumb_cmc = (int(hand_landmarks.landmark[1].x * image_width),
                                    int(hand_landmarks.landmark[1].y * image_height))

                        index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                            int(hand_landmarks.landmark[8].y * image_height))
                        index_finger_dip = (int(hand_landmarks.landmark[7].x * image_width),
                                            int(hand_landmarks.landmark[7].y * image_height))
                        index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                            int(hand_landmarks.landmark[6].y * image_height))
                        index_finger_mcp = (int(hand_landmarks.landmark[5].x * image_width),
                                            int(hand_landmarks.landmark[5].y * image_height))

                        middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                            int(hand_landmarks.landmark[12].y * image_height))
                        middle_finger_dip = (int(hand_landmarks.landmark[11].x * image_width),
                                            int(hand_landmarks.landmark[11].y * image_height))
                        middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                            int(hand_landmarks.landmark[10].y * image_height))
                        middle_finger_mcp = (int(hand_landmarks.landmark[9].x * image_width),
                                            int(hand_landmarks.landmark[9].y * image_height))

                        ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                        int(hand_landmarks.landmark[16].y * image_height))
                        ring_finger_dip = (int(hand_landmarks.landmark[15].x * image_width),
                                        int(hand_landmarks.landmark[15].y * image_height))
                        ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                        int(hand_landmarks.landmark[14].y * image_height))
                        ring_finger_mcp = (int(hand_landmarks.landmark[13].x * image_width),
                                        int(hand_landmarks.landmark[13].y * image_height))

                        pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                                    int(hand_landmarks.landmark[20].y * image_height))
                        pinky_dip = (int(hand_landmarks.landmark[19].x * image_width),
                                    int(hand_landmarks.landmark[19].y * image_height))
                        pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                                    int(hand_landmarks.landmark[18].y * image_height))
                        pinky_mcp = (int(hand_landmarks.landmark[17].x * image_width),
                                    int(hand_landmarks.landmark[17].y * image_height))

                        wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))

                        # Aquí incluir tu lógica para detectar letras
                        # Por ejemplo, para la letra A:
                        if abs(thumb_tip[1] - index_finger_pip[1]) < 45 \
                            and abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and abs(thumb_tip[1] - ring_finger_pip[1]) < 30\
                            and abs(thumb_tip[1] - pinky_pip[1]) < 30:
                            letra_detectada = 'A'
                        # Añade aquí el resto de tus condiciones para detectar otras letras
                        # ...

                        # Si se detecta una letra, añadirla al historial
                        if letra_detectada:
                            historial_letras.append(letra_detectada)
                            historial_listbox.insert(tk.END, letra_detectada)

            # Convertir el frame a ImageTk
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Mostrar la imagen en el label
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            # Actualizar el tamaño de la ventana para que se ajuste al contenido
            video_label.update()

            # Salir si la ventana de Tkinter se cierra
            if not root.winfo_exists():
                break

        # Liberar la cámara al finalizar
        if cap is not None:
            cap.release()
            cap = None

# Botón para iniciar la cámara
start_button = ttk.Button(buttons_frame, text="Encender/Apagar Cámara", command=iniciar_camara)
start_button.pack(side="left", padx=5)

# Botón para detener la cámara
stop_button = ttk.Button(buttons_frame, text=" Cambiar Gestos/Abecedario", command=detener_camara)
stop_button.pack(side="left", padx=5)

# Botón para salir de la aplicación
exit_button = ttk.Button(buttons_frame, text="Salir", command=root.quit)
exit_button.pack(side="left", padx=5)

# Ejecutar la interfaz
root.mainloop()
