import tkinter as tk
from tkinter import Label, Button, Text, Scrollbar, Frame
import cv2
import numpy as np
from PIL import Image, ImageTk
import copy
import mediapipe as mp
from collections import deque
import time


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Lenguaje de Señas")
        self.root.geometry("1300x660")  # Adjust the size according to your needs

        self.running = False
        self.cap = None

        # Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.last_added_time = 0  # Time of the last history update

        # Layout improvements
        self.main_frame = Frame(root, bg="#2C3E50", padx=10, pady=10)  # Add padding for spacing
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Camera section with a dark background
        self.camera_frame = Frame(self.main_frame, bg="#2C3E50", width=300, height=40, bd=10, relief="solid")
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Ensure the frame expands to fill the space available
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.camera_label = Label(self.camera_frame, bg="#000000")
        self.camera_label.grid(row=0, column=0, padx=0, pady=0, sticky='nsew')

        # Chat section with a dark background
        self.chat_frame = Frame(self.main_frame, bg="#2C3E50", width=300, bd=10, relief="solid", height=480)
        self.chat_frame.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

        self.chat_scrollbar = Scrollbar(self.chat_frame)
        self.chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_text = Text(self.chat_frame, yscrollcommand=self.chat_scrollbar.set, state=tk.DISABLED, wrap=tk.WORD,
                              font=("Arial", 12), bg="#34495E", fg="white", width=40, height=20)  # Adjust width and height
        self.chat_text.pack(padx=10, pady=10)

        self.chat_scrollbar.config(command=self.chat_text.yview)

        # Control frame with a dark background
        self.control_frame = Frame(root, bg="#2C3E50", pady=10)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Using grid layout for better organization
        self.start_button = Button(self.control_frame, text="Encender Cámara", command=self.start_camera,
                                   font=("Arial", 12), bg="#27AE60", fg="white", relief="flat", width=20)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = Button(self.control_frame, text="Apagar Cámara", command=self.stop_camera,
                                  font=("Arial", 12), bg="#E74C3C", fg="white", relief="flat", width=20)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.clear_button = Button(self.control_frame, text="Eliminar Historial", command=self.clear_chat,
                                   font=("Arial", 12), bg="#F39C12", fg="white", relief="flat", width=20)
        self.clear_button.grid(row=0, column=2, padx=5)

    def start_camera(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550)
            self.update_frame()

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.camera_label.config(image="")

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                debug_image = copy.deepcopy(frame)

                # Detección con Mediapipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                detected_letter = None

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Calcular puntos de referencia
                        landmark_list = self.calc_landmark_list(frame, hand_landmarks)
                        # Dibujar esqueleto
                        debug_image = self.draw_landmarks(debug_image, landmark_list)
                        # Detectar letra
                        detected_letter = self.detect_letter(landmark_list)

                if detected_letter:
                    self.update_chat_with_delay(detected_letter)

                # Mostrar en Tkinter
                img = Image.fromarray(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

            self.camera_label.after(10, self.update_frame)

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for lm in landmarks.landmark:
            x = min(int(lm.x * image_width), image_width - 1)
            y = min(int(lm.y * image_height), image_height - 1)
            landmark_point.append([x, y])
        return landmark_point

    def detect_letter(self, landmark_point):
        def distancia_euclidiana(p1, p2):
            d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            return d
        if len(landmark_point) >= 21:
            thumb_mcp = landmark_point[2]
            thumb_ip = landmark_point[3]
            thumb_tip = landmark_point[4]
            index_finger_tip = landmark_point[8]
            middle_finger_mcp = landmark_point[9]
            index_finger_pip = landmark_point[6]
            middle_finger_tip = landmark_point[12]
            middle_finger_pip = landmark_point[10]
            ring_finger_tip = landmark_point[16]
            ring_finger_pip = landmark_point[14]
            pinky_tip = landmark_point[20]
            pinky_pip = landmark_point[18]
            ring_finger_mcp = landmark_point[13]  # MCP del dedo anular
            pinky_mcp = landmark_point[17]

            # Letra A (ya implementada)
            if abs(thumb_tip[1] - index_finger_pip[1]) < 45 \
                    and abs(thumb_tip[1] - middle_finger_pip[1]) < 30 \
                    and abs(thumb_tip[1] - ring_finger_pip[1]) < 30 \
                    and abs(thumb_tip[1] - pinky_pip[1]) < 30:
                return "A"

            # Letra B (implementada según los criterios que mencionaste)
            elif index_finger_pip[1] - index_finger_tip[1] > 0 and pinky_pip[1] - pinky_tip[1] > 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] > 0 and ring_finger_pip[1] - ring_finger_tip[1] > 0 and \
                    middle_finger_tip[1] - ring_finger_tip[1] < 0 and abs(thumb_tip[1] - ring_finger_mcp[1]) < 40:
                return "B"
            # Letra C (implementada según los criterios que mencionaste)
            elif abs(index_finger_tip[1] - thumb_tip[1]) < 360 and \
                     index_finger_tip[1] - middle_finger_pip[1] < 0 and index_finger_tip[1] - middle_finger_tip[1] < 0 and \
                     index_finger_tip[1] - index_finger_pip[1] > 0:
                return "C"
                # Letra D (implementada según los criterios proporcionados)
            elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 and \
                 distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 and \
                 pinky_pip[1] - pinky_tip[1] < 0 < index_finger_pip[1] - index_finger_tip[1]:
                return "D"
                # Letra E (implementada según los criterios proporcionados)
            elif index_finger_pip[1] - index_finger_tip[1] < 0 and \
                 pinky_pip[1] - pinky_tip[1] < 0 and \
                 middle_finger_pip[1] - middle_finger_tip[1] < 0 and \
                 ring_finger_pip[1] - ring_finger_tip[1] < 0 and \
                 abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                 thumb_tip[1] - index_finger_tip[1] > 0 and \
                 thumb_tip[1] - middle_finger_tip[1] > 0 and \
                 thumb_tip[1] - ring_finger_tip[1] > 0 and \
                 thumb_tip[1] - pinky_tip[1] > 0:
                return "E"
            elif pinky_pip[1] - pinky_tip[1] > 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
                    ring_finger_pip[1] - ring_finger_tip[1] > 0 and \
                    index_finger_pip[1] - index_finger_tip[1] < 0 and \
                    abs(thumb_ip[1] - thumb_tip[1]) > 0 and \
                    distancia_euclidiana(index_finger_tip, thumb_tip) < 65:
                return "F"
            elif thumb_tip[1] < thumb_mcp[1] and \
                    abs(index_finger_tip[1] - index_finger_pip[1]) < 20 and \
                    abs(index_finger_tip[0] - thumb_tip[0]) > 50 and \
                    middle_finger_tip[1] > middle_finger_mcp[1] and \
                    ring_finger_tip[1] > ring_finger_mcp[1] and \
                    pinky_tip[1] > pinky_mcp[1]:
                return "G"

        return None

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Dibujar líneas del esqueleto
            for i, j in [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
                (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
                (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
                (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
                (0, 17), (17, 18), (18, 19), (19, 20)  # Meñique
            ]:
                cv2.line(image, tuple(landmark_point[i]), tuple(landmark_point[j]), (0, 0, 255), 2)

            # Dibujar puntos clave
            for index, landmark in enumerate(landmark_point):
                radius = 5 if index not in [4, 8, 12, 16, 20] else 8
                cv2.circle(image, tuple(landmark), radius, (255, 255, 255), -1)
                cv2.circle(image, tuple(landmark), radius, (0, 0, 0), 1)
        return image

    def update_chat_with_delay(self, message):
        current_time = time.time()
        if current_time - self.last_added_time >= 1:  # 1 segundos de intervalo
            self.add_to_chat(message)
            self.last_added_time = current_time

    def add_to_chat(self, message):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"Letra detectada: {message}\n")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def clear_chat(self):
        self.chat_text.config(state=tk.NORMAL)  # Habilitar la edición del widget Text
        self.chat_text.delete(1.0, tk.END)  # Eliminar todo el contenido del Text
        self.chat_text.config(state=tk.DISABLED)  # Deshabilitar la edición del widget Text

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


# Iniciar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
