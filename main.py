import cv2
import mediapipe as mp
import numpy as np
import pygame

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializar Pygame para sonidos
pygame.init()

# Frecuencias de las notas (C4, D4, E4, F4, G4, A4, B4)
frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]

# Duración del sonido (en segundos)
duration = 0.2

# Tasa de muestreo (sample rate)
sample_rate = 44000

# Función para generar un tono sinusoidal
def generate_tone(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    tone = np.int16(tone * 32767)  # Convertir a formato de 16 bits
    stereo_tone = np.column_stack((tone, tone))  # Convertir a estéreo
    return pygame.sndarray.make_sound(stereo_tone)

# Crear sonidos para cada nota
sounds = {
    i: generate_tone(frequencies[i], duration, sample_rate) for i in range(7)
}

# Configurar cámara
cap = cv2.VideoCapture(0)

# Definir teclas virtuales
keys = [(50 + i * 100, 400, 100, 200) for i in range(7)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Dibujar teclas
    for i, (x, y, w, h) in enumerate(keys):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.putText(frame, chr(67 + i), (x + 30, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.polylines(frame, [np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])], True, (0, 0, 0), 2);

    # Procesar detección de manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for tip in [4, 8, 12, 16, 20]:
                # Obtener coordenadas del índice (landmark 8)
                finger_tip = hand_landmarks.landmark[tip]
                h, w, _ = frame.shape
                x, y = int(finger_tip.x * w), int(finger_tip.y * h)
                
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                
                # Verificar si toca una tecla
                for i, (kx, ky, kw, kh) in enumerate(keys):
                    if kx < x < kx + kw and ky < y < ky + kh:
                        sounds[i].play()
    
    cv2.imshow("Virtual Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()