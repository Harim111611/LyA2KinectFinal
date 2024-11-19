import cv2
import mediapipe as mp

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)
cap.set(3,1920)
cap.set(4,1080)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands:  # max_num_hands en 2 para detectar ambas manos
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape
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
                            int(hand_landmarks.landmark[1].y * image_height))  # PUNTO 1: Base del pulgar

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
                        int(hand_landmarks.landmark[0].y * image_height))  # PUNTO 0: Muñeca

                                # Detectar letras
                # Aumentar el margen de flexibilidad para las posiciones de los dedos
                flexibilidad = 40
                
                                
                # Distancia de referencia entre el MCP y la punta del dedo índice (P5 y P8)
                distancia_referencia = distancia_euclidiana(index_finger_mcp, index_finger_tip)
                
                # Detectar letras
                # Letra A
                if abs(thumb_tip[1] - index_finger_pip[1]) < 45 \
                    and abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and abs(thumb_tip[1] - ring_finger_pip[1]) < 30\
                    and abs(thumb_tip[1] - pinky_pip[1]) < 30:
                    cv2.putText(image, 'A', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                # Letra B
                elif index_finger_pip[1] - index_finger_tip[1] > 0 and pinky_pip[1] - pinky_tip[1] > 0 and \
                middle_finger_pip[1] - middle_finger_tip[1] > 0 and ring_finger_pip[1] - ring_finger_tip[1] > 0 and \
                middle_finger_tip[1] - ring_finger_tip[1] < 0 and abs(thumb_tip[1] - ring_finger_mcp[1]) < 40:
                    cv2.putText(image, 'B', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                
                # Letra C  
                elif abs(index_finger_tip[1] - thumb_tip[1]) < 360 and \
                    index_finger_tip[1] - middle_finger_pip[1]<0 and index_finger_tip[1] - middle_finger_tip[1] < 0 and \
                        index_finger_tip[1] - index_finger_pip[1] > 0:
                   cv2.putText(image, 'C', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                # Letra D
                elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 \
                    and distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 \
                    and pinky_pip[1] - pinky_tip[1] < 0 \
                    and index_finger_pip[1] - index_finger_tip[1] > 0:
                    cv2.putText(image, 'D', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)    
                    
                 # Letra E
                elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
                    middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
                    and abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                    thumb_tip[1] - index_finger_tip[1] > 0 \
                    and thumb_tip[1] - middle_finger_tip[1] > 0 \
                    and thumb_tip[1] - ring_finger_tip[1] > 0 \
                    and thumb_tip[1] - pinky_tip[1] > 0:
                            cv2.putText(image, 'E', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                    # Letra F
                elif pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
                    ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 \
                    and abs(thumb_ip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) < 65:
                    cv2.putText(image, 'F', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)    
                 
                 # Letra G 
                elif thumb_tip[1] < thumb_mcp[1] and \
                  abs(index_finger_tip[1] - index_finger_pip[1]) < 20 and \
                  abs(index_finger_tip[0] - thumb_tip[0]) > 50 and \
                  middle_finger_tip[1] > middle_finger_mcp[1] and \
                  ring_finger_tip[1] > ring_finger_mcp[1] and \
                  pinky_tip[1] > pinky_mcp[1]:
                    cv2.putText(image, 'G', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)

                 
                  # Letra H
                elif abs(index_finger_tip[1] - index_finger_pip[1]) < 30 and \
                      abs(middle_finger_tip[1] - middle_finger_pip[1]) < 30 and \
                      ring_finger_tip[1] > ring_finger_pip[1] and \
                      pinky_tip[1] > pinky_pip[1] and \
                      thumb_tip[0] < index_finger_pip[0] and \
                      abs(index_finger_tip[0] - middle_finger_tip[0]) < 40:  # Índice y medio horizontales y juntos

                      cv2.putText(image, 'H', (700, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  3.0, (0, 0, 255), 6)
                 
                 
                 
                 # Letra I
                elif abs(pinky_tip[1] - pinky_mcp[1]) > 30 and \
                abs(index_finger_mcp[1] - middle_finger_mcp[1]) < 20 and \
                abs(middle_finger_mcp[1] - ring_finger_mcp[1]) < 20 and \
                index_finger_tip[1] > index_finger_mcp[1] and \
                thumb_tip[1] < ring_finger_tip[1] and thumb_tip[1] > ring_finger_pip[1] and \
                thumb_tip[0] < index_finger_mcp[0] + 15:

                    cv2.putText(image, 'I', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                # Letra J 
                elif abs(pinky_tip[1] - pinky_mcp[1]) > 30 and \
                abs(index_finger_mcp[1] - middle_finger_mcp[1]) < 20 and \
                abs(middle_finger_mcp[1] - ring_finger_mcp[1]) < 20 and \
                index_finger_tip[1] > index_finger_mcp[1] and \
                thumb_tip[1] < ring_finger_tip[1] and thumb_tip[1] > ring_finger_pip[1] and \
                abs(thumb_tip[0] - index_finger_mcp[0]) > 20:  # Ajuste para que el pulgar esté alineado horizontalmente

                    cv2.putText(image, 'J', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6) 
                

                # Letra K
                elif abs(index_finger_mcp[1] - index_finger_pip[1]) < distancia_referencia * 0.2 and \
                  abs(index_finger_pip[1] - index_finger_dip[1]) < distancia_referencia * 0.2 and \
                  abs(index_finger_dip[1] - index_finger_tip[1]) < distancia_referencia * 0.2 and \
                  middle_finger_tip[1] > middle_finger_pip[1] and \
                  middle_finger_pip[1] > middle_finger_mcp[1] and \
                  abs(middle_finger_mcp[0] - middle_finger_tip[0]) < distancia_referencia * 0.4:  # Curvatura del dedo medio
                    
                    cv2.putText(image, 'K', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6) 
                
                
                # Letra L
                elif abs(index_finger_mcp[1] - index_finger_pip[1]) > 40 and \
                  abs(index_finger_pip[1] - index_finger_dip[1]) > 40 and \
                  abs(index_finger_dip[1] - index_finger_tip[1]) > 40 and \
                  abs(thumb_cmc[0] - thumb_tip[0]) > 40 and \
                  abs(thumb_mcp[0] - thumb_tip[0]) > 40 and \
                  abs(thumb_tip[1] - index_finger_mcp[1]) < 50:  # Asegurarse de que el pulgar esté horizontal y cerca del MCP del índice

                    cv2.putText(image, 'L', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                
                 # Letra M
                elif ring_finger_tip[1] > ring_finger_mcp[1] + 20 and \
                  middle_finger_tip[1] > middle_finger_mcp[1] + 20 and \
                  index_finger_tip[1] > index_finger_mcp[1] + 20 and \
                  thumb_tip[1] < index_finger_pip[1] - 20 and \
                  thumb_tip[1] < middle_finger_pip[1] - 20 and \
                  thumb_tip[1] < ring_finger_pip[1] - 20:  # El pulgar debe cubrir las puntas de los dedos índice, medio y anular con más flexibilidad
                    
                    cv2.putText(image, 'M', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                    
                # Letra N
                elif ring_finger_tip[1] > ring_finger_mcp[1] + 30 and \
                  middle_finger_tip[1] > middle_finger_mcp[1] + 30 and \
                  index_finger_tip[1] > index_finger_mcp[1] + 30 and \
                  thumb_tip[1] < index_finger_pip[1] - 30 and \
                  thumb_tip[1] < middle_finger_pip[1] - 30 and \
                  thumb_tip[1] < ring_finger_pip[1] - 30:  # El pulgar debe cubrir las puntas de los dedos índice, medio y anular con mayor flexibilidad
                    
                    cv2.putText(image, 'N', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                
                
                # Letra O
                elif abs(index_finger_tip[0] - thumb_tip[0]) < 50 and \
                  abs(index_finger_tip[1] - thumb_tip[1]) < 50 and \
                  abs(middle_finger_tip[0] - index_finger_tip[0]) < 60 and \
                  abs(ring_finger_tip[0] - middle_finger_tip[0]) < 60 and \
                  abs(pinky_tip[0] - ring_finger_tip[0]) < 60:  # Los dedos deben estar curvados formando un círculo
                    
                    cv2.putText(image, 'O', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                    
                # Letra P Modificar
                elif abs(index_finger_tip[1] - index_finger_mcp[1]) > 40 and \
                  abs(middle_finger_tip[1] - middle_finger_mcp[1]) > 40 and \
                  ring_finger_tip[1] < ring_finger_mcp[1] and \
                  pinky_tip[1] < pinky_mcp[1] and \
                  thumb_tip[1] < index_finger_mcp[1]:  # Pulgar doblado, índice y medio extendidos hacia arriba
                    
                    cv2.putText(image, 'P', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)    
                    
                    
                # Letra R
                elif abs(index_finger_tip[1] - index_finger_mcp[1]) > 40 and \
                    abs(middle_finger_tip[1] - middle_finger_mcp[1]) > 40 and \
                    abs(index_finger_tip[0] - middle_finger_tip[0]) < 20 and \
                    ring_finger_tip[1] > ring_finger_mcp[1] + 20 and \
                    pinky_tip[1] > pinky_mcp[1] + 20 and \
                    thumb_tip[1] > index_finger_mcp[1]: 
                      
                      cv2.putText(image, 'R', (700, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        3.0, (0, 0, 255), 6)    
                      
                      
                # Letra S
                elif index_finger_tip[1] > index_finger_mcp[1] + 20 and \
                  middle_finger_tip[1] > middle_finger_mcp[1] + 20 and \
                  ring_finger_tip[1] > ring_finger_mcp[1] + 20 and \
                  pinky_tip[1] > pinky_mcp[1] + 20 and \
                  thumb_tip[1] > index_finger_pip[1] and \
                  thumb_tip[0] > index_finger_mcp[0]:  # El pulgar está sobre los dedos

                    cv2.putText(image, 'S', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)     
                      
                      
                # Letra U
                elif index_finger_tip[1] < index_finger_pip[1] and \
                    middle_finger_tip[1] < middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1] and \
                    abs(index_finger_tip[0] - middle_finger_tip[0]) < 40:  # Aumentamos la tolerancia a 40 píxeles para mayor flexibilidad

                    cv2.putText(image, 'U', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)      
                      
                # Letra V
                elif index_finger_tip[1] < index_finger_pip[1] and \
                    middle_finger_tip[1] < middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1] and \
                    abs(index_finger_tip[0] - middle_finger_tip[0]) > 40:  # Índice y medio separados

                    cv2.putText(image, 'V', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)     
                      
                # Letra W
                elif index_finger_tip[1] < index_finger_pip[1] and \
                    middle_finger_tip[1] < middle_finger_pip[1] and \
                    ring_finger_tip[1] < ring_finger_pip[1] and \
                    pinky_tip[1] > pinky_pip[1] and \
                    abs(index_finger_tip[0] - middle_finger_tip[0]) > 40 and \
                    abs(middle_finger_tip[0] - ring_finger_tip[0]) > 40:  # Medio y anular separados

                    cv2.putText(image, 'W', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)      
                # Letra X  
                elif index_finger_tip[1] > index_finger_pip[1] and \
                      middle_finger_tip[1] > middle_finger_pip[1] and \
                      ring_finger_tip[1] > ring_finger_pip[1] and \
                      pinky_tip[1] > pinky_pip[1] and \
                      thumb_tip[0] < index_finger_mcp[0]:

                      cv2.putText(image, 'X', (700, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  3.0, (0, 0, 255), 6)
                
                # Letra Y
                elif index_finger_tip[1] > index_finger_pip[1] and \
                    middle_finger_tip[1] > middle_finger_pip[1] and \
                    ring_finger_tip[1] > ring_finger_pip[1] and \
                    pinky_tip[1] < pinky_pip[1] and \
                    thumb_tip[1] < thumb_ip[1]:  # El pulgar y el meñique están hacia arriba

                    cv2.putText(image, 'Y', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                # Letra Z
                elif index_finger_tip[1] > index_finger_pip[1] and \
                      middle_finger_tip[1] > middle_finger_pip[1] and \
                      ring_finger_tip[1] > ring_finger_pip[1] and \
                      pinky_tip[1] < pinky_pip[1] and \
                      thumb_tip[1] > thumb_ip[1]:  # El meñique está hacia arriba y el pulgar no cuenta

                      cv2.putText(image, 'Z', (700, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  3.0, (0, 0, 255), 6)
                # Letra Q
                elif index_finger_tip[1] < index_finger_pip[1] and \
                      middle_finger_tip[1] > middle_finger_pip[1] and \
                      thumb_tip[1] < index_finger_pip[1] and \
                      ring_finger_tip[1] > ring_finger_pip[1] and \
                      pinky_tip[1] > pinky_pip[1]:

                      cv2.putText(image, 'Q', (700, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  3.0, (0, 0, 255), 6)
                # Letra T    
                elif thumb_tip[0] > index_finger_mcp[0] and \
                    thumb_tip[1] < index_finger_pip[1] and \
                    abs(thumb_tip[1] - index_finger_pip[1]) < 30 and \
                    index_finger_tip[1] > index_finger_mcp[1] and \
                    middle_finger_tip[1] > middle_finger_mcp[1] and \
                    ring_finger_tip[1] > ring_finger_mcp[1] and \
                    pinky_tip[1] > pinky_mcp[1]:  # Los demás dedos deben estar cerrados

                    cv2.putText(image, 'T', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
  
                
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
