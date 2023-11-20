import cv2


#Parametros de configuración
# Color máximo de celula (Busca los colores más blancos)
max_color_for_cell = 190
# Tamaño máximo de celula máximo. Una figura que tenga un área mayor a este valor no será considerada como celula
max_size_for_cell = 400
# Tamaño mínimo de celula. Una figura que tenga un área menor a este valor no será considerada como celula
min_size_for_cell = 35
# Video a analizar
video = cv2.VideoCapture('celulas_2.mp4')

if __name__ == '__main__':

    # Inicialización de variables
    # Contador de celulas
    initialCells = 0
    # Bandera para saber si es el primer frame
    firstFrame = True

    # Ciclo para recorrer el video
    while video.isOpened():
        # Lectura de video
        ret, frame = video.read()
        # Si no hay video, se termina el ciclo
        if not ret:
            break
        # Contador de celulas
        cells = 0

        # Conversión de imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, max_color_for_cell, 255, cv2.THRESH_BINARY)

        # Detección de contornos
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # Ciclo para recorrer los contornos
        for cnt in contours:
            # Calcula el área de cada contorno
            area = cv2.contourArea(cnt)
            # Si el área está entre los valores de tamaño mínimo y máximo, se considera como celula
            if min_size_for_cell <= area <= max_size_for_cell:
                # Se cuenta una célula más
                cells += 1
                # Se determina la posición y tamaño de la celula
                x, y, w, h = cv2.boundingRect(cnt)
                # Se dibuja el contorno de la celula
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                # Se agrega el número de celula
                cv2.putText(frame, '1', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 2)

        #Se agrega el conteo total de celulas en la esquina superior izquierda
        cv2.putText(frame, 'Celulas: ' + str(cells), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Se guarda el número de celulas iniciales
        if firstFrame:
            firstFrame = False
            initialCells = cells

        # Muestra el procesamiento en contraste
        # cv2.imshow('Object Tracking', thresholded)

        # Muestra el video original con el procesamiento
        cv2.imshow('Object Tracking', frame)


        # Si se presiona la tecla 'q', se termina el ciclo
        if cv2.waitKey(35) & 0xFF == ord('q'):
            break

    # Se liberan los recursos
    video.release()
    # Se cierran las ventanas
    cv2.destroyAllWindows()

    # Se muestran los resultados
    print('Celulas iniciales: ' + str(initialCells))
    print('Total de celulas: ' + str(cells))
    print('Celulas nuevas: ' + str(cells - initialCells))