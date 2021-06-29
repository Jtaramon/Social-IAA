# imports
import cv2
import numpy as np
import time
import argparse

# own modules
import utills, plot

confid = 0.5
thresh = 0.5
mouse_pts = []

# Función para obtener puntos por Región de interés (ROI) y escala de distancia. Se necesitarán 8 puntos en el primer fotograma con un clic del mouse
# event. Los primeros cuatro puntos definirán el ROI donde queremos monitorear el distanciamiento social. Además, estos puntos deben formar paralelos
# líneas en el mundo real si se ven desde arriba (vista de pájaro). Los siguientes 3 puntos definirán una distancia de 6 pies (unidad de longitud) en
# dirección horizontal y vertical y deben formar líneas paralelas con ROI. Unidad de longitud que podemos tomar en función de nuestra elección.
# Los puntos deben aparecer en un orden predefinido: abajo a la izquierda, abajo a la derecha, arriba a la derecha, arriba a la izquierda, los puntos 5 y 6 deben formarse
# La línea horizontal y los puntos 5 y 7 deben formar una línea vertical. La escala horizontal y vertical será diferente.

# Se llamará a la función en los eventos del mouse

def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        # print ("Punto detectado")
        # print (mouse_pts)
        


def calculate_social_distancing(vid_path, net, output_dir, output_vid, ln1):
    
    count = 0
    vs = cv2.VideoCapture(vid_path)    

    # Obtenga la altura, el ancho y los fps del video
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    # Establecer escala para vista de pájaro
    # La vista de pájaro solo mostrará el ROI
    scale_w, scale_h = utills.get_scale(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
        
    points = []
    global image
    
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            print('here')
            break
            
        (H, W) = frame.shape[:2]
        
        # El primer cuadro se utilizará para dibujar el ROI y la distancia horizontal y vertical de 180 cm (longitud de la unidad en ambas direcciones)
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("image")
                    break
               
            points = mouse_pts

        # Usando los primeros 4 puntos o coordenadas para la transformación de perspectiva. La región marcada por estos 4 puntos son
        # considerado ROI. Este ROI en forma de polígono se deforma en un rectángulo que se convierte en la vista de pájaro.
        # Esta vista de pájaro tiene entonces la propiedad de que los puntos se distribuyen uniformemente horizontalmente y
        # verticalmente (la escala para la dirección horizontal y vertical será diferente). Entonces, para los puntos de vista de pájaro son
        # distribuidos por igual, lo que no era el caso de la vista normal.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # usando los siguientes 3 puntos para la longitud de la unidad horizontal y vertical (en este caso, 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # dado que la vista de pájaro tiene la propiedad de que todos los puntos son equidistantes en dirección horizontal y vertical.
        # distance_w y distance_h nos darán una distancia de 180 cm tanto en dirección horizontal como vertical
        # (cuántos píxeles habrá en 180 cm de longitud en dirección horizontal y vertical a vista de pájaro),
        # que podemos usar para calcular la distancia entre dos humanos en vista transformada o vista de pájaro
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
    ####################################################################################
    
        # YOLO v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detección de humanos en el marco
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue

        # Aquí usaremos el punto central inferior del cuadro delimitador para todos los cuadros y transformaremos todos esos
        # puntos centrales inferiores a vista de pájaro
        person_points = utills.get_transformed_points(boxes1, prespective_transform)
        
        # Aquí calcularemos la distancia entre puntos transformados (humanos)
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)
    
        frame1 = np.copy(frame)
        
        # Dibuje una vista de pájaro y un marco con cuadros delimitadores alrededor de los humanos de acuerdo con el factor de riesgo
        bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
        
        # Mostrar / escribir imágenes y videos
        if count != 0:
            output_movie.write(img)
            bird_movie.write(bird_image)
    
            cv2.imshow('Bird Eye View', bird_image)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)
    
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    # Argumentos recibidos especificados por el usuario
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/video.webm' ,
                    help='Path for input video')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')
    
    parser.add_argument('-O', '--output_vid', action='store', dest='output_vid', default='./output_vid/' ,
                    help='Path for Output videos')

    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                    help='Path for models directory')
                    
    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                    help='Use open pose or not (YES/NO)')
                    
    values = parser.parse_args()
    
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
        
    output_dir = values.output_dir
    if output_dir[len(output_dir) - 1] != '/':
        output_dir = output_dir + '/'
    
    output_vid = values.output_vid
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'


    # cargar pesas Yolov3
    
    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # establecer devolución de llamada del mouse

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)
    
    calculate_social_distancing(values.video_path, net_yl, output_dir, output_vid, ln1)



