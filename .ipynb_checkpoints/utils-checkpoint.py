import os
import sqlite3
import numpy as np
from PIL import Image
from glob import glob
import tensorflow as tf
# from deepface import DeepFace
from google.cloud import vision
from sklearn.cluster import KMeans

# Function to extract the dominant RGB color from an image
def color_extractor(image_path, k=4):
    # Open and resize the image using PIL
    # image = image_path.resize((64, 64))
    
    # Convert the image to an array
    image_array = np.array(image_path)
    
    # Reshape the array to be a list of pixels
    image_array = image_array.reshape((image_array.shape[0] * image_array.shape[1], 3))
    
    # Apply K-means clustering algorithm
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image_array)
    label_counts = np.bincount(labels)
    
    # Identify the most popular cluster and get its central color
    dominant_color = clt.cluster_centers_[np.argmax(label_counts)]
    
    return dominant_color.astype(int)  # Return the dominant color

# Funcion que recibe como argumento el path de la imagen que queremos analizar
def agepredictor_(image_name):
    agep = DeepFace.analyze(img_path = image_name, 
        actions = 'age')
    jsonage = agep[0]
    age = jsonage['age']
    x = jsonage['region']['x']
    y = jsonage['region']['y']
    w = jsonage['region']['w']
    h = jsonage['region']['h']
    return age,(x,y,w,h)

# Dummy function, replace it with your age prediction model
# Dummy function to simulate DeepFace.analyze
def deepface_analyze(image_np):
    return {"age": 25}

# Function to preprocess image with Pillow and detect face & age with Mediapipe
def agepredictor_(image_name):
    # Open image using PIL
    image_pil = Image.open(image_name).convert('RGB')
    
    # Convert PIL Image to numpy array
    image_np = np.array(image_pil)
    
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Perform face detection
    results = face_detection.process(image_np)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image_np.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Crop the face area
            face_image = image_np[y:y+h, x:x+w]
            
            # Predict age (Replace predict_age with your actual age prediction model)
            age = predict_age(face_image)
            
            return age, (x, y, w, h)

# Function to preprocess image and detect face & age
def agepredictor_(image_name):
    # Open image using PIL
    image_pil = Image.open(image_name)
    
    # Convert PIL Image to numpy array
    image_np = np.array(image_pil)
    
    # Initialize MTCNN
    detector = MTCNN()
    
    # Detect faces
    results = detector.detect_faces(image_np)
    
    if results:
        # Assuming the first detected face is the one you're interested in
        x, y, w, h = results[0]['box']
        
        # Crop the face area
        face_image = image_np[y:y+h, x:x+w]
        
        # Predict age (In your case, it could be a DeepFace call)
        age = deepface_analyze(face_image)['age']
        
        return age, (x, y, w, h)

# Function to preprocess image and detect face & age
def agepredictor(image_name):
    # Initialize Google Vision client
    client = vision.ImageAnnotatorClient.from_service_account_json("credentials.json") ### file from gcloud account project
    # client = vision.ImageAnnotatorClient()

    # Open image using PIL
    image_pil = Image.open(image_name)

    # Convert PIL Image to numpy array
    image_np = np.array(image_pil)

    # Load image to Google Vision API
    with open(image_name, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Detect faces
    response = client.face_detection(image=image)
    face_annotations = response.face_annotations

    if face_annotations:
        # Assuming the first detected face is the one you're interested in
        x, y, w, h = (face_annotations[0].bounding_poly.vertices[0].x, face_annotations[0].bounding_poly.vertices[0].y,
                      face_annotations[0].bounding_poly.vertices[2].x - face_annotations[0].bounding_poly.vertices[0].x,
                      face_annotations[0].bounding_poly.vertices[2].y - face_annotations[0].bounding_poly.vertices[0].y)

        # Crop the face area
        face_image = image_np[y:y + h, x:x + w]

        # Predict age (In your case, it could be a DeepFace call)
        age = deepface_analyze(face_image)['age']

        return age, (x, y, w, h)

# Función que devuelve la edad y el color RGB dominante del rostro
def extract_dominant_color(image_name):
    # Obtiene la edad y coordenadas del rostro usando la función 'agepredictor'
    age, coords = agepredictor(image_name)
    
    # Read the image and convert it into an array
    image = Image.open(image_name).convert('RGB')
    rgb_array = np.array(image)
    
    # Extrae las coordenadas del rostro de la imagen
    x, y, w, h = coords
    
    # Obtiene el área de la imagen que corresponde al rostro detectado
    face_region = rgb_array[y:y+h, x:x+w]
    
    # Extrae el color RGB dominante de esa región del rostro usando la función 'color_extractor'
    RGB = color_extractor(face_region)
    
    # Devuelve la edad y el color RGB dominante
    return age, RGB

# Función que clasifica el tono de piel basado en un vector RGB
def classify_skin_tone(rgb_vector):
    # Diccionario con tonos de piel y sus valores RGB representativos
    COLOR_RANGES = {
        "Muy claro / pálido": [245, 214.5, 184.5],
        "Claro / caucásico": [219, 193.5, 171.5],
        "Claro-medio / moreno claro": [184.5, 166.5, 149],
        "Medio / moreno": [151, 135.5, 123],
        "Medio-oscuro / moreno oscuro": [121, 104.5, 93.5],
        "Oscuro": [57.5, 45.5, 39.5]
    }
    
    # Inicializamos la menor distancia con un valor infinitamente grande
    min_distance = float('inf')
    closest_skin_tone = None
    
    # Iteramos sobre los tonos de piel predefinidos
    for skin_tone, central_rgb in COLOR_RANGES.items():
        # Calculamos la distancia euclidiana entre el vector RGB y el valor RGB central del tono de piel actual
        distance = np.linalg.norm(np.array(rgb_vector) - np.array(central_rgb))
        
        # Si la distancia es menor que la más pequeña registrada hasta el momento, la actualizamos
        if distance < min_distance:
            min_distance = distance
            closest_skin_tone = skin_tone
    
    # Devolvemos el tono de piel más cercano al vector RGB ingresado
    return closest_skin_tone

### Funcion que classifica la edad en un rango determinado.
def classify_age_range(age):
    if age < 18:return "menores de 18"
    elif 18 <= age <= 25:return "18-25"
    elif 26 <= age <= 35:return "26-35"
    elif 36 <= age <= 45:return "36-45"
    elif 46 <= age <= 55:return "46-55"
    else:return "mayores de 55"

### Función que devuelve el rango de edad y el tono de piel de la persona en la imagen
def age_skin_tone(image_name):
    # Extrae la edad y el color RGB dominante del rostro en la imagen
    age, rgb_input = extract_dominant_color(image_name)
    
    # Clasifica el vector RGB para determinar el tono de piel
    skin_tone = classify_skin_tone(rgb_input)
    
    # Clasifica la edad en un rango determinado
    age_range = classify_age_range(age)
    
    # Devuelve el rango de edad y el tono de piel
    return age_range, skin_tone

### Funcion que predice la forma del rostro y la probabilidad
def predict_face_shape(file_path, model, img_size1=100, img_size2=120):
    
    # Procesa la imagen para prepararla para la predicción
    img = Image.open(file_path)
    img = img.resize((img_size1, img_size2))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normaliza la imagen al rango [0,1]
    
    # Realiza predicciones sobre la forma del rostro
    predictions = model.predict(img_array)
    
    # Encuentra el índice de la categoría con la probabilidad más alta
    i_max = tf.argmax(predictions[0])
    prob = predictions[0][i_max]
    category_index = tf.argmax(predictions[0]).numpy()
    
    # Define las categorías de formas de rostro
    category_names = ['CORAZON', 'ALARGADA', 'OVALADA', 'REDONDA', 'CUADRADA']

    # Devuelve el nombre de la forma del rostro y su probabilidad asociada
    return category_names[category_index], prob

def obtener_recomendaciones_db(rango_edad, tono_piel, forma_cara, prob, filename="recommendations.db"):
    # Connecting to the database
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Fetching recommendations from the database
    cursor.execute('SELECT recomendacion, link FROM edad_recomendaciones WHERE rango=?', (rango_edad,))
    recomendacion_edad_data = cursor.fetchone()
    recomendacion_edad = f"Basado en tu edad ({rango_edad}):\n{recomendacion_edad_data[0]}\n\n"
    link = recomendacion_edad_data[1]

    cursor.execute('SELECT recomendacion FROM tono_recomendaciones WHERE tono=?', (tono_piel,))
    recomendacion_tono_data = cursor.fetchone()
    recomendacion_tono = f"Basado en tu tono de piel ({tono_piel}):\n{recomendacion_tono_data[0]}\n\n"

    cursor.execute('SELECT recomendacion FROM forma_cara_recomendaciones WHERE forma=?', (forma_cara,))
    recomendacion_cara_data = cursor.fetchone()
    recomendacion_cara = f"Basado en tu forma de cara ({forma_cara}):\n{recomendacion_cara_data[0]}"

    # Closing the connection
    conn.close()

    return recomendacion_edad, recomendacion_tono, recomendacion_cara, link

