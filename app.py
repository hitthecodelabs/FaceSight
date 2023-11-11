import os
import sqlite3
import tensorflow as tf

from PIL import Image
from utils import *
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from flask import redirect, url_for, flash, send_from_directory

# Crear una instancia de la aplicación Flask
app = Flask(__name__, static_url_path='/tmp', static_folder='tmp')

# Clave secreta para la aplicación (cambia "clave_secreta" por tu clave real)
app.secret_key = "clave_secreta"

# Carpeta donde se cargarán los archivos
UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cargar el modelo de red neuronal desde un archivo previamente entrenado
model = tf.keras.models.load_model("model.h5")

# Configurar la carpeta de carga de archivos en la aplicación
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Función para verificar si la extensión de un archivo es permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Definir la ruta para la URL raíz y especificar los métodos que puede manejar
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Verificar si el método de la solicitud es POST (envío de formulario)
    if request.method == 'POST':
        # Verificar si no se proporcionó ningún archivo en la solicitud
        if 'file' not in request.files:
            flash('No se proporcionó archivo')
            return redirect(request.url)

        # Obtener el archivo enviado en la solicitud
        file = request.files['file']

        # Verificar si no se seleccionó ningún archivo
        if file.filename == '':
            flash('No se seleccionó ningún archivo')
            return redirect(request.url)

        # Verificar si el archivo tiene una extensión permitida
        if file and allowed_file(file.filename):
            # Generar un nombre de archivo seguro
            filename = secure_filename(file.filename)

            # Construir la ruta completa donde se guardará el archivo
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Guardar el archivo en la ruta especificada
            file.save(filepath)
            print(filepath)

            try:
                # Obtener edad y tono de piel utilizando una función (age_skin_tone) definida en otro lugar
                edad, tono = age_skin_tone(filepath)

                # Estimar la forma de la cara utilizando una función (predict_face_shape) y el modelo cargado
                forma, prob = predict_face_shape(filepath, model)

                # Obtener recomendaciones basadas en la edad, tono de piel y forma de la cara
                recomendacion_edad, recomendacion_tono, recomendacion_cara, link = obtener_recomendaciones_db(edad, tono, forma, prob)
                
                # Crear una representación de la forma de la cara con probabilidad
                forma2 = f"{forma} (Probabilidad: {round(prob*100, 2)}%)"

                # Devolver los resultados a una plantilla HTML (results.html)
                return render_template('results.html', edad=edad, tono=tono, 
                                       forma=forma2, link=link, filepath=filepath,
                                       recomendacion_edad=recomendacion_edad,
                                       recomendacion_tono=recomendacion_tono,
                                       recomendacion_cara=recomendacion_cara)
            except ValueError as e:
                # Manejar casos de error al procesar la imagen
                if "Face could not be detected" in str(e):
                    error_message = "No se pudo detectar una cara en la imagen. Por favor, confirme que la imagen es una foto de un rostro."
                    return render_template('error.html', error_message=error_message)
                else:
                    error_message = "Hubo un error al procesar la imagen."
                    return render_template('error.html', error_message=error_message)

    # Si el método de la solicitud es GET (carga inicial de la página), mostrar la plantilla de carga (upload.html)
    return render_template('upload.html')

# Definir la ruta para procesar la imagen, solo aceptando solicitudes POST
@app.route('/process_image', methods=['POST'])
def process_image():
    # Verificar si la solicitud POST incluye la parte del archivo
    if 'file' not in request.files:
        return 'No se proporcionó archivo'
    
    # Obtener el archivo de la solicitud
    file = request.files['file']
    
    # Si el usuario no selecciona un archivo, el navegador envía una parte vacía sin nombre de archivo
    if file.filename == '':
        return 'No se seleccionó ningún archivo'
    
    # Verificar si el archivo tiene una extensión permitida
    if file and allowed_file(file.filename):
        # Agregar la marca de tiempo al nombre del archivo para hacerlo único
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        
        # Construir la ruta completa donde se guardará el archivo
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Guardar el archivo en la ruta especificada
        file.save(filepath)
        print(filepath)

        try:
            # Procesar la imagen cargada
            edad, tono = age_skin_tone(filepath)
            forma, prob = predict_face_shape(filepath, model)
            recomendacion_edad, recomendacion_tono, recomendacion_cara, link = obtener_recomendaciones_db(edad, tono, forma, prob)
            forma2 = f"{forma} (Probabilidad: {round(prob*100, 2)}%)"

            # Devolver los resultados
            return render_template('results.html', edad=edad, tono=tono, 
                                   forma=forma2, link=link, filepath=filepath,
                                   recomendacion_edad=recomendacion_edad,
                                   recomendacion_tono=recomendacion_tono,
                                   recomendacion_cara=recomendacion_cara)
        except ValueError as e:
            if "Face could not be detected" in str(e):
                error_message = "No se pudo detectar una cara en la imagen. Por favor, confirme que la imagen es una foto de un rostro."
                return render_template('error.html', error_message=error_message)
            else:
                error_message = "Hubo un error al procesar la imagen."
                return render_template('error.html', error_message=error_message)

    # Devolver la plantilla de carga (upload.html) si la solicitud no es POST o el archivo no es permitido
    return render_template('upload.html')

@app.route('/tmp/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ejecutar la aplicación si el script es ejecutado directamente
if __name__ == '__main__':
    app.debug = True
    app.run()
