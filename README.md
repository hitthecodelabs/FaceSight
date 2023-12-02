# FaceSight

## Overview
This Flask application is designed for facial analysis and recommendation generation. It uses TensorFlow and a trained convolutional neural network model to estimate age, skin tone, and face shape from uploaded images. The application then provides personalized recommendations based on these features.

## Installation

### Prerequisites
- Python 3.x
- Flask
- TensorFlow
- PIL (Python Imaging Library)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/hitthecodelabs/FaceSight.git
   ```
2. Navigate to the project directory:
   ```
   cd FaceSight
   ```
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```
2. Access the application through a web browser at `http://localhost:5000/`.
3. Upload an image file (PNG, JPG, or JPEG) using the web interface.
4. The application will process the image and display the results, including estimated age, skin tone, face shape, and personalized recommendations.

## Features
- Image upload and secure file handling.
- Facial analysis for age, skin tone, and face shape estimation.
- Personalized recommendations based on facial analysis.

## Additional Notes
- Ensure that the `model.h5` file for the neural network is placed in the project directory.
- The application uses SQLite for database operations (if relevant to your project).
- Customize the secret key in `app.py` for security purposes.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License
This project is open source and available under the [MIT License](LICENSE).
