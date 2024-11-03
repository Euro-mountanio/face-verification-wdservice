from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from deepface import DeepFace
import base64
import tempfile
import os
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure debug mode
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

@app.route('/')
def index():
    return render_template('index.html')

# Define the testRecognition function
def testRecognition(image1, image2):
    try:
        result = DeepFace.verify(img1_path=image1, img2_path=image2)
        return result
    except Exception as e:
        raise Exception(f"DeepFace error: {str(e)}")

# Endpoint to receive Base64 images and process them


@app.route('/verify', methods=['POST'])
def compare_images():
    try:
        # Parse JSON data from the request
        data = request.get_json()
        if 'image1' not in data or 'image2' not in data:
            return jsonify({'error': 'Missing image1 or image2 in request payload'}), 400

        # Decode Base64 images
        image1_data = base64.b64decode(data['image1'])
        image2_data = base64.b64decode(data['image2'])

        # Save decoded images temporarily
        temp1_path = None
        temp2_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1:
                temp1.write(image1_data)
                temp1_path = temp1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:
                temp2.write(image2_data)
                temp2_path = temp2.name

            # Run DeepFace verification
            result = testRecognition(temp1_path, temp2_path)
            return jsonify(result)

        finally:
            # Clean up temporary files
            if temp1_path and os.path.exists(temp1_path):
                os.remove(temp1_path)
            if temp2_path and os.path.exists(temp2_path):
                os.remove(temp2_path)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500




# Run the web server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
