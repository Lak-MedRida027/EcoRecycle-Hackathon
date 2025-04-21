from flask import Flask, request, jsonify
import google.generativeai as genai
import util
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

application = Flask(__name__)

CORS(application)

util.load_artifacts()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name

@application.route("/")
def home():
    return "Hello, I am alive"


@application.route("/classifywaste", methods=["POST"])
def classifywaste():
    image_data = request.files["file"]
    # Save the image to upload
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(basepath, "uploads", secure_filename(image_data.filename))
    
    # Save the file
    image_data.save(image_path)
    
    # Make sure the file handle is closed
    image_data.close()
    
    # Process the image
    predicted_value, step1, step2, step3, step4, difficulty = util.classify_waste(image_path)
    
    # Try to remove the file with error handling
    try:
        os.remove(image_path)
    except PermissionError:
        # Log that we couldn't delete the file, but continue
        print(f"Could not delete {image_path}, file is in use")
    
    return jsonify(predicted_value=predicted_value, step1=step1, step2=step2, step3=step3, step4=step4, difficulty=difficulty)

@application.route('/chat-bot', methods=['POST'])
def chat_bot():
    try:
        data = request.get_json()
        prompt = data['prompt']
        
        response = model.generate_content(prompt)
        return jsonify({
            "success": True,
            "response": response.text
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    application.run()