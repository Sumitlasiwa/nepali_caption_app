from flask import Flask, render_template, request, redirect, url_for
import os
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from PIL import Image
from model import load_checkpoint, generate_caption
from gtts import gTTS  # For text-to-speech
import time
import threading
from datetime import datetime, timedelta
import gdown

device = 'cpu'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['AUDIO_FOLDER'] = 'static/audio/'  # Folder to store TTS audio files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model once at startup

url = "https://drive.google.com/uc?id=1wcEIqZwYdDETxKw8aks1aQD1QiyMsohD"
CHECKPOINT_PATH = "FinalAEsolveoverfit_checkpoint.pth"
gdown.download(url, CHECKPOINT_PATH, quiet=False)
# CHECKPOINT_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\CodePlayGround\Image_cap_app\FinalAEsolveoverfit_checkpoint.pth"
model, optimizer, start_epoch, train_losses, val_losses, word2idx, idx2word = load_checkpoint(CHECKPOINT_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to delete old audio files
def cleanup_audio_or_img_files(folder_key):
    while True:
        try:
            folder = app.config[folder_key]
            now = datetime.now()
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if now - file_creation_time > timedelta(hours=1):  # Delete files older than 1 hour
                    os.remove(filepath)
                    print(f"Deleted old file: {filename}")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(3600)  # Run cleanup every hour

# Start the cleanup thread when the app starts
cleanup_thread = threading.Thread(target=cleanup_audio_or_img_files, args=("AUDIO_FOLDER",), daemon=True)
cleanup_thread.start()
cleanup_thread = threading.Thread(target=cleanup_audio_or_img_files, args=("UPLOAD_FOLDER",), daemon=True)
cleanup_thread.start()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate caption
            image = Image.open(filepath).convert('RGB')
            image = image_transform(image)
            caption = generate_caption(model, image, word2idx, idx2word, device, max_length=20, beam_size=5)
            
            # Generate TTS audio using gTTS
            audio_filename = f"caption_{filename.rsplit('.', 1)[0]}.mp3"
            audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)

            try:
                tts = gTTS(text=caption, lang='ne')  # Use 'ne' for Nepali
                tts.save(audio_path)
            except Exception as e:
                print(f"Error generating TTS: {e}")
                audio_filename = None  # Ensure no broken audio file is used
            
            return render_template(
                'index.html', 
                image_path=filepath,
                caption=caption,
                audio_file=audio_filename
            )
    
    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)  # Create audio folder
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true')