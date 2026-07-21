import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Konfigurasi Folder Upload & Ekstensi yang Diizinkan
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Muat Model AI (Sesuaikan path jika berbeda)
MODEL_PATH = 'models/egg_model.keras'
try:
    model = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")

# Daftar Kelas sesuai label training
CLASS_NAMES = ['Crack', 'Empty', 'Good']

def allowed_file(filename):
    """Memeriksa apakah format file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(224, 224)):
    """Memproses gambar agar siap dimasukkan ke dalam model."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ubah shape jadi (1, 224, 224, 3)
    img_array = img_array / 255.0                  # Normalisasi nilai piksel (0-1)
    return img_array

# Route untuk Halaman Utama
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route untuk Prediksi (Inference API)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if file and allowed_file(file.filename):
        # Simpan file secara aman
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Preprocessing gambar
            processed_img = preprocess_image(file_path)

            # Jalankan Prediksi
            predictions = model.predict(processed_img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100

            predicted_label = CLASS_NAMES[predicted_class_idx]

            # Mengembalikan respon dalam format JSON
            return jsonify({
                'success': True,
                'class': predicted_label,
                'confidence': f"{confidence:.2f}%",
                'image_path': file_path
            })

        except Exception as e:
            return jsonify({'error': f"Gagal memproses gambar: {str(e)}"}), 500

    return jsonify({'error': 'Format file tidak didukung (Gunakan JPG, JPEG, atau PNG)'}), 400

if __name__ == '__main__':
    # Jalankan server dalam mode Debug
    app.run(debug=True, host='0.0.0.0', port=5000)