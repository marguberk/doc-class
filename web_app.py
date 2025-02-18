import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from google.cloud import storage

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Google Cloud Storage configuration
BUCKET_NAME = "doc-class-storage"  # Замените на имя вашего бакета
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# Настройки загрузки файлов
UPLOAD_FOLDER = 'classified_documents'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Подключение к базе данных
def get_db():
    db = sqlite3.connect('classified_documents.db')
    db.row_factory = sqlite3.Row
    return db

# Инициализация базы данных
def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT UNIQUE,
            author TEXT,
            secrecy_level TEXT,
            class_name TEXT,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.commit()

# Модель
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Identity()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(1280 + 768, 11)

    def forward(self, x):
        x1 = self.efficientnet(x)
        x2 = self.vit(x)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

# Загрузка модели
def load_model():
    model_path = 'results/fold_3/model.pth'
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        blob = bucket.blob('model.pth')
        blob.download_to_filename(model_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = HybridModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Преобразование изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Классы документов
class_names = [
    'қызмет хат', 'өкім', 'қосымша', 'диплом', 'анықтама',
    'бітіру бұйрығы', 'қабылдау', 'өндірістік бұйрықтар',
    'транскрипт', 'оқу картасы', 'есепке алу'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_gcs(file, filename):
    blob = bucket.blob(f'documents/{filename}')
    blob.upload_from_file(file)
    return blob.public_url

@app.route('/')
def index():
    if 'user_role' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == '123456':
            session['user_role'] = 'admin'
            flash('Сәтті кірдіңіз!')
            return redirect(url_for('index'))
        else:
            flash('Қате логин немесе құпия сөз')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Файл таңдалмады')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Файл таңдалмады')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Сохраняем временно для классификации
            temp_path = os.path.join('/tmp', filename)
            file.save(temp_path)
            
            # Классификация
            image = Image.open(temp_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
            
            # Загружаем в Google Cloud Storage
            file.seek(0)
            file_url = upload_to_gcs(file, filename)
            
            # Сохранение в БД
            db = get_db()
            db.execute(
                'INSERT INTO documents (document_name, class_name, file_path) VALUES (?, ?, ?)',
                (filename, predicted_class, file_url)
            )
            db.commit()
            
            # Удаляем временный файл
            os.remove(temp_path)
            
            flash(f'Файл "{filename}" жүктелді және "{predicted_class}" санатына жіктелді')
            return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/search')
def search():
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    query = request.args.get('query', '')
    db = get_db()
    if query:
        documents = db.execute(
            'SELECT * FROM documents WHERE document_name LIKE ? OR class_name LIKE ?',
            (f'%{query}%', f'%{query}%')
        ).fetchall()
    else:
        documents = db.execute('SELECT * FROM documents').fetchall()
    
    return render_template('search.html', documents=documents)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080))) 