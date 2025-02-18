import sys
import os
import sqlite3
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F  # For softmax
import torchvision.transforms as transforms
import timm

# --- Деректер базасымен байланыс ---
conn = sqlite3.connect("classified_documents.db")
cursor = conn.cursor()

# --- Кестені құру (егер жоқ болса) ---
cursor.execute("""
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
conn.commit()

# --- Модель анықтамасы ---
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Match training structure

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.fc = nn.Linear(1280 + 768, 11)

    def forward(self, x):
        x1 = self.efficientnet(x)
        x2 = self.vit(x)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)

# --- Модельді жүктеу ---
model_path = 'results/fold_3/model.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = HybridModel().to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict, strict=False)
model.eval()

# --- Өңдеуді анықтау ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Сынып атауларын анықтау ---
class_names = [
    'қызмет хат',      # 0
    'өкім',           # 1
    'қосымша',        # 2
    'диплом',         # 3
    'анықтама',       # 4
    'бітіру бұйрығы',  # 5
    'қабылдау',       # 6
    'өндірістік бұйрықтар',  # 7
    'транскрипт',     # 8
    'оқу картасы',    # 9
    'есепке алу'      # 10
]

if "Басқа" not in class_names:
    class_names.append("Басқа")  # Соңына қосу

# --- Дубликат тексеру ---
def is_duplicate(document_name):
    cursor.execute("SELECT COUNT(*) FROM documents WHERE document_name = ?", (document_name,))
    return cursor.fetchone()[0] > 0

def insert_document(name, author, secrecy_level, class_name, file_path):
    if not is_duplicate(name):
        with conn:
            conn.execute("""
                INSERT INTO documents (document_name, author, secrecy_level, class_name, file_path)
                VALUES (?, ?, ?, ?, ?)
            """, (name, author, secrecy_level, class_name, file_path))
        return True
    return False

def classify_image(image_path):
    """Суретті жіктеу және болжамды сынып пен сенімділікті қайтару."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)

    if max_prob.item() < 0.8:
        predicted_class = len(class_names) - 1  # "Басқа" индексі
    return predicted_class, max_prob.item()

# --- New dialog for editing file metadata after classification ---
class EditFileMetadataDialog(QtWidgets.QDialog):
    def __init__(self, original_file_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Файл метадеректерін өзгерту")
        layout = QtWidgets.QFormLayout(self)
        
        # Field for file name (without extension)
        self.filename_edit = QtWidgets.QLineEdit(original_file_name)
        # Field for security level
        self.security_level_combo = QtWidgets.QComboBox()
        self.security_level_combo.addItems(["Жоғары", "Орташа", "Төмен"])
        # Field for author
        self.author_edit = QtWidgets.QLineEdit()
        self.author_edit.setPlaceholderText("Авторды енгізіңіз")
        
        layout.addRow("Файл атауы:", self.filename_edit)
        layout.addRow("Құпиялық деңгейі:", self.security_level_combo)
        layout.addRow("Автор:", self.author_edit)
        
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def getValues(self):
        return (self.filename_edit.text().strip(), 
                self.security_level_combo.currentText().strip(), 
                self.author_edit.text().strip())

# --- Негізгі терезе ---
class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Құжаттарды Басқару")
        self.setGeometry(300, 100, 800, 600)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.setCentralWidget(widget)
        
        self.user_role = self.show_login_dialog()
        
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #1E1E2E;
                font-weight: bold;
                margin-top: 10px;
            }
        """)
        layout.addWidget(self.status_label)
        
        logo_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap("static/AYU_Logo.png")
        logo_label.setPixmap(pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio))
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)
        
        self.title_label = QtWidgets.QLabel("Архив құжаттарын басқару жүйесі")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; color: #1E1E2E; margin-bottom: 20px;")
        layout.addWidget(self.title_label)

        self.upload_button = QtWidgets.QPushButton("Құжатты жүктеу")
        self.upload_button.setObjectName("uploadButton")
        self.upload_button.clicked.connect(self.open_upload_page)
        layout.addWidget(self.upload_button)

        self.search_button = QtWidgets.QPushButton("Құжат іздеу")
        self.search_button.setObjectName("searchButton")
        self.search_button.clicked.connect(self.open_search_page)
        layout.addWidget(self.search_button)

        self.view_all_button = QtWidgets.QPushButton("Деректер базасы")
        self.view_all_button.setObjectName("viewAllButton")
        self.view_all_button.clicked.connect(self.view_all_documents)
        layout.addWidget(self.view_all_button)
        
        self.update_font_sizes()
        self.resizeEvent = self.handle_resize

    def handle_resize(self, event):
        self.update_font_sizes()

    def update_font_sizes(self):
        width = self.size().width()
        base_font_size = max(12, int(width * 0.02))
        self.title_label.setStyleSheet(f"font-size: {base_font_size + 10}px; font-weight: bold; color: #1E1E2E;")
        button_font = QtGui.QFont()
        button_font.setPointSize(base_font_size)
        self.upload_button.setFont(button_font)
        self.search_button.setFont(button_font)
        self.view_all_button.setFont(button_font)
    
    def open_upload_page(self):
        self.upload_window = UploadPage(self)
        self.upload_window.show()
        self.hide()

    def open_search_page(self):
        self.search_window = SearchPage(self)
        self.search_window.show()
        self.hide()
        
    def view_all_documents(self):
        self.view_window = ViewAllDocuments(self)
        self.view_window.show()
        self.close()

    def show_login_dialog(self):
        login_dialog = LoginDialog(self)
        if login_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return login_dialog.user_role
        else:
            return "user"

# --- Жүйеге кіру диалогы ---
class LoginDialog(QtWidgets.QDialog):
    """Пайдаланушыны аутентификациялау үшін кіру диалогы."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Кіру")
        self.setGeometry(300, 100, 800, 600)  # Басқа терезелермен үйлесімді өлшем

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Логотип (егер қажет болса)
        logo_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap("static/AYU_Logo.png")
        logo_label.setPixmap(pixmap.scaled(150, 150, QtCore.Qt.KeepAspectRatio))
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(logo_label)
        
        # Пайдаланушы аты мен құпия сөзді енгізу үшін форма
        form_layout = QtWidgets.QFormLayout()
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Пайдаланушы аты")
        form_layout.addRow("Пайдаланушы аты:", self.username_input)
        
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setPlaceholderText("Құпия сөз")
        self.password_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        form_layout.addRow("Құпия сөз:", self.password_input)
        layout.addLayout(form_layout)
        
        # Кіру батырмасы
        login_button = QtWidgets.QPushButton("Кіру")
        login_button.clicked.connect(self.authenticate)
        layout.addWidget(login_button)
        
        layout.addStretch()
        
        self.user_role = None

    def authenticate(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if username == "admin" and password == "123456":
            self.user_role = "admin"
            QtWidgets.QMessageBox.information(self, "Сәттілік", "Жүйеге администратор ретінде кірдіңіз")
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Қате", "Жүйеге кіре алмадыңыз")
            self.user_role = "user"
            self.accept()


class EditFileMetadataDialog(QtWidgets.QDialog):
    def __init__(self, original_file_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Файл метадеректерін өзгерту")
        self.setGeometry(300, 100, 800, 600)  # Өлшемі іздеу бетіне ұқсас
        layout = QtWidgets.QFormLayout(self)
        
        self.filename_edit = QtWidgets.QLineEdit(original_file_name)
        self.filename_edit.setObjectName("fileNameEdit")

        self.security_level_combo = QtWidgets.QComboBox()
        self.security_level_combo.addItems(["Жоғары", "Орташа", "Төмен"])
        self.author_edit = QtWidgets.QLineEdit()
        self.author_edit.setPlaceholderText("Авторды енгізіңіз")
        
        layout.addRow("Файл атауы:", self.filename_edit)
        layout.addRow("Құпиялық деңгейі:", self.security_level_combo)
        layout.addRow("Автор:", self.author_edit)
        
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def getValues(self):
        return (self.filename_edit.text().strip(), 
                self.security_level_combo.currentText().strip(), 
                self.author_edit.text().strip())

# --- Drag & Drop арқылы файл жүктеу виджеті ---
class DragAndDropWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # Биіктігін арттырамыз, фон түсі мен шекарасы өзгереді
        self.setMinimumHeight(200)
        self.setStyleSheet("""
            QWidget {
                border: 2px dashed #FF6F61;
                border-radius: 10px;
                background-color: #F8F8F8;
            }
            QLabel {
                color: #1E1E2E;
                font-size: 16px;
            }
            QPushButton {
                background-color: #FFFFFF;
                color: #1E1E2E;
                border: 1px solid #FF6F61;
                border-radius: 5px;
                padding: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #FFF0F0;
            }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.label = QtWidgets.QLabel("Файлды осында тастаңыз\nнемесе 'Файлды таңдау' батырмасын басыңыз")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)
        
        self.button = QtWidgets.QPushButton("Файлды таңдау")
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)
        
        self.setLayout(layout)

    def open_file_dialog(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Файлды таңдаңыз",
            "",
            "Суреттер мен PDF файлдар (*.pdf *.tiff *.jpg *.jpeg *.png)"
        )
        if file_path:
            self.parentWidget().file_path = file_path
            self.parentWidget().classify_and_save()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf")):
                self.parentWidget().file_path = file_path
                self.parentWidget().classify_and_save()


    def open_file_dialog(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Файлды таңдаңыз",
            "",
            "Суреттер мен PDF файлдар (*.pdf *.tiff *.jpg *.jpeg *.png)"
        )
        if file_path:
            self.parentWidget().file_path = file_path
            self.parentWidget().classify_and_save()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf")):
                self.parentWidget().file_path = file_path
                self.parentWidget().classify_and_save()

# --- Құжатты жүктеу беті (без author/security inputs) ---
class UploadPage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Құжатты жүктеу")
        self.setGeometry(300, 100, 800, 600)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        self.drag_and_drop_widget = DragAndDropWidget(self)
        layout.addWidget(self.drag_and_drop_widget)
        
        self.upload_status = QtWidgets.QLabel("")
        self.upload_status.setObjectName("uploadMessage")
        layout.addWidget(self.upload_status)
        
        self.back_button = QtWidgets.QPushButton("Артқа")
        self.back_button.setObjectName("backButton")
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)
        
        self.file_path = None

    def select_file(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Файлды таңдаңыз",
            "",
            "Суреттер мен PDF файлдар (*.pdf *.tiff *.jpg *.jpeg *.png)"
        )
        if self.file_path:
            self.classify_and_save()

    def classify_and_save(self):
        try:
            if not self.file_path:
                raise ValueError("Файл таңдалмады")

            predicted_index, confidence = classify_image(self.file_path)
            predicted_index = predicted_index.item() if isinstance(predicted_index, torch.Tensor) else predicted_index
            class_name = class_names[predicted_index]
            save_dir = os.path.join('classified_documents', class_name)
            os.makedirs(save_dir, exist_ok=True)
            
            original_name = os.path.basename(self.file_path)
            ext = os.path.splitext(original_name)[1]
            
            # Ауыстырылатын метадеректерді өзгерту үшін диалог ашамыз
            dialog = EditFileMetadataDialog(original_file_name=os.path.splitext(original_name)[0], parent=self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                new_file_name, new_sec_level, new_author = dialog.getValues()
            else:
                return  # Пайдаланушы диалогты жапты
            
            if not new_file_name.endswith(ext):
                new_file_name = new_file_name + ext
            save_path = os.path.join(save_dir, new_file_name)
            
            # PDF файлдар үшін: бірінші бетті суретке айналдырамыз
            if self.file_path.lower().endswith('.pdf'):
                import fitz
                doc = fitz.open(self.file_path)
                if doc.page_count < 1:
                    raise ValueError("PDF құжаты беттерсіз")
                page = doc[0]
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                image.save(save_path)
            else:
                Image.open(self.file_path).save(save_path)
            
            if insert_document(new_file_name, new_author, new_sec_level, class_name, save_path):
                self.upload_status.setText(f"«{new_file_name}» құжаты «{class_name}» санатына сақталды")
            else:
                self.upload_status.setText(f"«{new_file_name}» атты құжат деректер базасында бар")
        except Exception as e:
            self.upload_status.setText(f"Қате: {str(e)}")

    def go_back(self):
        self.close()
        self.parent.show()

# --- Құжат іздеу беті ---
class SearchPage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Құжат іздеу")
        self.setGeometry(300, 100, 800, 600)

        # Main vertical layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Horizontal layout for search controls
        search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Іздеу сөзі")
        search_layout.addWidget(self.search_input)
        self.search_class_combo = QtWidgets.QComboBox()
        self.search_class_combo.addItems(["Барлық"] + class_names)
        search_layout.addWidget(self.search_class_combo)
        self.search_button = QtWidgets.QPushButton("Іздеу")
        self.search_button.clicked.connect(self.search_documents)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)

        # QTableWidget for search results
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Файл атауы", "Құпиялық деңгейі", "Санат", "Файл орналасқан жері"])
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Back button
        self.back_button = QtWidgets.QPushButton("Артқа")
        self.back_button.setObjectName("backButton")
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        # Connect double-click event to open the document
        self.table.itemDoubleClicked.connect(self.open_document)

    def search_documents(self):
        keyword = self.search_input.text()
        selected_class = self.search_class_combo.currentText()
        query = "SELECT document_name, secrecy_level, class_name, file_path FROM documents WHERE 1=1"
        params = []
        if keyword:
            query += " AND document_name LIKE ?"
            params.append(f"%{keyword}%")
        if selected_class != "Барлық":
            query += " AND class_name = ?"
            params.append(selected_class)
        cursor.execute(query, params)
        results = cursor.fetchall()

        self.table.setRowCount(len(results))
        for row_idx, row in enumerate(results):
            for col_idx, item in enumerate(row):
                cell_item = QtWidgets.QTableWidgetItem(str(item))
                self.table.setItem(row_idx, col_idx, cell_item)
        # If you prefer not to display the file path, uncomment the next line:
        # self.table.setColumnHidden(3, True)

    def open_document(self, item):
        row = item.row()
        file_path_item = self.table.item(row, 3)
        if file_path_item:
            file_path = file_path_item.text()
            # Check if the security level is "Жоғары"
            security_item = self.table.item(row, 1)
            if security_item and security_item.text() == "Жоғары" and self.parent.user_role != "admin":
                self.prompt_admin_login(file_path)
            else:
                os.startfile(file_path)

    def prompt_admin_login(self, file_path):
        login_dialog = LoginDialog(self)
        if login_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and login_dialog.user_role == "admin":
            os.startfile(file_path)
        else:
            QtWidgets.QMessageBox.warning(self, "Рұқсат берілмеді", "Тек администраторға рұқсат етілген.")

    def go_back(self):
        self.close()
        self.parent.show()


# --- Деректер базасын қарау және өңдеу беті ---
class ViewAllDocuments(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Деректер базасы")
        self.setGeometry(300, 100, 800, 600)

        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget()
        # We still load 7 columns but later we hide column 0 (ID)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Файл атауы", "Автор", "Құпиялық деңгейі", "Санат", "Файл орналасқан жері", "Өңдеу"]
        )
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        self.back_button = QtWidgets.QPushButton("Артқа")
        self.back_button.setObjectName("backButton")
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        self.load_data()

    def load_data(self):
        cursor.execute("SELECT document_id, document_name, author, secrecy_level, class_name, file_path FROM documents")
        results = cursor.fetchall()
        self.table.setRowCount(len(results))
        for row_idx, row_data in enumerate(results):
            for col_idx, col_data in enumerate(row_data):
                self.table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(str(col_data)))
            edit_button = QtWidgets.QPushButton("Өңдеу")
            edit_button.setObjectName("editButton")
            edit_button.clicked.connect(lambda checked, r=row_idx: self.edit_document(r))
            self.table.setCellWidget(row_idx, 6, edit_button)
        # Hide the ID column (column 0)
        self.table.setColumnHidden(0, True)

    def edit_document(self, row):
        doc_id = self.table.item(row, 0).text()
        self.edit_window = EditDocumentWindow(doc_id, self)
        self.edit_window.show()

    def go_back(self):
        self.parent.show()
        self.close()


# --- Құжатты өңдеу терезесі ---
class EditDocumentWindow(QtWidgets.QWidget):
    def __init__(self, doc_id, parent):
        super().__init__()
        self.doc_id = doc_id
        self.parent = parent
        self.setWindowTitle("Өңдеу")
        self.setGeometry(300, 100, 400, 300)

        layout = QtWidgets.QVBoxLayout(self)
        self.name_input = QtWidgets.QLineEdit()
        self.author_input = QtWidgets.QLineEdit()
        self.secrecy_level_combo = QtWidgets.QComboBox()
        self.secrecy_level_combo.addItems(["Жоғары", "Орташа", "Төмен"])
        self.class_input = QtWidgets.QLineEdit()

        layout.addWidget(QtWidgets.QLabel("Файл атауы"))
        layout.addWidget(self.name_input)
        layout.addWidget(QtWidgets.QLabel("Автор"))
        layout.addWidget(self.author_input)
        layout.addWidget(QtWidgets.QLabel("Құпиялық деңгейі"))
        layout.addWidget(self.secrecy_level_combo)
        layout.addWidget(QtWidgets.QLabel("Санат"))
        layout.addWidget(self.class_input)

        save_button = QtWidgets.QPushButton("Сақтау")
        save_button.clicked.connect(self.save_changes)
        layout.addWidget(save_button)
        
        delete_button = QtWidgets.QPushButton("Жою")
        delete_button.clicked.connect(self.delete_document)
        layout.addWidget(delete_button)

        cancel_button = QtWidgets.QPushButton("Болдырмау")
        cancel_button.clicked.connect(self.close)
        layout.addWidget(cancel_button)

        self.load_document_data()

    def load_document_data(self):
        cursor.execute(
            "SELECT document_name, author, secrecy_level, class_name FROM documents WHERE document_id = ?",
            (self.doc_id,)
        )
        document = cursor.fetchone()
        if document:
            self.name_input.setText(document[0])
            self.author_input.setText(document[1])
            self.secrecy_level_combo.setCurrentText(document[2])
            self.class_input.setText(document[3])

    def save_changes(self):
        new_name = self.name_input.text()
        new_author = self.author_input.text()
        new_secrecy_level = self.secrecy_level_combo.currentText()
        new_class = self.class_input.text()
        try:
            cursor.execute(
                """
                UPDATE documents
                SET document_name = ?, author = ?, secrecy_level = ?, class_name = ?
                WHERE document_id = ?
                """,
                (new_name, new_author, new_secrecy_level, new_class, self.doc_id)
            )
            conn.commit()
            QtWidgets.QMessageBox.information(self, "Сәттілік", "Деректер сәтті жаңартылды.")
            self.parent.load_data()
            self.close()
        except sqlite3.IntegrityError:
            QtWidgets.QMessageBox.warning(self, "Қате", "Бұл құжат деректер базасында бар.")

    def delete_document(self):
        confirm = QtWidgets.QMessageBox.question(
            self, "Растау", "Сіз бұл құжатты жою керектігіне сенімдісіз бе?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            cursor.execute("SELECT file_path FROM documents WHERE document_id = ?", (self.doc_id,))
            file_path = cursor.fetchone()[0]
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                cursor.execute("DELETE FROM documents WHERE document_id = ?", (self.doc_id,))
                conn.commit()
                QtWidgets.QMessageBox.information(self, "Сәттілік", "Құжат сәтті жойылды.")
                self.parent.load_data()
                self.close()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Қате", f"Жою кезінде қате: {str(e)}")

# --- Бағдарламаны іске қосу ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    with open("styles.qss", "r") as f:
        app.setStyleSheet(f.read())
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
