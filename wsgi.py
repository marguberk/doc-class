import os
import sys

# Добавляем путь к приложению в PYTHONPATH
path = '/home/YOUR_USERNAME/doc-class'  # Замените YOUR_USERNAME на ваше имя пользователя
if path not in sys.path:
    sys.path.append(path)

# Активируем виртуальное окружение
activate_this = os.path.join(path, 'venv/bin/activate_this.py')
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

# Импортируем и создаем приложение
from app import app as application

# Убедимся, что директории существуют
os.makedirs(os.path.join(path, 'classified_documents'), exist_ok=True)
os.makedirs(os.path.join(path, 'results/fold_3'), exist_ok=True)

if __name__ == '__main__':
    application.run() 