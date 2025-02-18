# Система классификации документов

Система автоматической классификации документов с использованием гибридной модели на основе EfficientNet и ViT.

## Функциональность

- Автоматическая классификация документов по 11 категориям
- Загрузка и хранение документов
- Поиск по базе документов
- Управление правами доступа (администратор/пользователь)
- Современный графический интерфейс

## Категории документов

1. Қызмет хат (служебное письмо)
2. Өкім (распоряжение)
3. Қосымша (приложение)
4. Диплом
5. Анықтама (справка)
6. Бітіру бұйрығы (приказ об окончании)
7. Қабылдау (прием)
8. Өндірістік бұйрықтар (производственные приказы)
9. Транскрипт
10. Оқу картасы (учебная карта)
11. Есепке алу (учет)

## Локальная установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/marguberk/doc-class.git
cd doc-class
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/macOS
# или
venv\Scripts\activate  # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте необходимые директории:
```bash
mkdir -p classified_documents
mkdir -p results/fold_3
```

5. Загрузите модель в директорию `results/fold_3/model.pth`

## Развертывание на PythonAnywhere

1. Создайте аккаунт на PythonAnywhere (если еще не создали)

2. В разделе "Web" создайте новое веб-приложение:
   - Выберите "Manual configuration"
   - Выберите Python 3.8 (или новее)

3. В разделе "Consoles" откройте Bash консоль и выполните:
```bash
git clone https://github.com/marguberk/doc-class.git
cd doc-class
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Создайте необходимые директории:
```bash
mkdir -p classified_documents
mkdir -p results/fold_3
```

5. Загрузите модель:
   - В разделе "Files" перейдите в директорию doc-class/results/fold_3/
   - Нажмите "Upload a file" и загрузите model.pth

6. Настройте веб-приложение:
   - В разделе "Web" найдите "Code" секцию
   - Установите "Source code" на: /home/YOUR_USERNAME/doc-class
   - Установите "Working directory" на: /home/YOUR_USERNAME/doc-class
   - В "WSGI configuration file" замените содержимое на код из wsgi.py

7. В разделе "Static files" добавьте:
   - URL: /static/
   - Directory: /home/YOUR_USERNAME/doc-class/static

8. Нажмите "Reload" для перезапуска приложения

## Доступ администратора

- Логин: admin
- Пароль: 123456

## Технологии

- Python 3.x
- PyQt5
- PyTorch
- Timm
- SQLite 