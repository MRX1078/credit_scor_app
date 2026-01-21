# Используем официальный легкий образ Python
FROM python:3.9-slim

# Отключаем буферизацию вывода (чтобы логи сразу появлялись в консоли)
ENV PYTHONUNBUFFERED=1

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# (Опционально) Указываем порт, но реально он задается в docker-compose
EXPOSE 8000 8501

# Команда по умолчанию (будет переопределена в docker-compose)
CMD ["bash"]
