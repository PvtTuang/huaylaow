# ใช้ Python Base Image
FROM python:3.11-slim

# ตั้งค่า Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# ลง Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# ก๊อปปี้ Code เข้าไปใน Container
COPY . .

# คำสั่งรัน Server ผ่าน Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT huaylaow-dc7da.wsgi:application