# Analisis emocional de audio

Servicio HTTP que analiza un audio en espanol por fragmentos y responde con el
porcentaje de cada emocion.

Modelo local: [UMUTeam/w2v-bert-emotion-es](https://huggingface.co/UMUTeam/w2v-bert-emotion-es).
Corre en CPU, aunque si la instancia tiene CUDA disponible tambien puede usar GPU.

## API

`POST /analyze` con JSON:

| Campo | Descripcion |
|---|---|
| `s3_url` | URL del audio en S3. Acepta `https://...` presignada o `s3://bucket/key` |
| `fragmento_segundos` | Opcional. Entre 1 y 10. Por defecto 3 |

Formatos admitidos por extension: `.aac`, `.flac`, `.m4a`, `.mp3`, `.mpeg`,
`.mpga`, `.oga`, `.ogg`, `.wav`.

La peticion espera el resultado. Si tarda mas que `JOB_WAIT_TIMEOUT`, responde
`202` con `job_id` y el resultado se consulta con `GET /jobs/{job_id}`.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cambia_este_token" \
  -d '{"s3_url":"s3://mi-bucket/audio/llamada.wav","fragmento_segundos":3}'
```

`GET /health` devuelve estado del modelo, cola y trabajo activo.

## Desarrollo local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py --model-dir models/w2v-bert-emotion-es
MODEL_DIR="$PWD/models/w2v-bert-emotion-es" python api.py
```

En Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_model.py --model-dir models/w2v-bert-emotion-es
$env:MODEL_DIR="$PWD\models\w2v-bert-emotion-es"
python api.py
```

Usa un solo proceso. La cola vive en memoria y no se comparte si lanzas varios
workers de uvicorn.

## Despliegue en EC2

Recomendacion inicial: Ubuntu 22.04/24.04, 4 vCPU, 8 GB RAM como minimo
comodo para CPU. Usa un EBS de 20 GB o mas, porque PyTorch y el modelo ocupan
varios GB.

### 1. Preparar la instancia

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libsndfile1 nginx
```

Clona o copia el proyecto en `/var/www/emotion-analyzer`:

```bash
sudo mkdir -p /var/www/emotion-analyzer
sudo chown ubuntu:www-data /var/www/emotion-analyzer
git clone <URL_DE_TU_REPO> /var/www/emotion-analyzer
cd /var/www/emotion-analyzer
```

Si subes los archivos por SCP en vez de Git, deja la carpeta con owner `ubuntu`.

### 2. Crear entorno e instalar dependencias

```bash
cd /var/www/emotion-analyzer
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si la instancia solo usa CPU y quieres evitar paquetes CUDA innecesarios,
puedes instalar PyTorch CPU antes del resto:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r requirements.txt
```

### 3. Descargar el modelo local

La app arranca en modo offline para Hugging Face. Por eso el modelo debe quedar
descargado antes de iniciar systemd.

```bash
cd /var/www/emotion-analyzer
source venv/bin/activate
python scripts/download_model.py --model-dir /var/www/emotion-analyzer/models/w2v-bert-emotion-es
```

### 4. Crear variables de entorno

```bash
sudo mkdir -p /etc/emotion-analyzer
sudo cp .env.example /etc/emotion-analyzer/emotion-analyzer.env
sudo nano /etc/emotion-analyzer/emotion-analyzer.env
```

Valores recomendados:

```env
MODEL_DIR=/var/www/emotion-analyzer/models/w2v-bert-emotion-es
HOST=0.0.0.0
PORT=8000
MAX_QUEUE_SIZE=8
JOB_WAIT_TIMEOUT=900
JOB_TTL_SECONDS=3600
MAX_UPLOAD_MB=50
API_TOKEN=cambia_este_token_largo
```

Para `s3://bucket/key`, asigna a la instancia un IAM Role con `s3:GetObject`
sobre el bucket. Para URL presignada `https://...`, no hace falta credencial AWS.

Si `API_TOKEN` queda vacio o comentado, la API no exige `X-API-Key`.

### 5. Instalar systemd

```bash
sudo cp deploy/emotion-analyzer.service /etc/systemd/system/emotion-analyzer.service
sudo systemctl daemon-reload
sudo systemctl enable emotion-analyzer
sudo systemctl start emotion-analyzer
```

Verifica logs y salud:

```bash
sudo journalctl -u emotion-analyzer -f
curl http://127.0.0.1:8000/health
```

### 6. Publicar con Nginx opcional

Si quieres exponer HTTP por puerto 80 y dejar la API escuchando solo detras de
Nginx:

```bash
sudo cp deploy/nginx-emotion-analyzer.conf /etc/nginx/sites-available/emotion-analyzer
sudo ln -s /etc/nginx/sites-available/emotion-analyzer /etc/nginx/sites-enabled/emotion-analyzer
sudo nginx -t
sudo systemctl reload nginx
```

En el Security Group de EC2 abre:

- Puerto `80` si usas Nginx.
- Puerto `8000` solo si decides acceder directo a Uvicorn.
- Puerto `22` restringido a tu IP.

En produccion publica preferentemente con HTTPS usando tu dominio y Certbot, o
coloca un Load Balancer/CloudFront delante.

## Operacion

Comandos utiles:

```bash
sudo systemctl status emotion-analyzer
sudo systemctl restart emotion-analyzer
sudo journalctl -u emotion-analyzer -n 200 --no-pager
curl http://127.0.0.1:8000/health
```

Actualizar codigo:

```bash
cd /var/www/emotion-analyzer
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart emotion-analyzer
```

Si cambias `MODEL_DIR` o parametros de cola, edita
`/etc/emotion-analyzer/emotion-analyzer.env` y reinicia el servicio.
