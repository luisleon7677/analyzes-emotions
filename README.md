# Analisis emocional de audio

Servicio HTTP que analiza un audio en espanol por fragmentos y responde con el
porcentaje de cada emocion.

Modelo local: [UMUTeam/w2v-bert-emotion-es](https://huggingface.co/UMUTeam/w2v-bert-emotion-es).
Corre en CPU, aunque si la instancia tiene CUDA disponible tambien puede usar GPU.

## API

URL local: `http://127.0.0.1:8000`. En produccion, reemplazala por la URL
del servidor. Puedes probar los endpoints desde `/docs`.

### Analizar un audio

`POST /analyze` con JSON:

| Campo | Descripcion |
|---|---|
| `s3_url` | URL del audio en S3. Acepta `https://...` presignada o `s3://bucket/key` |
| `fragmento_segundos` | Opcional. Entre 1 y 10. Por defecto 3 |

Formatos admitidos por extension: `.aac`, `.flac`, `.m4a`, `.mp3`, `.mpeg`,
`.mpga`, `.oga`, `.ogg`, `.wav`.

Tamano maximo predeterminado: **150 MB** (150 x 1024 x 1024 bytes), configurable
con `MAX_UPLOAD_MB`. El archivo debe poder decodificarse como audio.
Se envia la URL en JSON; este endpoint no recibe archivos mediante multipart.
Para HTTPS, usa una URL accesible por el servidor, por ejemplo una URL presignada
de S3 vigente. Para `s3://`, el servidor necesita credenciales AWS o un rol IAM
con permiso `s3:GetObject` sobre el archivo.

Si el administrador configuro `API_TOKEN`, incluye `X-API-Key` en las peticiones
a `/analyze` y `/jobs/{job_id}`. Si no lo configuro, puedes omitir ese header.

La peticion valida el JSON y guarda el job en RAM. Responde HTTP `202` con
`job_id` y `estado: "en_cola"` antes de descargar o analizar el audio.
El cliente consulta `GET /jobs/{job_id}` cada 30 segundos.
`JOB_WAIT_TIMEOUT` ya no se utiliza. La descarga y el analisis se ejecutan
en hilos internos del mismo servicio.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cambia_este_token" \
  -d '{"s3_url":"s3://mi-bucket/audio/llamada.wav","fragmento_segundos":3}'
```

Cuando `GET /jobs/{job_id}` devuelve HTTP `200`, recibes `estado: "listo"` y estos campos:

| Campo | Descripcion |
|---|---|
| `job_id` | Identificador del trabajo |
| `archivo`, `duracion_segundos` | Nombre y duracion del audio |
| `fragmentos`, `fragmento_segundos` | Cantidad de fragmentos y tamano solicitado en segundos |
| `emociones` | Porcentajes de `triste`, `miedo`, `disgusto`, `enojo`, `neutral` y `alegre`; suman 100 |
| `valencia`, `tono` | Valencia de 0 a 100 y descripcion del tono general |
| `detalle` | Por fragmento: `tiempo_segundos`, `emocion`, `confianza` y `valencia` |

### Consultar un trabajo pendiente

Ejemplo de respuesta HTTP `202` de `/analyze`:

```json
{
  "job_id": "identificador_del_trabajo",
  "estado": "en_cola"
}
```

Consulta el identificador recibido cada pocos segundos:

```bash
curl http://127.0.0.1:8000/jobs/identificador_del_trabajo \
  -H "X-API-Key: cambia_este_token"
```

Devuelve `202` para `en_cola` o `procesando`, `200` con el
resultado cuando este `listo`, o `422` con `error` si falla.
Los jobs, estados, claves de idempotencia y resultados se guardan en RAM.
Se pierden al reiniciar o cerrar el proceso. Los resultados caducan 24 horas
despues de finalizar (`JOB_TTL_SECONDS=86400`); se eliminan al consultar o
encolar. GET devuelve `404` si el job se perdio o caduco.

### Salud y errores

`GET /health` no requiere API key y devuelve estado del modelo (`cargando`,
`ok` o `error`), cantidad de pendientes y trabajos en procesamiento:

```bash
curl http://127.0.0.1:8000/health
```

| HTTP | Significado |
|---|---|
| `400` | URL o extension invalida |
| `401` | API key ausente o incorrecta |
| `404` | Trabajo inexistente, caducado o perdido por reinicio |
| `409` | Idempotency-Key reutilizada con otro payload |
| `422` | JSON invalido, fragmento fuera de 1 a 10 segundos o fallo de analisis |
| `429` | Cola llena; reintenta mas tarde |
| `503` | Fallo al cargar el modelo; revisa los logs |

Los errores de peticion suelen incluir `detail`; los fallos de analisis
incluyen `job_id`, `estado: "error"` y `error`.

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

Ejecuta un solo servicio y un solo proceso: `python api.py` o
`uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1`.
El servicio inicia automaticamente los hilos de descarga y analisis.
La cola no se comparte entre procesos o instancias; no uses varios workers.

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
####
```env
MODEL_DIR=/var/www/emotion-analyzer/models/w2v-bert-emotion-es
HOST=0.0.0.0
PORT=8000
MAX_QUEUE_SIZE=20
AUDIO_DIR=/var/www/emotion-analyzer/data/audio
JOB_TTL_SECONDS=86400
DOWNLOAD_MAX_WAIT_SECONDS=3300
DOWNLOAD_TIMEOUT_SECONDS=600
MAX_UPLOAD_MB=150
API_TOKEN=cambia_este_token_largo
```

Para `s3://bucket/key`, asigna a la instancia un IAM Role con `s3:GetObject`
sobre el bucket. Para URL presignada `https://...`, no hace falta credencial AWS.

Si `API_TOKEN` queda vacio o comentado, la API no exige `X-API-Key`.

### 5. Instalar systemd

```bash
sudo mkdir -p /var/www/emotion-analyzer/data/audio
sudo chown -R ubuntu:ubuntu /var/www/emotion-analyzer/data
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

## Cola en RAM y NestJS

Solo se inicia `python api.py`. `worker.py` es un modulo interno importado por
la API; no se ejecuta ni instala como otro servicio. No hay base de datos,
migraciones ni cola externa. Los resultados se guardan como diccionarios Python
con acceso protegido por un lock. El POST encola y devuelve el ID sin esperar
la descarga o inferencia; cerrar la conexion del cliente no cancela el job.

Hay un hilo de descarga y uno de analisis dentro del proceso. La descarga
avanza mientras otro audio se analiza, para reducir el riesgo de que las URLs
firmadas venzan. GET devuelve `procesando` durante la descarga, la espera por
CPU despues de descargar y el analisis. Se mantienen 20 pendientes y un analisis activo.
El limite incluye los jobs en descarga y los audios esperando el modelo.
Los archivos de audio usan temporalmente `AUDIO_DIR` en disco, por el lector
actual del modelo; los jobs y resultados usan RAM. El audio se borra al terminar.
Una caida forzada puede dejar archivos temporales, que requieren limpieza.

Para firmas SigV4 se comprueba X-Amz-Date + X-Amz-Expires con margen de 60 s;
tambien se admite Expires. Sin fecha reconocida, se permite esperar hasta
DOWNLOAD_MAX_WAIT_SECONDS (3300 s por defecto) para iniciar la descarga.
Si vence el plazo, el job pasa a error. Las descargas lentas o una caida pueden
superar la hora disponible: la API no puede renovar una firma. NestJS debe crear
la URL al enviar y generar otra URL y clave para reintentar un vencimiento.
La alternativa es s3://bucket/key con rol IAM. Sigue vigente MAX_UPLOAD_MB=150.

Se conserva Idempotency-Key opcional: solicitudes concurrentes con la misma
clave y payload devuelven el mismo ID. POST siempre confirma `en_cola`;
GET informa el estado actual. Otra URL completa
(incluida la firma) o fragmentacion con esa clave devuelve 409.
La deduplicacion solo dura hasta el TTL o el reinicio, dentro de este proceso.
Despues de reiniciar, GET devuelve 404 y NestJS debe reenviar el trabajo.
Despues de caducar, una clave puede crear un job nuevo. Puedes activar
SENTIMENT_API_SUPPORTS_IDEMPOTENCY=true teniendo en cuenta este alcance.
No hay deduplicacion entre instancias. Se conservan x-api-key, estados,
campos y escalas. NestJS puede consultar cada 30 s: 202 es pendiente, 200 es
resultado, 422 con estado error es fallo terminal y 429 es rechazo reintentable.

Si instalaste previamente el servicio separado, detenlo y deshabilitalo antes
de actualizar la API:

```bash
sudo systemctl disable --now emotion-analyzer-worker
```

Elimina JOB_DB_PATH del archivo de entorno; JOB_WAIT_TIMEOUT ya no se utiliza.
Conserva MAX_QUEUE_SIZE=20 y el resto de variables de .env.example. Luego:

```bash
sudo systemctl restart emotion-analyzer
curl http://127.0.0.1:8000/health
```

Al reiniciar se pierden trabajos y resultados en RAM. No existe recuperacion
tras caidas en esta arquitectura. La carga del modelo ocurre en el hilo interno;
la API acepta jobs mientras carga. Si falla, los pendientes pasan a error y las
nuevas solicitudes reciben 503 hasta corregir el problema y reiniciar.

## Limites y pruebas

El analisis no tiene timeout: puede durar 30 minutos sin una conexion abierta.
La descarga tiene socket timeout de 60 s y presupuesto de 600 s comprobado
entre lecturas; DOWNLOAD_TIMEOUT_SECONDS cambia este presupuesto.
Nginx conserva connect 60 s y send/read 900 s, pero POST y GET no esperan el
analisis. No hay archivos de configuracion de un balanceador en el repositorio.
TimeoutStartSec=900 de systemd no es un limite de tiempo por job.

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Las pruebas simulan 30 minutos sin esperarlos, bloquean la descarga y el analisis,
comprueban la respuesta inicial, consultas concurrentes, errores, autenticacion,
idempotencia en RAM, TTL, perdida tras reinicio y los campos del resultado.
Tambien verifican que un solo arranque de la API inicia los hilos necesarios.
No usan audios reales ni servicios externos.
