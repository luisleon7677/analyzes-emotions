# Análisis emocional de audio

Aplicación de escritorio en Python que analiza audios en **español** por fragmentos temporales y muestra una gráfica de emociones a lo largo del tiempo (triste → alegre), con reproductor sincronizado.

Modelo: [UMUTeam/w2v-bert-emotion-es](https://huggingface.co/UMUTeam/w2v-bert-emotion-es)

---

## Qué subir a producción (repositorio / servidor)

| Archivo / carpeta | ¿Subir? | Motivo |
|-------------------|---------|--------|
| `app.py` | Sí | Interfaz gráfica (punto de entrada) |
| `analyzer.py` | Sí | Lógica de análisis por fragmentos |
| `umu_model.py` | Sí | Arquitectura del modelo UMUTeam |
| `requirements.txt` | Sí | Dependencias |
| `README.md` | Sí | Documentación |
| `.gitignore` | Sí | Evita subir archivos innecesarios |
| `.env.example` | Sí | Plantilla de variables (sin secretos) |
| `audio/.gitkeep` | Sí | Mantiene la carpeta `audio/` vacía |
| `emotion.py` | Opcional | Script de prueba antiguo (no necesario para la app) |

## Qué NO subir

| Elemento | Motivo |
|----------|--------|
| `venv/` o `.venv/` | Entorno virtual; se crea en cada máquina con `pip install` |
| `__pycache__/` y `*.pyc` | Caché de Python |
| `audio/*.wav`, `audio/*.ogg`, etc. | Audios de usuario o de prueba |
| `.env` | Puede contener tokens o secretos |
| Modelos descargados (`*.safetensors`, `models/`) | Se descargan solos desde Hugging Face al primer uso |
| Caché de Hugging Face (`~/.cache/huggingface`) | Se regenera automáticamente |
| Archivos de IDE (`.vscode/`, `.idea/`) | Configuración local del editor |

> **Nota:** El modelo (~600 MB) **no va en el repositorio**. La primera vez que ejecutes la app se descargará desde Hugging Face.

---

## Requisitos del sistema

- **Python** 3.10 o superior (probado con 3.13)
- **Windows**, **Linux** o **macOS**
- **RAM:** mínimo 4 GB (recomendado 8 GB)
- **Disco:** ~2 GB libres (dependencias + modelo)
- **GPU:** opcional (acelera el análisis; funciona en CPU)
- **Audio:** salida de sonido para el reproductor integrado

---

## Instalación

### 1. Clonar o copiar el proyecto

```bash
git clone <url-del-repositorio>
cd test-voice-model
```

Si no usas Git, copia solo los archivos listados en la tabla “Qué subir”.

### 2. Crear entorno virtual

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

En Windows, si `sounddevice` falla al reproducir audio, instala también [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### 4. (Opcional) Token de Hugging Face

Para evitar límites de descarga en entornos con muchas instalaciones:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Edita `.env` y añade tu token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

O exporta la variable antes de ejecutar:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxx"
```

---

## Uso

Con el entorno virtual activado:

```bash
python app.py
```

### Flujo en la interfaz

1. Espera a que cargue el modelo (primera vez puede tardar varios minutos).
2. Pulsa **Agregar audio** y selecciona un archivo (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`).
3. Ajusta el **tamaño de fragmento** si quieres (por defecto 3 s).
4. Pulsa **Procesar** para generar la gráfica.
5. Usa el reproductor inferior; la **línea blanca** marca la posición actual en el tiempo.

---

## Estructura del proyecto

```
test-voice-model/
├── app.py              # Interfaz gráfica (ejecutar este archivo)
├── analyzer.py         # Análisis por fragmentos
├── umu_model.py        # Clase del modelo Wav2Vec2-BERT (UMUTeam)
├── emotion.py          # Script de prueba legacy (opcional)
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
└── audio/              # Carpeta local para audios (no se sube al repo)
    └── .gitkeep
```

---

## Despliegue en producción

Esta app es de **escritorio** (Tkinter + CustomTkinter). No es una API web.

### Opciones recomendadas

1. **Repositorio Git**  
   Sube solo el código fuente. Cada usuario o máquina instala con los pasos de arriba.

2. **Máquina / PC de trabajo**  
   Clona el repo, crea `venv`, instala dependencias y ejecuta `python app.py`.

3. **Empaquetado como ejecutable (opcional)**  
   Puedes usar PyInstaller más adelante; el ejecutable seguirá necesitando descargar el modelo la primera vez (o incluirlo manualmente, ~600 MB).

### Checklist antes de publicar

- [ ] No hay archivos en `audio/` con datos reales de usuarios
- [ ] No hay `.env` con tokens en el repositorio
- [ ] No hay carpeta `venv/` en el commit
- [ ] `requirements.txt` está actualizado
- [ ] Probaste `python app.py` en una instalación limpia

---

## Emociones detectadas

| Etiqueta del modelo | Español en la UI | Color en gráfica |
|---------------------|------------------|------------------|
| `sadness` | Triste | Azul |
| `fear` | Miedo | Morado |
| `disgust` | Disgusto | Turquesa |
| `anger` | Enojo | Rojo |
| `neutral` | Neutral | Amarillo |
| `joy` | Alegre | Verde |

---

## Solución de problemas

| Problema | Posible solución |
|----------|------------------|
| Error al cargar el modelo | Comprueba conexión a internet y espacio en disco |
| Descarga muy lenta | Configura `HF_TOKEN` en `.env` |
| No suena el audio | Revisa dispositivo de salida y drivers; en Windows instala VC++ Redistributable |
| `ModuleNotFoundError` | Activa el `venv` y ejecuta `pip install -r requirements.txt` |
| La app va lenta | Usa fragmentos más grandes (p. ej. 5 s) o GPU si está disponible |

---

## Licencia del modelo

El modelo [UMUTeam/w2v-bert-emotion-es](https://huggingface.co/UMUTeam/w2v-bert-emotion-es) tiene su propia licencia en Hugging Face. Revisa las condiciones de uso antes de un despliegue comercial.
