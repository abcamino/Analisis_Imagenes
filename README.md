# 🧠 Aneurysm Detection System

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **⚠️ DISCLAIMER: This is an educational project. NOT intended for clinical use or medical diagnosis.**

Sistema educativo de detección de aneurismas cerebrales en imágenes CT usando deep learning. Incluye una aplicación web completa para gestionar análisis y visualizar resultados.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview)

## ✨ Características

- 🔬 **Detección con Deep Learning**: Pipeline basado en MobileNetV3 optimizado para imágenes CT cerebrales
- 🌐 **Aplicación Web Completa**: Dashboard interactivo con FastAPI + Jinja2 + HTMX
- 👤 **Sistema de Usuarios**: Autenticación con sesiones, roles de admin
- 📊 **Gestión de Análisis**: Historial, filtros, estadísticas, notas
- 🎨 **Visualizaciones**: Overlay de detecciones con mapas de calor
- 🧪 **Panel de Admin**: Ejecutor de tests y explorador de base de datos
- ⚡ **Alto Rendimiento**: ~25ms por imagen (preprocesamiento + inferencia)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Jinja2 + HTMX)                     │
│         login │ dashboard │ upload │ results │ admin           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│  /auth/*     → Autenticación (login, register, logout)          │
│  /api/*      → REST API (analyses, sessions, dashboard, admin)  │
│  /dashboard  → Páginas HTML renderizadas                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     SQLite      │  │ Detection       │  │ File Storage    │
│  Users, Analyses│  │ Pipeline (ONNX) │  │ uploads/        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Pipeline de Detección

```
Imagen CT (input)
      │
      ▼
┌──────────────────────────┐
│ 1. Preprocesamiento      │  ~4ms
│    • Grayscale + CLAHE   │
│    • Resize 224×224      │
│    • Normalización       │
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 2. Inferencia            │  ~20ms
│    • ONNX Runtime        │
│    • OpenCV DNN (backup) │
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 3. Post-procesamiento    │
│    • Softmax + NMS       │
│    • Filtrado confianza  │
└──────────────────────────┘
      │
      ▼
{has_aneurysm, confidence, detections[]}
```

## 🚀 Instalación

### Requisitos
- Python 3.11+ (recomendado 3.14)
- OpenCV 4.9+

### 1. Clonar repositorio
```bash
git clone https://github.com/abcamino/Analisis_Imagenes.git
cd Analisis_Imagenes
```

### 2. Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
pip install -r webapp_requirements.txt
```

### 4. Iniciar aplicación web
```bash
uvicorn webapp.main:app --reload --port 8000
```

Abrir http://localhost:8000 en el navegador.

## 💻 Uso

### Aplicación Web

1. **Registrarse**: Crear una cuenta en `/register`
2. **Dashboard**: Ver estadísticas y análisis recientes
3. **Subir imagen**: Arrastrar o seleccionar imagen CT en `/upload`
4. **Ver resultado**: Visualización con detecciones marcadas
5. **Historial**: Buscar y filtrar análisis anteriores en `/analyses`

### CLI (Línea de comandos)

```bash
# Analizar una imagen
python main.py --image data/raw/scan.jpg

# Analizar directorio completo
python main.py --dir data/raw/

# Con visualización
python main.py --image scan.jpg --visualize --save-viz

# Benchmark de rendimiento
python main.py --benchmark --image scan.jpg
```

## 🗄️ Base de Datos

### Esquema

| Tabla | Descripción |
|-------|-------------|
| `users` | Usuarios del sistema (username, email, password hash, is_admin) |
| `analyses` | Resultados de análisis (imagen, detecciones, confianza, tiempos) |
| `analysis_sessions` | Agrupación de análisis relacionados |
| `user_sessions` | Tokens de sesión HTTP |

### Panel de Admin

Los usuarios con `is_admin=True` tienen acceso a:
- `/admin/tests` - Ejecutar suite de tests y ver resultados
- `/admin/database` - Explorar tablas y datos

## 🧪 Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_api/test_auth_routes.py -v

# Con cobertura
pytest tests/ --cov=webapp --cov-report=html
```

**Cobertura actual**: 40+ tests unitarios cubriendo:
- Autenticación (registro, login, logout, sesiones)
- API de análisis (upload, CRUD, permisos)
- Sesiones de análisis (crear, finalizar, estadísticas)
- Modelos de base de datos (relaciones, constraints)
- Seguridad (hashing, tokens)

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| Backend | FastAPI 0.109+ |
| Database | SQLite + SQLAlchemy 2.0 |
| Frontend | Jinja2 + HTMX |
| ML Model | MobileNetV3 (ONNX) |
| Inference | ONNX Runtime / OpenCV DNN |
| Auth | PBKDF2-SHA256 + Session Cookies |
| Testing | pytest + httpx |

## 📋 TODO / Roadmap

### 🔴 Alta Prioridad

- [ ] **Entrenar modelo con datos reales**: El modelo actual usa pesos de ImageNet. Necesita entrenamiento con el dataset [ADAM Challenge](http://adam.isi.uu.nl/) para detección real de aneurismas
- [ ] **Validación de imágenes**: Verificar que las imágenes subidas son realmente CT cerebrales (no fotos genéricas)
- [ ] **Rate limiting**: Protección contra abuso de la API
- [ ] **HTTPS en producción**: Configurar certificados SSL

### 🟡 Media Prioridad

- [ ] **Recuperación de contraseña**: Flujo de reset por email
- [ ] **Exportar reportes**: Generar PDF con resultados del análisis
- [ ] **Comparación de análisis**: Ver dos análisis lado a lado
- [ ] **API de webhooks**: Notificar sistemas externos al completar análisis
- [ ] **Soporte DICOM**: Cargar archivos DICOM directamente (formato médico estándar)
- [ ] **Batch upload**: Subir múltiples imágenes a la vez

### 🟢 Baja Prioridad

- [ ] **Tema oscuro**: Toggle para dark mode en la UI
- [ ] **Internacionalización**: Soporte multi-idioma (EN/ES)
- [ ] **Logs estructurados**: Integrar con sistemas de logging (ELK, CloudWatch)
- [ ] **Docker**: Containerización para deployment fácil
- [ ] **CI/CD**: GitHub Actions para tests automáticos
- [ ] **Documentación API**: Swagger UI mejorado con ejemplos

### 🔧 Deuda Técnica

- [ ] **Migrar a Alembic**: Sistema de migraciones de base de datos
- [ ] **Cache de modelos**: Evitar recargar el modelo ONNX en cada request
- [ ] **Tests E2E**: Tests de integración con Playwright/Selenium
- [ ] **Typing completo**: Añadir type hints en todo el código

## 📁 Estructura del Proyecto

```
Analisis_Imagenes/
├── main.py                 # CLI entry point
├── config.yaml             # Configuración del pipeline
├── requirements.txt        # Dependencias core
├── webapp_requirements.txt # Dependencias web
│
├── src/                    # Core detection logic
│   ├── inference/          # Pipeline, ONNX inference
│   └── visualization/      # Overlay, reportes
│
├── webapp/                 # Aplicación web
│   ├── main.py             # FastAPI app
│   ├── config.py           # Settings
│   ├── database/           # SQLAlchemy models
│   ├── auth/               # Autenticación
│   ├── api/                # REST endpoints
│   ├── services/           # Business logic
│   ├── schemas/            # Pydantic models
│   ├── templates/          # Jinja2 HTML
│   └── static/             # CSS, JS
│
├── tests/                  # Test suite
│   ├── test_api/
│   ├── test_auth/
│   ├── test_database/
│   └── test_integration/
│
├── training/               # Scripts de entrenamiento
│   ├── train_model.py
│   ├── export_onnx.py
│   └── prepare_dataset.py
│
├── models/                 # Modelos entrenados
│   └── onnx/               # MobileNetV3 ONNX
│
└── data/                   # Datos de entrada
    ├── raw/                # Imágenes originales
    └── processed/          # Imágenes procesadas
```

## 🤝 Contribuir

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## ⚠️ Aviso Legal

Este software es únicamente para **fines educativos y de investigación**.

**NO debe utilizarse para:**
- Diagnóstico médico real
- Toma de decisiones clínicas
- Cualquier aplicación en pacientes reales

Los resultados del modelo no han sido validados clínicamente y pueden contener errores significativos. Siempre consulte con profesionales médicos cualificados para el diagnóstico de aneurismas cerebrales.

---

<p align="center">
  Desarrollado con ❤️ para aprendizaje de ML en medicina
  <br>
  <a href="https://github.com/abcamino/Analisis_Imagenes">GitHub</a> •
  <a href="https://github.com/abcamino/Analisis_Imagenes/issues">Issues</a>
</p>
