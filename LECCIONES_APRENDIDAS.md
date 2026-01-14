# Lecciones Aprendidas y Recomendaciones Estratégicas

## Sistema de Detección de Aneurismas - Análisis Post-Mortem

---

## 1. RESUMEN EJECUTIVO

Este documento analiza las decisiones arquitectónicas, técnicas y de alcance tomadas durante el desarrollo del sistema de detección de aneurismas. El objetivo es identificar qué funcionó bien, qué podría mejorarse, y proporcionar recomendaciones para proyectos futuros similares.

**Veredicto general**: El proyecto logró sus objetivos educativos con una arquitectura sólida, pero acumuló deuda técnica en áreas específicas que deben abordarse antes de cualquier uso en producción.

---

## 2. LO QUE SE HIZO BIEN

### 2.1 Arquitectura de Pipeline con Fallbacks

```
ONNX Runtime → OpenCV DNN → Heurístico
     ↓              ↓            ↓
  Preferido     Fallback     Testing
```

**Decisión**: Implementar un patrón Factory con degradación graceful.

**Por qué fue acertado**:
- El proyecto sobrevive a cambios de entorno (Python 3.14 sin ONNX Runtime)
- Testing posible sin modelo real (fallback heurístico)
- Transparente para el consumidor (`create_inference_engine()`)

**Lección**: Diseñar sistemas con múltiples niveles de fallback desde el inicio. El costo adicional de implementación se paga con creces en mantenibilidad.

---

### 2.2 Separación de Entornos Python

```
Main Environment (3.14)     Training Environment (3.11)
├── Inferencia ONNX         ├── PyTorch + CUDA
├── FastAPI webapp          ├── timm (modelos)
└── OpenCV                  └── Exportación ONNX
```

**Por qué fue acertado**:
- Permite usar Python 3.14 (sin GIL) para producción
- Training con ecosistema PyTorch estable
- El artefacto de intercambio (ONNX) es portable

**Lección**: En proyectos ML, separar entornos de training e inferencia es casi siempre la decisión correcta. Las dependencias de training son pesadas y cambian rápido.

---

### 2.3 Configuración Centralizada (config.yaml)

```yaml
preprocessing:
  clahe:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
inference:
  confidence_threshold: 0.5
postprocessing:
  nms_threshold: 0.45
```

**Por qué fue acertado**:
- Un solo lugar para todos los hiperparámetros
- Fácil de versionar con git
- Reproducibilidad de experimentos

**Lección**: Nunca hardcodear hiperparámetros. Incluso en prototipos, el costo de externalizar configuración es mínimo y el beneficio enorme.

---

### 2.4 Dependency Injection en FastAPI

```python
async def create_analysis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
```

**Por qué fue acertado**:
- Testing trivial con `app.dependency_overrides`
- Separación clara de concerns
- Autenticación declarativa en cada endpoint

**Lección**: Frameworks con DI nativo (FastAPI, NestJS) reducen drásticamente el boilerplate de testing y mejoran la legibilidad.

---

### 2.5 Cascade Deletes y Relaciones ORM

```python
analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="SET NULL"))
```

**Por qué fue acertado**:
- `cascade="all, delete-orphan"`: Al borrar usuario, se borran sus análisis
- `ondelete="SET NULL"`: Al borrar sesión, los análisis quedan huérfanos pero no se pierden
- Integridad referencial automática

**Lección**: Definir comportamiento de borrado al crear el modelo, no después. Las migraciones de FK son dolorosas.

---

### 2.6 Singleton para AnalysisService

```python
class AnalysisService:
    _instance: Optional["AnalysisService"] = None
    _pipeline: Optional[DetectionPipeline] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Por qué fue acertado**:
- El modelo ONNX (~50MB) se carga una sola vez
- La sesión ONNX Runtime se reutiliza (optimizaciones de warmup)
- Thread-safe por GIL de Python

**Lección**: Para recursos caros (modelos ML, conexiones DB), el patrón Singleton o un pool es esencial. Crear instancias por request es un anti-patrón costoso.

---

## 3. DEUDA TÉCNICA IDENTIFICADA

### 3.1 CRÍTICA: Paths Hardcodeados

**Ubicación**: `webapp/api/admin.py:47`
```python
cwd="C:/Users/luado/Desktop/Claude_Projects/Analisis_Imagenes"
```

**Impacto**:
- El código no funciona en otra máquina
- Fallos silenciosos en CI/CD
- Imposible dockerizar

**Costo de corrección**: 30 minutos
```python
# Corrección
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
```

**Prioridad**: P0 - Bloquea despliegue

---

### 3.2 CRÍTICA: Modelo Sin Entrenar

**Estado actual**: MobileNetV3 con pesos ImageNet
**Problema**: El modelo no sabe qué es un aneurisma

```
Precisión actual en imágenes CT: ~50% (aleatorio)
Precisión necesaria para uso real: >95%
```

**Impacto**:
- El sistema es un placeholder educativo
- Cualquier resultado es estadísticamente sin valor
- Riesgo reputacional si se interpreta como funcional

**Costo de corrección**: 2-4 semanas
- Obtener ADAM Challenge dataset
- Entrenar con validación cruzada
- Validar con radiólogo

**Prioridad**: P0 - Define si el proyecto tiene valor real

---

### 3.3 ALTA: Validación de Entrada Débil

**Estado actual**:
```python
if ext not in {".jpg", ".jpeg", ".png"}:
    raise HTTPException(400, "Invalid file type")
```

**Problema**: Solo valida extensión, no contenido
- Un archivo .exe renombrado a .jpg pasa la validación
- Una foto de gato pasa la validación
- No verifica si es imagen médica (DICOM, dimensiones CT)

**Costo de corrección**: 1 semana
```python
def validate_medical_image(file_path: Path) -> ValidationResult:
    # 1. Magic bytes (es realmente una imagen?)
    # 2. Dimensiones razonables para CT (512x512 típico)
    # 3. Si es DICOM, validar Modality == "CT"
    # 4. Alertar si es foto de cámara (EXIF presente)
```

**Prioridad**: P1 - Afecta integridad de resultados

---

### 3.4 ALTA: Sin Rate Limiting

**Estado actual**: Cualquier usuario puede hacer requests ilimitados

**Impacto**:
- Vulnerable a DoS
- Un usuario puede monopolizar recursos
- Sin métricas de uso por usuario

**Costo de corrección**: 2 horas
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/analyses")
@limiter.limit("10/minute")
async def create_analysis(...):
```

**Prioridad**: P1 - Riesgo de disponibilidad

---

### 3.5 MEDIA: Sin Migraciones de Base de Datos

**Estado actual**: `Base.metadata.create_all()` en startup

**Problema**:
- No hay historial de cambios de schema
- Rollback imposible
- Pérdida de datos en cambios destructivos

**Costo de corrección**: 4 horas
```bash
pip install alembic
alembic init migrations
alembic revision --autogenerate -m "Initial schema"
```

**Prioridad**: P2 - Afecta mantenibilidad a largo plazo

---

### 3.6 MEDIA: Logging No Estructurado

**Estado actual**: `print()` y `logging.info()` con texto plano

**Problema**:
- Difícil de parsear para alertas
- No hay correlation IDs entre requests
- Sin métricas de latencia automáticas

**Costo de corrección**: 1 día
```python
import structlog
logger = structlog.get_logger()

logger.info("analysis_completed",
    user_id=user.id,
    has_aneurysm=result.has_aneurysm,
    latency_ms=timing.total
)
```

**Prioridad**: P2 - Afecta observabilidad

---

### 3.7 BAJA: Type Hints Incompletos

**Estado actual**:
```python
def run(self, image_path) -> Dict:  # Dict de qué?
```

**Problema**:
- IDE sin autocompletado útil
- MyPy no puede validar
- Documentación implícita perdida

**Corrección ideal**:
```python
from typing import TypedDict

class AnalysisResult(TypedDict):
    has_aneurysm: bool
    confidence: float
    detections: list[Detection]
    timings: Timings

def run(self, image_path: Path) -> AnalysisResult:
```

**Prioridad**: P3 - Mejora DX pero no afecta runtime

---

## 4. ANÁLISIS DE SELECCIÓN DE LIBRERÍAS

### 4.1 Decisiones Acertadas

| Librería | Decisión | Justificación |
|----------|----------|---------------|
| **FastAPI** | ✅ Excelente | Async, DI nativo, OpenAPI automático, tipado |
| **SQLAlchemy 2.0** | ✅ Excelente | ORM maduro, async support, migraciones con Alembic |
| **ONNX Runtime** | ✅ Excelente | Portable, optimizado, independiente de framework |
| **Pydantic** | ✅ Excelente | Validación robusta, serialización, settings |
| **pytest** | ✅ Excelente | Fixtures potentes, plugins, standard de facto |

### 4.2 Decisiones Cuestionables

| Librería | Decisión | Problema | Alternativa |
|----------|----------|----------|-------------|
| **SQLite** | ⚠️ Aceptable para dev | No escala, sin concurrencia real | PostgreSQL para prod |
| **Jinja2 + HTMX** | ⚠️ Aceptable | Limitado para UX compleja | React/Vue para dashboards interactivos |
| **PBKDF2** | ⚠️ Forzado | bcrypt mejor pero incompatible 3.14 | Argon2 cuando esté disponible |

### 4.3 Dependencias Faltantes

| Necesidad | Librería Recomendada | Prioridad |
|-----------|---------------------|-----------|
| Rate limiting | `slowapi` | P1 |
| Migraciones DB | `alembic` | P2 |
| Logging estructurado | `structlog` | P2 |
| Validación imágenes | `python-magic` + `pydicom` | P1 |
| Monitoring | `prometheus-fastapi-instrumentator` | P3 |

---

## 5. ANÁLISIS DE ARQUITECTURA

### 5.1 Arquitectura Actual

```
┌─────────────────────────────────────────────────────────────┐
│                     MONOLITO                                 │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Jinja2)  │  API (FastAPI)  │  ML (ONNX)          │
│        ↓                    ↓                ↓               │
│  Templates          │  Routes         │  Pipeline            │
│  Static files       │  Services       │  Inference           │
│  HTMX               │  Schemas        │  Postprocess         │
├─────────────────────────────────────────────────────────────┤
│                     SQLite                                   │
└─────────────────────────────────────────────────────────────┘
```

**Veredicto**: Apropiado para fase educativa/prototipo.

### 5.2 Problemas de la Arquitectura Actual

1. **Acoplamiento ML-Web**: El pipeline corre en el mismo proceso que la webapp
   - Un modelo lento bloquea requests HTTP
   - No hay cola de trabajo
   - No escala horizontalmente

2. **Base de datos embebida**: SQLite tiene límites de concurrencia
   - Write locks globales
   - No réplicas
   - Backup requiere parar el servicio

3. **Sin separación de servicios**: Todo en un proceso
   - Fallo en ML tumba la webapp
   - No se puede escalar ML independientemente
   - Deployment atómico (todo o nada)

### 5.3 Arquitectura Recomendada para Producción

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   NGINX     │────▶│   FastAPI   │────▶│  PostgreSQL │
│   (proxy)   │     │   (webapp)  │     │   (data)    │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Redis     │
                    │   (queue)   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             ┌─────────────┐ ┌─────────────┐
             │  ML Worker  │ │  ML Worker  │
             │   (ONNX)    │ │   (ONNX)    │
             └─────────────┘ └─────────────┘
```

**Beneficios**:
- ML workers escalan independientemente
- Cola absorbe picos de carga
- Webapp siempre responsive
- Workers pueden usar GPU dedicada

**Costo de migración**: 2-3 semanas

---

## 6. ANÁLISIS DE ALCANCE

### 6.1 Alcance Original vs Entregado

| Funcionalidad | Planeado | Entregado | Estado |
|---------------|----------|-----------|--------|
| Pipeline de detección | ✅ | ✅ | Completo (sin modelo real) |
| Webapp con auth | ✅ | ✅ | Completo |
| Gestión de sesiones | ✅ | ✅ | Completo |
| Visualizaciones | ✅ | ✅ | Completo |
| Panel admin | ✅ | ✅ | Completo |
| Tests | ✅ | ✅ | Cobertura ~70% |
| Módulo C++ | ✅ | ⚠️ | Existe pero no se usa |
| Modelo entrenado | ✅ | ❌ | Solo placeholder |
| Documentación | ✅ | ✅ | Completo |

### 6.2 Scope Creep Identificado

1. **Módulo C++**: Se construyó infraestructura completa (pybind11, CMake, headers) pero el preprocessing en Python resultó más rápido por evitar overhead de IPC.

   **Tiempo invertido**: ~1 semana
   **Valor entregado**: Marginal (solo educativo)
   **Lección**: Medir antes de optimizar. El cuello de botella era inferencia, no preprocessing.

2. **Panel de Admin con DB Explorer**: Funcionalidad avanzada para un proyecto educativo.

   **Tiempo invertido**: ~2 días
   **Valor entregado**: Alto para desarrollo, bajo para usuarios finales
   **Lección**: Herramientas de desarrollo son valiosas, pero deberían priorizarse correctamente.

### 6.3 Lo que Faltó

1. **Validación con datos reales**: El modelo no se entrenó con imágenes CT reales.
2. **Feedback de especialistas**: Sin revisión por radiólogos.
3. **Pruebas de carga**: Sin benchmark de concurrencia.
4. **Documentación de API para integradores**: OpenAPI existe pero sin ejemplos.

---

## 7. RECOMENDACIONES ESTRATÉGICAS

### 7.1 Corto Plazo (Sprint actual)

| Acción | Esfuerzo | Impacto | Responsable |
|--------|----------|---------|-------------|
| Eliminar paths hardcodeados | 30 min | Crítico | Dev |
| Agregar rate limiting | 2h | Alto | Dev |
| Validación básica de imágenes | 4h | Alto | Dev |
| Documentar variables de entorno | 1h | Medio | Dev |

### 7.2 Mediano Plazo (Próximo mes)

| Acción | Esfuerzo | Impacto | Responsable |
|--------|----------|---------|-------------|
| Entrenar modelo con ADAM dataset | 2-4 sem | Crítico | ML Engineer |
| Migrar a PostgreSQL | 1 sem | Alto | Dev |
| Implementar Alembic | 4h | Medio | Dev |
| CI/CD con GitHub Actions | 1 día | Alto | DevOps |
| Dockerizar aplicación | 2 días | Alto | DevOps |

### 7.3 Largo Plazo (Próximo trimestre)

| Acción | Esfuerzo | Impacto | Responsable |
|--------|----------|---------|-------------|
| Separar ML workers con Redis | 2-3 sem | Alto | Arquitecto |
| Validación clínica con radiólogos | 4+ sem | Crítico | PM + Clinical |
| Certificación HIPAA/GDPR | Variable | Crítico para prod | Legal + Dev |
| Soporte DICOM nativo | 2 sem | Alto | Dev |

---

## 8. MÉTRICAS DE ÉXITO

### 8.1 Técnicas

| Métrica | Actual | Objetivo | Cómo medir |
|---------|--------|----------|------------|
| Cobertura de tests | ~70% | >90% | `pytest --cov` |
| Latencia P95 | ~50ms | <100ms | Prometheus |
| Uptime | N/A | 99.9% | Monitoring |
| Bugs en prod | N/A | <1/semana | Issue tracker |

### 8.2 ML

| Métrica | Actual | Objetivo | Cómo medir |
|---------|--------|----------|------------|
| Precisión | ~50% | >95% | Test set holdout |
| Recall | ~50% | >98% | Test set holdout |
| F1 Score | ~50% | >96% | Test set holdout |
| Falsos negativos | Alto | <2% | Validación clínica |

### 8.3 Producto

| Métrica | Actual | Objetivo | Cómo medir |
|---------|--------|----------|------------|
| Tiempo análisis E2E | N/A | <5s | User timing |
| Tasa de error upload | N/A | <1% | Logs |
| Usuarios activos | 1 (dev) | N/A | Analytics |

---

## 9. CONCLUSIONES

### Lo que funcionó

1. **Arquitectura modular** con separación clara de concerns
2. **Configuración externalizada** que facilita experimentación
3. **Testing comprehensivo** con fixtures reutilizables
4. **Fallbacks automáticos** para resiliencia
5. **Documentación técnica** (CLAUDE.md) que acelera onboarding

### Lo que necesita trabajo

1. **Modelo ML** es placeholder sin valor predictivo real
2. **Seguridad** básica sin rate limiting ni HTTPS
3. **Validación de entrada** insuficiente para dominio médico
4. **Escalabilidad** limitada por arquitectura monolítica
5. **Observabilidad** básica sin métricas estructuradas

### Veredicto final

El proyecto cumple su propósito **educativo** exitosamente. Demuestra buenas prácticas de arquitectura de software y provee una base sólida para desarrollo futuro. Sin embargo, **no está listo para uso clínico** y requiere trabajo significativo en:

1. Entrenamiento del modelo con datos reales
2. Validación clínica con especialistas
3. Hardening de seguridad
4. Separación de servicios para escalabilidad

La deuda técnica acumulada es manejable (~2-3 semanas de trabajo enfocado) y no compromete la arquitectura fundamental.

---

*Documento generado: Enero 2026*
*Proyecto: Sistema de Detección de Aneurismas v1.0.0*
