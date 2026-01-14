# Entrenamiento del Modelo de Detección de Aneurismas

## Requisitos

El entrenamiento requiere **Python 3.11 o 3.12** debido a compatibilidad con PyTorch y timm.

## Configuración del Entorno de Entrenamiento

```bash
# Crear entorno virtual con Python 3.11
py -3.11 -m venv training_env
training_env\Scripts\activate

# Instalar dependencias de entrenamiento
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm onnx onnxruntime
pip install opencv-python-headless numpy matplotlib
pip install albumentations scikit-learn tqdm pyyaml
```

## Datasets Disponibles

### 1. ADAM Challenge (Recomendado)
- **URL**: https://adam.isi.uu.nl/
- **Contenido**: 113 casos TOF-MRA con anotaciones de aneurismas
- **Formato**: NIfTI (.nii.gz)
- **Requiere**: Registro y aceptación de términos

### 2. RSNA Intracranial Hemorrhage Detection
- **URL**: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
- **Contenido**: ~750,000 imágenes CT
- **Nota**: Hemorragias, no específicamente aneurismas

### 3. BraTS (Brain Tumor Segmentation)
- **URL**: https://www.synapse.org/brats
- **Contenido**: MRI cerebrales con tumores
- **Útil para**: Transfer learning en imágenes cerebrales

## Entrenamiento

```bash
# Activar entorno
training_env\Scripts\activate

# Entrenar modelo
python train_model.py --dataset data/adam --epochs 50 --output models/onnx/

# Exportar a ONNX
python export_onnx.py --checkpoint best_model.pth --output models/onnx/mobilenetv3_aneurysm.onnx
```

## Uso del Modelo Entrenado

Una vez exportado a ONNX, el modelo puede usarse con Python 3.14:

```python
# En el proyecto principal (Python 3.14)
from src.inference.pipeline import DetectionPipeline

pipeline = DetectionPipeline()
result = pipeline.run("imagen.jpg")
print(result)
```
