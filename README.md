# Laboratorio #2 — Redes Neuronales y Aprendizaje Profundo

**Universidad EAFIT — Escuela de Ciencias Aplicadas e Ingeniería**  
Curso de Deep Learning · Módulo II · Valor: 15%

---

## Integrantes

| Nombre | 
|--------|
| Nicolás Ospina Torres |
| Alejandro Garcés Ramírez |
| Jean Carlo Londoño Ocampo |

---

## Estructura del Repositorio

```
├── ANJ_LAB2A_TLF.ipynb              ← Notebook completo Parte A (Transfer Learning)
├── ANJ_LAB2B_TLF.ipynb              ← Notebook completo Parte B (Segmentación con U-Net)
├── Laboratorio__2-DL.pdf           ← Enunciado original del laboratorio
└── README.md
```

---

## Parte A — Transfer Learning con PyTorch

Implementación de tres estrategias de transfer learning sobre el dataset **Cats vs. Dogs** (~25,000 imágenes):

| Modelo | Arquitectura | Estrategia |
|--------|-------------|------------|
| M1 | VGG-16 | Extractor de features fijo + clasificador lineal |
| M2 | VGG-16 | Clasificador reemplazado → congelado → fine-tuning completo |
| M3 | ResNet-18 | Adaptación propuesta por el equipo con congelamiento + entrenamiento |

### Contenido del notebook

- **Actividad 1:** Dataset, limpieza de imágenes corruptas y pipeline de preprocesamiento (Resize → CenterCrop → ToTensor → Normalize con estadísticas de ImageNet).
- **Actividad 2:** Carga de VGG-16 preentrenado (API moderna `weights=`), verificación de inferencia y extracción de features para 800 imágenes.
- **Actividad 3:** Entrenamiento de clasificador lineal sobre features precomputados (5 épocas). Análisis de CrossEntropyLoss vs NLLLoss.
- **Actividad 4:** Transfer learning end-to-end con VGG-16: reemplazo de clasificador, congelamiento de backbone, entrenamiento (1 época).
- **Actividad 5:** Fine-tuning completo: descongelamiento del backbone con lr=1e-4 (1 época).
- **Actividad 6:** Adaptación de ResNet-18 a Cats vs. Dogs con congelamiento (2 épocas). Comparación de velocidad y accuracy con VGG-16.
- **Preguntas de investigación:** Normalización ImageNet, data augmentation (implementado con RandomCrop, HorizontalFlip, ColorJitter), CosineAnnealingLR, límites del transfer learning, BatchNorm en ResNet vs VGG.

### Tecnologías

- PyTorch + torchvision
- TensorBoard (logging de experimentos)
- torchinfo (inspección de arquitecturas)

---

## Parte B — Segmentación Semántica con U-Net y Transfer Learning

*(Por completar — entrega: viernes 27 de marzo de 2026)*

Dataset: **Oxford-IIIT Pet Dataset** (segmentación binaria: animal vs. fondo).

| Modelo | Arquitectura | Estrategia |
|--------|-------------|------------|
| M1 | U-Net | Encoder aleatorio, línea base |
| M2 | ResNetUNet | Encoder congelado (solo decoder) |
| M3 | ResNetUNet | Fine-tuning completo (encoder + decoder) |

---

## Cómo ejecutar

1. Abrir el notebook en **Google Colab** o un entorno local con GPU.
2. Ejecutar las celdas en orden secuencial.
3. El dataset se descarga automáticamente desde Microsoft.
4. Los logs de TensorBoard se guardan en `runs/`.

```bash
# Para visualizar experimentos localmente:
tensorboard --logdir=runs --port=6006
```

---

## Fechas de entrega

| Parte | Fecha límite |
|-------|-------------|
| Parte A | Lunes 23 de marzo de 2026, 11:59 p.m. |
| Parte B | Viernes 27 de marzo de 2026, 11:59 p.m. |