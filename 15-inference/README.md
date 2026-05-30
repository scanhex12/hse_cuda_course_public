# TensorRT: gRPC-сервер inference

В этой задаче вам необходимо написать gPRC сервис, который развернет сверточную сетку. Сама сетка задана в ONNX формате и находится в файле resnet50.onnx.

## Указания

1. В этой задаче будет проверяться корректность на некоторой картинке + бенчмарк.
2. Бенчмарк состоит из параллельных запросов с непостоянной нагрузкой.
3. Чтобы хорошо справляться с неравномерной нагрузкой, попробуйте поэкспериментировать с батчингом запросов.
4. Попробуйте потрогать квантизацию в TensorRT (например FP16/INT8 в настройках билдера).
5. Используйте стримы и асинхронные копирования
6. CUDA graphs. Заметьте, что пайплайн здесь от картинке к картинке не меняется, а значит можно использовать это чудо.

## Как работать с TensorRT

Вся документация по C++ API TensorRT находится тут https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html

# Inference service (gRPC)

Контракт задаётся в [`inference.proto`](inference.proto): сообщения **Protobuf** и сервис **`InferenceService`** с RPC **`Classify`**. Транспорт — **gRPC** поверх HTTP/2 (порт по умолчанию в шаблоне: **50051**).

## Сервис

| RPC | Запрос | Ответ |
|-----|--------|--------|
| `Classify` | `InferRequest` | `InferResponse` |

Полный путь метода в gRPC: `trt.inference.v1.InferenceService/Classify`.

## Сообщения

### `InferRequest`

| Поле | Тип | Смысл |
|------|-----|-------|
| `width` | `int32` | ширина изображения |
| `height` | `int32` | высота изображения |
| `rgb_image` | `bytes` | сырые RGB: `width * height * 3` байт, построчно (row-major) |

### `InferResponse`

| Поле | Тип | Смысл |
|------|-----|-------|
| `top5` | `repeated ClassPrediction` | топ-5 по убыванию `score` (ожидается 5 элементов) |

### `ClassPrediction`

| Поле | Тип | Смысл |
|------|-----|-------|
| `class_index` | `int32` | индекс класса |
| `score` | `float` | score / logit |

## Setup (TensorRT)

На VM TensorRT уже установлен в общем каталоге (`/opt/tensorrt/TensorRT-10.4.0.26`):

Локально в `15-inference/TensorRT-*` распаковывать не нужно.
