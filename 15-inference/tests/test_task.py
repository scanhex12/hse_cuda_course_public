import json
import random
import sys
import unittest
from pathlib import Path
import grpc
import numpy as np

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

import inference_pb2
import inference_pb2_grpc
import onnxruntime as ort

import client_benchmark
import task_env
from PIL import Image

_REF_PATH = _TASK_ROOT / 'tests' / 'reference_top5_image_png.json'
_ONNX_PATH = _TASK_ROOT / 'resnet50.onnx'


def _load_reference():
    with open(_REF_PATH, encoding='utf-8') as f:
        return json.load(f)


def _onnx_top5_for_png(onnx_path: Path, png_path: Path):
    img = Image.open(png_path).convert('RGB').resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = (x - mean) / std
    x = np.expand_dims(x, 0)

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    logits = sess.run(None, {inp.name: x})[0]
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    top = np.argsort(-logits)[:5]
    scores = logits[top]
    return top.tolist(), [float(s) for s in scores]


class TestReferenceMatchesOnnx(unittest.TestCase):
    def test_json_matches_onnx_resnet50(self):
        png = _TASK_ROOT / 'image.png'
        ref = _load_reference()
        self.assertEqual(ref.get('image'), 'image.png')
        idx, scores = _onnx_top5_for_png(_ONNX_PATH, png)
        self.assertEqual(idx, ref['indices'])
        np.testing.assert_allclose(scores, ref['scores'], rtol=1e-5, atol=1e-5)


class TestLoadFunction(unittest.TestCase):
    def test_load_positive_and_bounded(self):
        random.seed(42)
        params = {
            'A': 5.0,
            'B': 3.0,
            'C': 2.0,
            'D': 1.5,
            'f1': 0.1,
            'f2': 0.15,
            'f3': 0.05,
            'tau': 20.0,
            'base': 2.0,
            'noise': 0.0,
        }
        for t in (0.0, 0.5, 1.0, 10.0, 100.0):
            load = client_benchmark.load_function(t, params)
            self.assertGreaterEqual(load, 0.1)
            self.assertLess(load, 500.0)


class TestProtoContract(unittest.TestCase):
    def test_infer_request_rgb_size(self):
        w, h = 224, 224
        req = inference_pb2.InferRequest()
        req.width = w
        req.height = h
        req.rgb_image = bytes(w * h * 3)
        self.assertEqual(len(req.rgb_image), w * h * 3)

    def test_infer_response_roundtrip_top5(self):
        resp = inference_pb2.InferResponse()
        for i in range(5):
            p = resp.top5.add()
            p.class_index = i * 10
            p.score = float(5 - i)
        raw = resp.SerializeToString()
        parsed = inference_pb2.InferResponse()
        self.assertTrue(parsed.ParseFromString(raw))
        self.assertEqual(len(parsed.top5), 5)
        scores = [x.score for x in parsed.top5]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_stub_has_classify(self):
        self.assertTrue(hasattr(inference_pb2_grpc.InferenceServiceStub, '__init__'))


class TestAssets(unittest.TestCase):
    def test_onnx_exists_and_large(self):
        onnx = _TASK_ROOT / 'resnet50.onnx'
        self.assertTrue(onnx.is_file(), msg=f'missing {onnx}')
        self.assertGreater(onnx.stat().st_size, 1_000_000)

    def test_image_rgb_size_if_present(self):
        rgb = _TASK_ROOT / 'image.rgb'
        if not rgb.is_file():
            self.skipTest('image.rgb not in tree')
        self.assertEqual(rgb.stat().st_size, 224 * 224 * 3)


class TestImagesHelpers(unittest.TestCase):
    def test_random_image_shape(self):
        img = client_benchmark.generate_random_image()
        self.assertEqual(img.size, (224, 224))


@unittest.skipUnless(
    task_env.e2e_tests_enabled(),
    'set TRT15_E2E=1 (or INFERENCE_E2E=1) and run trt_server',
)
class TestGrpcE2E(unittest.TestCase):
    def test_classify_top5_matches_reference(self):
        endpoint = task_env.e2e_settings()
        png = _TASK_ROOT / 'image.png'
        if not png.is_file():
            self.skipTest('image.png missing')

        ref = _load_reference()
        img = Image.open(png).convert('RGB').resize((224, 224))
        data = np.array(img, dtype=np.uint8).tobytes()
        req = inference_pb2.InferRequest(width=224, height=224, rgb_image=data)

        channel = grpc.insecure_channel(f'{endpoint.host}:{endpoint.port}')
        try:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            resp = stub.Classify(req, timeout=120.0)
            self.assertEqual(len(resp.top5), 5)
            got_idx = [p.class_index for p in resp.top5]
            got_scores = [p.score for p in resp.top5]
            self.assertEqual(got_idx, ref['indices'], msg='')
            np.testing.assert_allclose(
                got_scores,
                ref['scores'],
                rtol=0.12,
                atol=0.4,
                err_msg='',
            )
        finally:
            channel.close()


if __name__ == '__main__':
    unittest.main()
