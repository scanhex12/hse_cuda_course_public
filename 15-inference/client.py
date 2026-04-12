#!/usr/bin/env python3

import sys
from pathlib import Path

import grpc
import numpy as np
from PIL import Image

import inference_pb2
import inference_pb2_grpc


def send_image(host: str, port: int, image_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.uint8)
    image_data = arr.tobytes()

    req = inference_pb2.InferRequest()
    req.width = 224
    req.height = 224
    req.rgb_image = image_data

    target = f"{host}:{port}"
    channel = grpc.insecure_channel(target)
    try:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        resp = stub.Classify(req, timeout=60.0)

        print("Top-5 predictions:")
        for i, pred in enumerate(resp.top5):
            print(f"{i + 1}. Class {pred.class_index}: {pred.score:.6f}")
    finally:
        channel.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <image_path> [host] [port]")
        sys.exit(1)

    image_path = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 50051

    if not Path(image_path).is_file():
        print(f"Error: not a file: {image_path}")
        sys.exit(1)

    send_image(host, port, image_path)
