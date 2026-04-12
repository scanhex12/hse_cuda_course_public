# -*- coding: utf-8 -*-
"""gRPC stubs for inference.proto. Перегенерация:
  pip install grpcio-tools
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. inference.proto
"""
import grpc

import inference_pb2 as inference__pb2


class InferenceServiceStub(object):
    def __init__(self, channel):
        self.Classify = channel.unary_unary(
            "/trt.inference.v1.InferenceService/Classify",
            request_serializer=inference__pb2.InferRequest.SerializeToString,
            response_deserializer=inference__pb2.InferResponse.FromString,
        )


class InferenceServiceServicer(object):
    def Classify(self, request, context):
        raise NotImplementedError("Method not implemented!")


def add_InferenceServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Classify": grpc.unary_unary_rpc_method_handler(
            servicer.Classify,
            request_deserializer=inference__pb2.InferRequest.FromString,
            response_serializer=inference__pb2.InferResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "trt.inference.v1.InferenceService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
