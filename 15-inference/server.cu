#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_helpers.h>
#include <cuda_runtime.h>

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <csignal>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference.grpc.pb.h"
#include "inference.pb.h"

using namespace nvinfer1;

template <typename T> using TrtUniquePtr = std::unique_ptr<T>;

struct InferenceEngine {
    TrtUniquePtr<IRuntime> runtime;
    TrtUniquePtr<ICudaEngine> engine;
    std::string input_name;
    std::string output_name;

    InferenceEngine() = default;
};

struct ThreadContext {
    TrtUniquePtr<IExecutionContext> context;
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaStream_t stream{};

    ThreadContext() { cudaStreamCreate(&stream); }

    ~ThreadContext() {
        if (d_input) {
            cudaFree(d_input);
        }
        if (d_output) {
            cudaFree(d_output);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};

TrtUniquePtr<IHostMemory> buildSerializedEngine(const std::string &onnx_file,
                                                IBuilder *builder) {
    (void)onnx_file;
    (void)builder;
    throw std::runtime_error("TODO: buildSerializedEngine (см. README)");
}

void preprocess(const std::vector<uint8_t> &input_image,
                std::vector<float> &output, int width, int height) {
    (void)input_image;
    (void)output;
    (void)width;
    (void)height;
    throw std::runtime_error(
        "TODO: preprocess — RGB HWC → NCHW float, ImageNet mean/std");
}

std::unique_ptr<InferenceEngine> loadEngine(const std::string &onnx_file) {
    (void)onnx_file;
    return std::make_unique<InferenceEngine>();
}

std::unique_ptr<ThreadContext> createThreadContext(InferenceEngine *engine) {
    auto ctx = std::make_unique<ThreadContext>();
    if (!engine || !engine->engine) {
        return ctx;
    }
    (void)engine;
    return ctx;
}

std::vector<std::pair<int, float>>
inferImage(InferenceEngine *engine, ThreadContext *ctx,
           const std::vector<uint8_t> &image_data, int width, int height) {
    (void)engine;
    (void)ctx;
    (void)image_data;
    (void)width;
    (void)height;
    throw std::runtime_error(
        "TODO: inferImage — preprocess → H2D → enqueueV3 → D2H → top-5");
}

class InferenceServiceImpl final
    : public trt::inference::v1::InferenceService::Service {
  public:
    InferenceServiceImpl(InferenceEngine *engine, ThreadContext *ctx,
                         std::mutex *infer_mutex)
        : engine_(engine), ctx_(ctx), mutex_(infer_mutex) {}

    grpc::Status
    Classify(grpc::ServerContext * /*context*/,
             const trt::inference::v1::InferRequest *request,
             trt::inference::v1::InferResponse *response) override {
        const int width = request->width();
        const int height = request->height();
        if (width <= 0 || height <= 0 || width > 10000 || height > 10000) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "invalid width/height");
        }
        const size_t expected =
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        if (static_cast<size_t>(request->rgb_image().size()) != expected) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "rgb_image size mismatch");
        }

        std::vector<uint8_t> image(request->rgb_image().begin(),
                                   request->rgb_image().end());
        std::vector<std::pair<int, float>> top5;
        try {
            std::lock_guard<std::mutex> lock(*mutex_);
            top5 = inferImage(engine_, ctx_, image, width, height);
        } catch (const std::exception &e) {
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        } catch (...) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "inference failed");
        }

        for (const auto &p : top5) {
            trt::inference::v1::ClassPrediction *cp = response->add_top5();
            cp->set_class_index(p.first);
            cp->set_score(p.second);
        }
        return grpc::Status::OK;
    }

  private:
    InferenceEngine *engine_;
    ThreadContext *ctx_;
    std::mutex *mutex_;
};

namespace {
grpc::Server *g_server = nullptr;

void handleShutdownSignal(int /*signum*/) {
    if (g_server != nullptr) {
        g_server->Shutdown();
    }
}
} // namespace

void runServer(int port, const std::string &onnx_file) {
    std::cerr << "Loading engine..." << std::endl;
    auto engine = loadEngine(onnx_file);
    if (!engine->engine) {
        throw std::runtime_error("Engine is null after loadEngine");
    }

    auto ctx = createThreadContext(engine.get());
    std::mutex infer_mutex;

    InferenceServiceImpl service(engine.get(), ctx.get(), &infer_mutex);

    const std::string addr = "0.0.0.0:" + std::to_string(port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        throw std::runtime_error("gRPC: BuildAndStart failed");
    }

    std::cerr << "gRPC InferenceService listening on " << addr << std::endl;
    g_server = server.get();
    std::signal(SIGINT, handleShutdownSignal);
    std::signal(SIGTERM, handleShutdownSignal);
    server->Wait();
    g_server = nullptr;
}

int main(int argc, char *argv[]) {
    int port = 50051;
    std::string onnx_file = "resnet50.onnx";

    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    if (argc > 2) {
        onnx_file = argv[2];
    }

    if (!std::ifstream(onnx_file).good()) {
        onnx_file = "../15-inference/resnet50.onnx";
    }
    if (!std::ifstream(onnx_file).good()) {
        onnx_file = "../../15-inference/resnet50.onnx";
    }

    try {
        runServer(port, onnx_file);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
