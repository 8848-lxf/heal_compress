#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t status = (expr);                                                 \
    if (status != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("cuda:") + cudaGetErrorString(status)); \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(expr)                                                       \
  do {                                                                           \
    cublasStatus_t status = (expr);                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                        \
      throw std::runtime_error("cublas_status:" + std::to_string(int(status))); \
    }                                                                            \
  } while (0)

struct Spec {
  cudaDataType_t operand;
  cublasComputeType_t compute;
  cudaDataType_t scale;
  cudaDataType_t output;
  size_t operandBytes;
  size_t outputBytes;
  std::string operandName;
  std::string computeName;
  std::string scaleName;
  std::string outputName;
};

static Spec specFor(std::string const& profile) {
  if (profile == "F32A32") {
    return {CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
            4, 4, "CUDA_R_32F", "CUBLAS_COMPUTE_32F", "CUDA_R_32F", "CUDA_R_32F"};
  }
  if (profile == "F16A32") {
    return {CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
            2, 4, "CUDA_R_16F", "CUBLAS_COMPUTE_32F", "CUDA_R_32F", "CUDA_R_32F"};
  }
  if (profile == "F16A16") {
    return {CUDA_R_16F, CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_16F,
            2, 2, "CUDA_R_16F", "CUBLAS_COMPUTE_16F", "CUDA_R_16F", "CUDA_R_16F"};
  }
  if (profile == "BF16A32") {
    return {CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F,
            2, 4, "CUDA_R_16BF", "CUBLAS_COMPUTE_32F", "CUDA_R_32F", "CUDA_R_32F"};
  }
  if (profile == "I8A32I") {
    return {CUDA_R_8I, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_32I,
            1, 4, "CUDA_R_8I", "CUBLAS_COMPUTE_32I", "CUDA_R_32I", "CUDA_R_32I"};
  }
  throw std::runtime_error("unknown_profile:" + profile);
}

static std::vector<char> readBytes(std::string const& path, size_t expected) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("input_open_failed:" + path);
  size_t size = static_cast<size_t>(input.tellg());
  if (size != expected) {
    throw std::runtime_error("input_size_mismatch:" + path + ":" +
                             std::to_string(size) + ":" + std::to_string(expected));
  }
  input.seekg(0);
  std::vector<char> data(size);
  input.read(data.data(), static_cast<std::streamsize>(size));
  return data;
}

static void writeBytes(std::string const& path, std::vector<char> const& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("output_open_failed:" + path);
  output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

static std::map<std::string, std::string> parseArgs(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::runtime_error("arguments_must_be_key_value_pairs");
    }
    args[std::string(argv[index]).substr(2)] = argv[index + 1];
  }
  for (auto const& key : {"a", "b", "out", "m", "n", "k", "batch", "profile"}) {
    if (!args.count(key)) throw std::runtime_error("missing_argument:" + std::string(key));
  }
  return args;
}

template <typename T>
static T algoAttribute(cublasLtMatmulAlgo_t const& algo,
                       cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t written = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), &written));
  return value;
}

int main(int argc, char** argv) {
  void* deviceA = nullptr;
  void* deviceB = nullptr;
  void* deviceD = nullptr;
  void* workspace = nullptr;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr,
                         layoutD = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  try {
    auto args = parseArgs(argc, argv);
    int64_t m = std::stoll(args.at("m"));
    int64_t n = std::stoll(args.at("n"));
    int64_t k = std::stoll(args.at("k"));
    int32_t batch = std::stoi(args.at("batch"));
    int warmup = args.count("warmup") ? std::stoi(args.at("warmup")) : 20;
    int iterations = args.count("iterations") ? std::stoi(args.at("iterations")) : 100;
    Spec spec = specFor(args.at("profile"));
    size_t aElements = static_cast<size_t>(batch) * m * k;
    size_t bElements = static_cast<size_t>(batch) * n * k;
    size_t dElements = static_cast<size_t>(batch) * m * n;
    auto hostA = readBytes(args.at("a"), aElements * spec.operandBytes);
    auto hostB = readBytes(args.at("b"), bElements * spec.operandBytes);
    std::vector<char> hostD(dElements * spec.outputBytes);
    CHECK_CUDA(cudaMalloc(&deviceA, hostA.size()));
    CHECK_CUDA(cudaMalloc(&deviceB, hostB.size()));
    CHECK_CUDA(cudaMalloc(&deviceD, hostD.size()));
    CHECK_CUDA(cudaMemcpy(deviceA, hostA.data(), hostA.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceB, hostB.data(), hostB.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(deviceD, 0, hostD.size()));
    CHECK_CUBLAS(cublasLtCreate(&handle));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operation, spec.compute, spec.scale));
    cublasOperation_t transposeA = CUBLAS_OP_N;
    cublasOperation_t transposeB = CUBLAS_OP_T;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA, &transposeA, sizeof(transposeA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSB, &transposeB, sizeof(transposeB)));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, spec.operand, m, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, spec.operand, n, k, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, spec.output, m, n, n));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, spec.output, m, n, n));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    for (auto layout : {layoutA, layoutB, layoutC, layoutD}) {
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    }
    int64_t strideA = m * k, strideB = n * k, strideD = m * n;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layoutD, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD)));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspaceSize = 64ULL * 1024ULL * 1024ULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, operation, layoutA, layoutB, layoutC, layoutD,
        preference, 1, &heuristic, &returned));
    if (returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("no_cublaslt_algorithm");
    }
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    float alphaF = 1.0F, betaF = 0.0F;
    __half alphaH = __float2half(1.0F), betaH = __float2half(0.0F);
    int32_t alphaI = 1, betaI = 0;
    void const* alpha = spec.scale == CUDA_R_16F ? static_cast<void const*>(&alphaH)
                        : spec.scale == CUDA_R_32I ? static_cast<void const*>(&alphaI)
                                                  : static_cast<void const*>(&alphaF);
    void const* beta = spec.scale == CUDA_R_16F ? static_cast<void const*>(&betaH)
                       : spec.scale == CUDA_R_32I ? static_cast<void const*>(&betaI)
                                                 : static_cast<void const*>(&betaF);
    auto launch = [&]() {
      CHECK_CUBLAS(cublasLtMatmul(
          handle, operation, alpha, deviceA, layoutA, deviceB, layoutB,
          beta, deviceD, layoutC, deviceD, layoutD, &heuristic.algo,
          workspace, workspaceSize, nullptr));
    };
    for (int index = 0; index < warmup; ++index) launch();
    CHECK_CUDA(cudaEventCreate(&begin));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(begin));
    for (int index = 0; index < iterations; ++index) launch();
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    float elapsedMs = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, begin, end));
    CHECK_CUDA(cudaMemcpy(hostD.data(), deviceD, hostD.size(), cudaMemcpyDeviceToHost));
    writeBytes(args.at("out"), hostD);
    int algoId = algoAttribute<int>(heuristic.algo, CUBLASLT_ALGO_CONFIG_ID);
    uint32_t tileId = algoAttribute<uint32_t>(heuristic.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
    uint32_t stagesId = algoAttribute<uint32_t>(heuristic.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
    uint32_t splitK = algoAttribute<uint32_t>(heuristic.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
    std::cout << "{\"status\":\"success\",\"operand_cuda_type\":\""
              << spec.operandName << "\",\"compute_type\":\"" << spec.computeName
              << "\",\"scale_type\":\"" << spec.scaleName
              << "\",\"output_cuda_type\":\"" << spec.outputName
              << "\",\"algorithm_id\":" << algoId << ",\"tile_id\":" << tileId
              << ",\"stages_id\":" << stagesId << ",\"split_k\":" << splitK
              << ",\"workspace_bytes\":" << heuristic.workspaceSize
              << ",\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
              << ",\"batch\":" << batch << ",\"warmup\":" << warmup
              << ",\"iterations\":" << iterations
              << ",\"mean_ms\":" << elapsedMs / iterations << "}" << std::endl;
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(workspace);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceD);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutD);
    cublasLtMatmulDescDestroy(operation);
    cublasLtDestroy(handle);
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "{\"status\":\"failed\",\"failure_reason\":\""
              << error.what() << "\"}" << std::endl;
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    if (workspace) cudaFree(workspace);
    if (deviceA) cudaFree(deviceA);
    if (deviceB) cudaFree(deviceB);
    if (deviceD) cudaFree(deviceD);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layoutA) cublasLtMatrixLayoutDestroy(layoutA);
    if (layoutB) cublasLtMatrixLayoutDestroy(layoutB);
    if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
    if (layoutD) cublasLtMatrixLayoutDestroy(layoutD);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (handle) cublasLtDestroy(handle);
    return 2;
  }
}
