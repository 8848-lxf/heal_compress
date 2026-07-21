#include "plugin_common.h"

#include <cuda_fp16.h>
#include <iostream>

namespace cobevt_mixed_accum
{
namespace
{
bool setLayoutAttribute(cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attribute,
    void const* value, size_t size)
{
    return cublasLtMatrixLayoutSetAttribute(layout, attribute, value, size) == CUBLAS_STATUS_SUCCESS;
}
}

int32_t launchMixedAccum(cublasLtHandle_t handle, void const* left, void const* right,
    void* output, void* workspace, size_t workspaceBytes,
    int32_t batch, int32_t m, int32_t n, int32_t k,
    bool transposeRight, float scale, nvinfer1::DataType outputType, cudaStream_t stream)
{
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t layoutA{}, layoutB{}, layoutC{};
    cublasLtMatmulPreference_t preference{};
    cublasLtMatmulHeuristicResult_t heuristic{};
    cudaDataType_t const outputCudaType
        = outputType == nvinfer1::DataType::kFLOAT ? CUDA_R_32F : CUDA_R_16F;
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = transposeRight ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    int returned = 0;

    auto cleanup = [&]() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
        if (layoutB) cublasLtMatrixLayoutDestroy(layoutB);
        if (layoutA) cublasLtMatrixLayoutDestroy(layoutA);
        if (operation) cublasLtMatmulDescDestroy(operation);
    };

    if (cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA))
            != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB))
            != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[QKMixedAccumPlugin] matmul descriptor setup failed" << std::endl;
        cleanup();
        return 1;
    }

    int32_t const bRows = transposeRight ? n : k;
    int32_t const bCols = transposeRight ? k : n;
    if (cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, m, k, k) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16F, bRows, bCols, bCols) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(&layoutC, outputCudaType, m, n, n) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[QKMixedAccumPlugin] matrix layout creation failed" << std::endl;
        cleanup();
        return 2;
    }
    int64_t strideA = static_cast<int64_t>(m) * k;
    int64_t strideB = static_cast<int64_t>(bRows) * bCols;
    int64_t strideC = static_cast<int64_t>(m) * n;
    if (!setLayoutAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order))
        || !setLayoutAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order))
        || !setLayoutAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order))
        || !setLayoutAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch))
        || !setLayoutAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch))
        || !setLayoutAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch))
        || !setLayoutAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA))
        || !setLayoutAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB))
        || !setLayoutAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)))
    {
        std::cerr << "[QKMixedAccumPlugin] matrix layout attribute setup failed" << std::endl;
        cleanup();
        return 3;
    }
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes))
            != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulAlgoGetHeuristic(handle, operation, layoutA, layoutB, layoutC,
            layoutC, preference, 1, &heuristic, &returned) != CUBLAS_STATUS_SUCCESS
        || returned != 1)
    {
        std::cerr << "[QKMixedAccumPlugin] heuristic selection failed returned=" << returned
                  << " heuristic_state=" << static_cast<int32_t>(heuristic.state) << std::endl;
        cleanup();
        return 4;
    }
    float const beta = 0.0F;
    cublasStatus_t const status = cublasLtMatmul(handle, operation, &scale, left, layoutA,
        right, layoutB, &beta, output, layoutC, output, layoutC,
        &heuristic.algo, workspace, workspaceBytes, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "[QKMixedAccumPlugin] cublasLtMatmul failed status="
                  << static_cast<int32_t>(status) << " workspace=" << workspaceBytes
                  << " heuristic_workspace=" << heuristic.workspaceSize << std::endl;
    }
    cleanup();
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 5;
}

} // namespace cobevt_mixed_accum
