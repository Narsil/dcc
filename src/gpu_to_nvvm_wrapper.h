#ifndef GPU_TO_NVVM_WRAPPER_H
#define GPU_TO_NVVM_WRAPPER_H

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mirror the C++ options struct for C API
struct MlirConvertGpuOpsToNVVMOpsOptions {
    unsigned indexBitwidth;
    bool hasRedux;
    bool useBarePtrCallConv;
};

// Create pass with full options control
MLIR_CAPI_EXPORTED MlirPass mlirCreateConversionConvertGpuOpsToNVVMOpsWithOptions(
    struct MlirConvertGpuOpsToNVVMOpsOptions options);

// Convenience function to create pass with bare pointer call convention enabled
MLIR_CAPI_EXPORTED MlirPass mlirCreateConversionConvertGpuOpsToNVVMOpsWithBarePtr(void);

#ifdef __cplusplus
}
#endif

#endif // GPU_TO_NVVM_WRAPPER_H 