#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/CAPI/Pass.h"

// C wrapper for ConvertGpuOpsToNVVMOps with options
extern "C" {

// Mirror the C++ options struct for C API
struct MlirConvertGpuOpsToNVVMOpsOptions {
    unsigned indexBitwidth;
    bool hasRedux;
    bool useBarePtrCallConv;
};

// Create pass with options
MLIR_CAPI_EXPORTED MlirPass mlirCreateConversionConvertGpuOpsToNVVMOpsWithOptions(
    struct MlirConvertGpuOpsToNVVMOpsOptions options) {
    
    // Convert C options to C++ options
    mlir::ConvertGpuOpsToNVVMOpsOptions cppOptions;
    cppOptions.indexBitwidth = options.indexBitwidth;
    cppOptions.hasRedux = options.hasRedux;
    cppOptions.useBarePtrCallConv = options.useBarePtrCallConv;
    
    // Create the pass with options
    return wrap(mlir::createConvertGpuOpsToNVVMOps(cppOptions).release());
}

// Convenience function to create pass with bare pointer call convention
MLIR_CAPI_EXPORTED MlirPass mlirCreateConversionConvertGpuOpsToNVVMOpsWithBarePtr(void) {
    mlir::ConvertGpuOpsToNVVMOpsOptions options;
    options.indexBitwidth = 0;
    options.hasRedux = false;
    options.useBarePtrCallConv = true;  // Enable bare pointer call convention
    
    return wrap(mlir::createConvertGpuOpsToNVVMOps(options).release());
}

} // extern "C" 