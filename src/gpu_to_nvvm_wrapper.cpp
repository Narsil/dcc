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

// Mirror the C++ options struct for ConvertFuncToLLVM
struct MlirConvertFuncToLLVMPassOptions {
    bool useBarePtrCallConv;
    unsigned indexBitwidth;
};

// Create ConvertFuncToLLVM pass with options
MLIR_CAPI_EXPORTED MlirPass mlirCreateConversionConvertFuncToLLVMPassWithOptions(
    struct MlirConvertFuncToLLVMPassOptions options) {
    
    // Convert C options to C++ options
    mlir::ConvertFuncToLLVMPassOptions cppOptions;
    cppOptions.useBarePtrCallConv = options.useBarePtrCallConv;
    cppOptions.indexBitwidth = options.indexBitwidth;
    
    // Create the pass with options
    return wrap(mlir::createConvertFuncToLLVMPass(cppOptions).release());
}

// Convenience function to create ConvertFuncToLLVM pass with bare pointer call convention
MLIR_CAPI_EXPORTED MlirPass mlirCreateConversionConvertFuncToLLVMPassWithBarePtr(void) {
    mlir::ConvertFuncToLLVMPassOptions options;
    options.useBarePtrCallConv = true;  // Enable bare pointer call convention
    options.indexBitwidth = 0;
    
    return wrap(mlir::createConvertFuncToLLVMPass(options).release());
}

} // extern "C" 