#include "lld/Common/Driver.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <vector>

// Declare that we have all the lld drivers
LLD_HAS_DRIVER(elf)
LLD_HAS_DRIVER(macho)
LLD_HAS_DRIVER(coff)
LLD_HAS_DRIVER(mingw)
LLD_HAS_DRIVER(wasm)

extern "C" {

// C wrapper for lldMain - handles all target types
int lld_main(const char* const* args, int argc) {
    std::vector<const char*> arg_vec;
    for (int i = 0; i < argc; i++) {
        arg_vec.push_back(args[i]);
    }
    
    llvm::raw_fd_ostream stdout_stream(1, false);
    llvm::raw_fd_ostream stderr_stream(2, false);
    
    // Include all available drivers - lldMain will pick the right one
    const lld::DriverDef drivers[] = LLD_ALL_DRIVERS;
    
    auto result = lld::lldMain(arg_vec, stdout_stream, stderr_stream, drivers);
    return result.retCode;
}

} 