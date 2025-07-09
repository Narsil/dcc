// This file serves as the main interface for the codegen module
// Re-export the main types from core
pub const CodeGen = @import("codegen/core.zig").CodeGen;
pub const CodeGenError = @import("codegen/core.zig").CodeGenError;
