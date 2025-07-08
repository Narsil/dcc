pub const LLVM = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/Support.h");
    @cInclude("llvm-c/BitReader.h");
    @cInclude("llvm-c/Object.h");
});