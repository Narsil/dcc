# Implementing Direct System Calls in DCC

This document describes how DCC implements system calls directly without linking to libc, enabling true cross-compilation and creating self-contained binaries.

## Overview

DCC generates binaries that make system calls directly using inline assembly, avoiding any dependency on system libraries. This approach enables:
- True cross-compilation without target system libraries
- Smaller, self-contained binaries
- Consistent behavior across platforms

## Architecture

### 1. Platform-Specific Syscall Numbers

Each OS/architecture combination has specific syscall numbers and calling conventions:

#### Linux x86_64
- Write syscall: 1
- Calling convention: syscall instruction
- Registers: rax=syscall#, rdi=fd, rsi=buf, rdx=count

#### Linux ARM64
- Write syscall: 64
- Calling convention: svc #0
- Registers: x8=syscall#, x0=fd, x1=buf, x2=count

#### macOS x86_64
- Write syscall: 0x2000004
- Calling convention: syscall instruction
- Registers: rax=syscall#, rdi=fd, rsi=buf, rdx=count

#### macOS ARM64
- Write syscall: 4
- Calling convention: svc #0x80
- Registers: x16=syscall#, x0=fd, x1=buf, x2=count

### 2. Implementation Strategy

The implementation uses LLVM inline assembly with AT&T syntax:

```zig
// Example for Linux x86_64
const asm_str = "movq $$1, %rax\nsyscall";
const constraints = "={rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}";
```

Key points:
- Use `movq` for 64-bit moves on x86_64
- `$$` prefix for immediate values in AT&T syntax
- `%` prefix for register names
- Proper clobber lists (~{rcx},~{r11} for Linux syscall)

### 3. Code Generation Flow

1. **Write Expression**: `write(io.stdout, "Hello")`
2. **Parser**: Creates AST node with write_expression
3. **Type Checker**: Validates handle and data types
4. **Code Generator**:
   - Computes string length (inline strlen loop)
   - Generates platform-specific syscall
   - Returns bytes written

### 4. Runtime Function Generation

DCC generates a `__dcc_write` function at compile time:

```llvm
define i64 @__dcc_write(i64 %fd, i8* %buf, i64 %count) {
entry:
  ; Platform-specific inline assembly
  %result = call i64 asm sideeffect "movq $$1, %rax\nsyscall",
    "={rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
    (i64 %fd, i8* %buf, i64 %count)
  ret i64 %result
}
```

## Adding New Syscalls

To add a new syscall:

1. **Define the syscall numbers** for each platform:
   ```zig
   const SYSCALL_READ = switch (target.os.tag) {
       .linux => switch (target.cpu.arch) {
           .x86_64 => 0,
           .aarch64 => 63,
           else => unreachable,
       },
       .macos => switch (target.cpu.arch) {
           .x86_64 => 0x2000003,
           .aarch64 => 3,
           else => unreachable,
       },
       else => unreachable,
   };
   ```

2. **Create the inline assembly**:
   - Follow platform calling conventions
   - Use correct instruction syntax
   - Include proper clobber lists

3. **Generate wrapper function**:
   - Create LLVM function type
   - Build inline assembly call
   - Handle return values

## Platform Considerations

### Linux
- Uses standard syscall numbers
- x86_64: syscall instruction clobbers rcx and r11
- ARM64: svc #0 with syscall number in x8

### macOS
- Syscall numbers have 0x2000000 offset on x86_64
- ARM64 uses x16 for syscall number (not x8)
- Different supervisor call: svc #0x80

### Cross-Compilation
- Target platform determined at compile time
- Inline assembly generated for target, not host
- No runtime platform detection needed

## Benefits

1. **No libc dependency**: Binaries work without system libraries
2. **Predictable behavior**: Direct syscalls avoid libc variations
3. **Smaller binaries**: No dynamic linking overhead
4. **True cross-compilation**: No need for target system headers/libs

## Limitations

1. **Limited syscall support**: Only implemented syscalls available
2. **Platform-specific**: Each OS/arch needs explicit support
3. **No libc conveniences**: Must implement higher-level functions

## Future Enhancements

1. **Syscall library**: Build reusable syscall wrappers
2. **Error handling**: Proper errno-style error returns
3. **More platforms**: Add Windows, BSDs, etc.
4. **Higher-level APIs**: Printf, file operations, etc.

## Example: Complete Write Implementation

```zig
fn generateWriteSyscall(self: *CodeGen, fd: Value, buf: Value, count: Value) !void {
    // 1. Generate strlen for string literals
    const len = try self.computeStrlen(buf);
    
    // 2. Create platform-specific syscall
    const asm_str = switch (self.target.os.tag) {
        .linux => "movq $$1, %rax\nsyscall",
        .macos => "movq $$0x2000004, %rax\nsyscall",
        else => unreachable,
    };
    
    // 3. Generate inline assembly call
    const result = LLVM.LLVMBuildInlineAsm(...);
    
    // 4. Use result (bytes written)
}
```

This approach makes DCC truly self-contained and enables reliable cross-compilation without external dependencies.