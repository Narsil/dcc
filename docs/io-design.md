# IO Design in DCC

This document describes the design philosophy and implementation of IO operations in DCC, specifically the `write` and `read` keywords.

## Design Philosophy

DCC treats all side effects as explicit operations through dedicated keywords. The `write` and `read` keywords are the only way to perform IO operations, making programs easier to reason about and optimize.

### Core Principles

1. **Explicit Side Effects**: All IO operations must use `write` or `read`
2. **Function-like Syntax**: Despite being keywords, they use familiar function call syntax
3. **No Hidden State**: No global file handles or implicit streams
4. **Type Safety**: Compile-time checking of handle types and data formats

## The IO Module

Until DCC has a proper module system, IO handles are accessed through a virtual `io` namespace:

```toy
write(io.stdout, "Hello, World!\n");
write(io.stderr, "Error message\n");
let input = read(io.stdin, [256]u8);
```

### Standard Handles

- `io.stdout` - Standard output (fd 1)
- `io.stderr` - Standard error (fd 2)  
- `io.stdin` - Standard input (fd 0)

These compile to integer constants representing file descriptors.

## Write Keyword

### Syntax
```toy
write(handle, data)
```

### Parameters
- `handle`: An IO handle (currently integer file descriptors)
- `data`: Data to write (currently string literals)

### Semantics
- Returns number of bytes written (currently unused)
- Performs syscall directly without buffering
- No automatic newline appending

### Implementation Details

1. **Parser Stage**:
   - Recognizes `write` as keyword token
   - Parses as `write_expression` AST node
   - Expects exactly 2 arguments

2. **Type Checking**:
   - Validates handle is appropriate type
   - Ensures data is writable (string literals for now)
   - No special return type handling yet

3. **Code Generation**:
   - Computes string length at compile time
   - Generates platform-specific write syscall
   - No runtime library calls

### Example
```toy
pub fn main() void {
    write(io.stdout, "Hello, World!\n");
    write(io.stderr, "Debug: Starting program\n");
}
```

## Read Keyword (Planned)

### Syntax
```toy
let data = read(handle, type);
```

### Parameters
- `handle`: An IO handle to read from
- `type`: The type to read into (e.g., `[100]u8` for 100-byte buffer)

### Semantics
- Returns data of the specified type
- Blocks until data available
- No automatic line buffering

### Planned Implementation

1. **Buffer Management**:
   - Stack-allocated buffers for fixed-size reads
   - Type determines buffer size
   - No dynamic allocation

2. **Error Handling**:
   - EOF returns zero-filled data
   - Errors currently unhandled (future: Result type)

3. **Type Integration**:
   ```toy
   // Read line into fixed buffer
   let line = read(io.stdin, [256]u8);
   
   // Read exact size
   let header = read(file, [4]u8);
   
   // Future: Read numbers directly
   let num = read(io.stdin, i32);
   ```

## File IO (Future)

The design extends naturally to file operations:

```toy
// Planned syntax
let file = io.file("data.txt", io.READ);
let contents = read(file, [1024]u8);
write(file, "New data");
io.close(file);
```

### Design Considerations

1. **No Implicit State**: Files must be explicitly opened/closed
2. **Type Safety**: File handles are distinct from integers
3. **Resource Management**: Consider RAII-style cleanup

## Integration with Language Features

### With Tensors
```toy
// Future: Read tensor data directly
let matrix = read(io.stdin, [10,10]f32);
write(io.stdout, matrix);  // Formatted output
```

### With Strings (Future)
```toy
// When DCC has proper strings
let name = read(io.stdin, string);
write(io.stdout, "Hello, " + name);
```

### With Error Handling (Future)
```toy
// When DCC has Result types
let result = try read(io.stdin, [100]u8);
match result {
    Ok(data) => process(data),
    Err(e) => write(io.stderr, "Read failed"),
}
```

## Implementation Status

### Completed
- ✅ `write` keyword lexing and parsing
- ✅ `io.stdout`, `io.stderr`, `io.stdin` namespace access
- ✅ String literal writing
- ✅ Direct syscall generation
- ✅ Cross-platform support (Linux/macOS, x86_64/ARM64)

### In Progress
- 🚧 `read` keyword implementation
- 🚧 Error handling design

### Future Work
- 📋 File operations (`io.file`, `io.close`)
- 📋 Formatted output (printf-style)
- 📋 Binary IO modes
- 📋 Async IO operations
- 📋 Network IO

## Design Rationale

### Why Keywords Instead of Functions?

1. **Semantic Clarity**: IO operations are fundamentally different from pure functions
2. **Optimization**: Compiler can make better decisions knowing these are IO operations
3. **Safety**: Can enforce stricter rules (e.g., no IO in pure functions)
4. **Future Proofing**: Easier to add IO-specific features later

### Why Direct Syscalls?

1. **Portability**: No dependency on libc variations
2. **Predictability**: Exact control over IO behavior
3. **Size**: Smaller binaries without libc
4. **Cross-compilation**: No need for target system libraries

### Why No Buffering?

1. **Simplicity**: Easier to understand and implement
2. **Control**: Applications can implement their own buffering
3. **Real-time**: Predictable latency for system programming
4. **Correctness**: No hidden state to manage

## Best Practices

1. **Always check return values** (when implemented):
   ```toy
   let written = write(io.stdout, message);
   if (written != message.length) {
       // Handle partial write
   }
   ```

2. **Use stderr for errors**:
   ```toy
   write(io.stderr, "Error: Invalid input\n");
   ```

3. **Plan for buffering** in performance-critical code:
   ```toy
   // Future: Build output in tensor, write once
   let buffer: [1024]u8;
   // ... fill buffer ...
   write(io.stdout, buffer);
   ```

## Comparison with Other Languages

| Language | IO Approach | Buffering | Dependency |
|----------|-------------|-----------|------------|
| C | stdio functions | Buffered | libc |
| Rust | std::io traits | Buffered | std library |
| Go | io.Writer interface | Buffered | runtime |
| Zig | std.io namespace | Optional | std library |
| **DCC** | **write/read keywords** | **None** | **None** |

DCC's approach is most similar to assembly language or embedded systems programming, where IO is explicit and unbuffered, but with high-level syntax and type safety.