# DCC - A Simple Compiler Language

DCC is a simple, statically-typed programming language designed for learning compiler development. It features a clean syntax, basic type system, and compiles to native executables using LLVM.

## Features

- **🚀 Cross-Compilation**: Compile for different architectures and platforms from a single machine
  - macOS ARM64 → Linux x86_64
  - Linux x86_64 → macOS ARM64
- **Static Typing**: Support for basic types including integers (i8, i16, i32, i64), unsigned integers (u8, u16, u32, u64), and floating-point numbers (f32, f64)
- **Function Declarations**: Define functions with parameters and return types
- **Variable Declarations**: Declare and initialize variables with type annotations
- **Basic Expressions**: Arithmetic operations (+, -, *, /), unary negation, and function calls
- **Return Statements**: Return values from functions
- **Native Code Generation**: Compiles directly to native executables using LLVM
- **Cross-Platform**: Supports macOS (ARM64/x86_64) and Linux (x86_64)
- **Shared Library Support**: Can generate both executables and shared libraries

## Language Syntax

### Basic Types

```toy
// Integer types
let x: i32 = 42;
let y: u64 = 1000000;
let z: i8 = -128;

// Floating-point types
let pi: f64 = 3.14159;
let e: f32 = 2.71828f32;
```

### Function Declarations

```toy
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn multiply(x: i64, y: i64) -> i64 {
    return x * y;
}

fn main() -> i32 {
    let result: i32 = add(5, 3);
    return result;
}
```

### Variable Declarations

```toy
fn example() -> i32 {
    let x: i32 = 10;
    let y: i64 = 20;
    let sum: i64 = x + y;
    return sum;
}
```

### Expressions

```toy
fn arithmetic() -> i32 {
    let a: i32 = 10;
    let b: i32 = 5;
    
    let sum: i32 = a + b;      // Addition
    let diff: i32 = a - b;     // Subtraction
    let product: i32 = a * b;  // Multiplication
    let quotient: i32 = a / b; // Division
    
    let negative: i32 = -a;    // Unary negation
    
    return sum + diff + product + quotient + negative;
}
```

### Function Calls

```toy
fn helper(x: i32) -> i32 {
    return x * 2;
}

fn main() -> i32 {
    let result: i32 = helper(21);
    return result;
}
```

## Getting Started

### Prerequisites

- **Zig**: Version 0.14.0 or later
- **LLVM**: Version 19 (included in the build)
- **macOS or Linux**: Currently supported platforms

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dcc
   ```

2. **Build the compiler**:
   ```bash
   zig build
   ```

3. **Verify installation**:
   ```bash
   ./zig-out/bin/dcc --help
   ```

### Your First Program

1. **Create a simple program** (`hello.toy`):
   ```toy
   fn main() -> i32 {
       return 42;
   }
   ```

2. **Compile and run**:
   ```bash
   ./zig-out/bin/dcc hello.toy -o hello
   ./hello
   echo $?  # Should print 42
   ```

3. **Try cross-compilation** (optional):
   ```bash
   # If you're on macOS ARM64, compile for Linux x86_64
   ./zig-out/bin/dcc hello.toy --target x86_64-pc-linux-gnu -o hello_linux
   
   # If you're on Linux, compile for macOS ARM64
   ./zig-out/bin/dcc hello.toy --target arm64-apple-darwin -o hello_macos
   ```

### More Examples

#### Basic Arithmetic (`arithmetic.toy`):
```toy
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn multiply(x: i64, y: i64) -> i64 {
    return x * y;
}

fn main() -> i32 {
    let sum: i32 = add(10, 20);
    let product: i64 = multiply(5, 6);
    
    return sum;
}
```

#### Type Demonstrations (`types.toy`):
```toy
fn demonstrate_types() -> i32 {
    let int8_val: i8 = 127;
    let int16_val: i16 = 32767;
    let int32_val: i32 = 2147483647;
    let int64_val: i64 = 9223372036854775807;
    
    let uint8_val: u8 = 255;
    let uint16_val: u16 = 65535;
    let uint32_val: u32 = 4294967295;
    let uint64_val: u64 = 18446744073709551615;
    
    let float32_val: f32 = 3.14159f32;
    let float64_val: f64 = 2.718281828459045;
    
    return int32_val;
}

fn main() -> i32 {
    return demonstrate_types();
}
```

## Compiler Usage

### Basic Compilation

```bash
# Compile to executable
./zig-out/bin/dcc input.toy -o output

# Compile to object file
./zig-out/bin/dcc input.toy --obj -o output.o

# Compile to shared library
./zig-out/bin/dcc input.toy --shared -o liboutput
```

### Command Line Options

- `-o <file>`: Specify output file name
- `--obj`: Generate object file instead of executable
- `--shared`: Generate shared library
- `--target <triple>`: Specify target triple (e.g., `x86_64-apple-darwin`)
- `--verbose`: Enable verbose output showing generated LLVM IR
- `--help`: Show help message

### Cross-Compilation

DCC supports powerful cross-compilation capabilities, allowing you to compile code for different architectures and platforms from a single machine.

#### Target Triples

The compiler supports various target triples:

- `arm64-apple-darwin`: macOS ARM64
- `x86_64-apple-darwin`: macOS x86_64
- `x86_64-pc-linux-gnu`: Linux x86_64

#### Cross-Compilation Examples

```bash
# Compile on macOS ARM64 for Linux x86_64
./zig-out/bin/dcc hello.toy --target x86_64-pc-linux-gnu -o hello_linux

# Compile on Linux x86_64 for macOS ARM64
./zig-out/bin/dcc hello.toy --target arm64-apple-darwin -o hello_macos_arm

# Compile on macOS x86_64 for macOS ARM64
./zig-out/bin/dcc hello.toy --target arm64-apple-darwin -o hello_arm64

# Generate shared library for different platform
./zig-out/bin/dcc examples/library.toy --shared --target x86_64-pc-linux-gnu -o libexample.so
```

#### Cross-Compilation Use Cases

- **Development**: Develop on your preferred platform, test on others
- **CI/CD**: Build for multiple platforms from a single build server
- **Distribution**: Create binaries for all supported platforms
- **Testing**: Verify code works across different architectures

## Language Limitations

- No control flow statements (if/else, loops)
- No arrays or complex data structures
- No string literals
- No modules or imports
- Limited standard library
- No error handling beyond basic type checking

## Development

### Project Structure

```
dcc/
├── src/
│   ├── main.zig          # Main compiler entry point
│   ├── lexer.zig         # Lexical analysis
│   ├── parser.zig        # Syntax analysis and AST
│   ├── typechecker.zig   # Type checking
│   ├── codegen.zig       # LLVM code generation
│   └── integration_tests.zig # Test suite
├── build.zig             # Build configuration
├── build.zig.zon         # Dependencies
└── *.toy                 # Example source files
```

### Running Tests

```bash
# Run all tests
zig build test

# Run specific test file
zig build test -- test_simple.toy
```

### Adding New Features

1. **Lexer**: Add new token types in `src/lexer.zig`
2. **Parser**: Add new AST nodes in `src/parser.zig`
3. **Type Checker**: Add type checking logic in `src/typechecker.zig`
4. **Code Generator**: Add LLVM IR generation in `src/codegen.zig`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Built with [Zig](https://ziglang.org/)
- Uses [LLVM](https://llvm.org/) for code generation
- Inspired by educational compiler projects 
