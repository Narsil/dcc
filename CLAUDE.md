# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Overview

DCC (Declarative Compiler Collection) is a compiler for the Toy language, a statically-typed language with support for tensors, implicit indexing, and GPU computation. The compiler is written in Zig and generates LLVM IR or MLIR.

## Build Commands

```bash
# Build the compiler
zig build

# Run all tests
zig build test

# Run specific test categories
zig build test-integration    # Integration tests
zig build test-cross          # Cross-compilation tests
zig build test-mlir           # MLIR codegen tests

# Run the compiler
zig build run -- <source_file.toy> [options]

# Common development pattern: compile and run a toy file
zig build run -- test.toy --verbose
```

## Architecture

### Compiler Pipeline
1. **Lexer** (`src/lexer.zig`): Tokenizes source code
2. **Parser** (`src/parser.zig`): Builds AST with nodes for:
   - Basic expressions, functions, variables
   - Tensor literals and operations
   - Implicit tensor indexing (`a[i]` means iterate over dimension)
   - Reduce expressions (`reduce(a[i,j], +)`)
3. **Type Checker** (`src/typechecker.zig`): 
   - Validates types and tensor dimensions
   - Tracks implicit indices and reduction operations
   - Stores reduction info for codegen
4. **Code Generator** (`src/codegen/core.zig`): 
   - Generates LLVM IR
   - Handles multi-dimensional tensor indexing
   - Implements reduction operations with nested loops

### Key Language Features

**Implicit Tensor Indexing**: `a[i] = b[i] + c[i]` means element-wise operation over all indices.

**Reduce Operations**: `a[i] = reduce(b[i,j], +)` reduces tensor `b` along dimension `j`, keeping dimension `i`. The operation:
- Requires implicit tensor expression (not just `reduce(b, +)`)
- Supports only `+` and `*` operators
- Validates index consistency between LHS and RHS
- Free indices appear on both sides, bound indices only in reduce

**Tensor Types**: `[dim1, dim2, ...]type`, e.g., `[3,4]i32` is a 3x4 matrix of i32.

### Testing Strategy

When adding features:
1. Add lexer tokens if needed
2. Update parser AST nodes
3. Implement typechecker validation
4. Add codegen implementation
5. Write tests in `src/integration_tests.zig`

Error handling tests are crucial - the compiler should provide clear error messages for:
- Type mismatches
- Invalid reduce operations
- Rank mismatches
- Index conflicts

### Common Gotchas

1. **Multi-dimensional tensors** are stored as flat arrays in LLVM. Index calculation: `linear_index = i * stride[0] + j * stride[1] + ...`

2. **Error positions** must use the correct offset from the AST node, not parent nodes.

3. **Variable scoping**: Implicit indices (like `i` in `a[i]`) must not conflict with existing variables.

4. **Reduction analysis** happens in typechecker and stores info for codegen to use.

5. **Cross-compilation**: Use `--target` flag with triples like `x86_64-pc-linux-gnu` or `arm64-apple-darwin`.

## Debugging Tips

- Use `--verbose` flag to see generated LLVM IR
- Add debug prints in typechecker with `if (self.verbose)`
- Check `generated_mlir.mlir` for MLIR output
- Run specific test: `zig build test 2>&1 | grep -A5 "test_name"`
