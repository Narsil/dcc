const std = @import("std");
const parser = @import("parser.zig");
const mlir_gpu = @import("mlir_gpu.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== DCC GPU Compilation Test ===\n", .{});

    // Create GPU target (A100 equivalent)
    const gpu_target = try mlir_gpu.GPUTarget.from_compute_capability(80);
    std.debug.print("GPU Target: {s} (CC {d})\n", .{ gpu_target.get_arch_name(), gpu_target.compute_capability });

    // Initialize GPU compiler
    var gpu_compiler = try mlir_gpu.GPUCompiler.init(allocator, "gpu_test", gpu_target);
    defer gpu_compiler.deinit();

    // Create a simple AST for testing
    const test_ast = try createTestAST(allocator);
    defer parser.freeAST(allocator, test_ast);

    std.debug.print("Generated AST for GPU compilation\n", .{});

    // Try to compile to PTX (this will fail without MLIR, but demonstrates the approach)
    const ptx_result = gpu_compiler.compileKernel(test_ast);
    if (ptx_result) |ptx| {
        std.debug.print("Generated PTX:\n{s}\n", .{ptx});
        allocator.free(ptx);
    } else |err| {
        std.debug.print("PTX generation failed (expected without MLIR): {}\n", .{err});
    }

    // Print the MLIR module
    gpu_compiler.printModule();

    std.debug.print("=== GPU Compilation Test Complete ===\n", .{});
}

fn createTestAST(allocator: std.mem.Allocator) !parser.ASTNode {
    // Create a simple function: fn add(a: i32, b: i32) -> i32 { return a + b; }
    
    const param_a = parser.Parameter{
        .name = try allocator.dupe(u8, "a"),
        .type = parser.Type{ .int = .i32 },
    };
    
    const param_b = parser.Parameter{
        .name = try allocator.dupe(u8, "b"),
        .type = parser.Type{ .int = .i32 },
    };
    
    const params = try allocator.alloc(parser.Parameter, 2);
    params[0] = param_a;
    params[1] = param_b;
    
    const a_expr = parser.ASTNode{ .identifier = .{ .name = try allocator.dupe(u8, "a") } };
    const b_expr = parser.ASTNode{ .identifier = .{ .name = try allocator.dupe(u8, "b") } };
    
    const add_expr = parser.ASTNode{
        .binary_expression = .{
            .left = try allocator.create(parser.ASTNode),
            .right = try allocator.create(parser.ASTNode),
            .operator = .add,
        },
    };
    add_expr.binary_expression.left.* = a_expr;
    add_expr.binary_expression.right.* = b_expr;
    
    const return_stmt = parser.ASTNode{
        .return_statement = .{
            .value = try allocator.create(parser.ASTNode),
        },
    };
    return_stmt.return_statement.value.* = add_expr;
    
    const body = try allocator.alloc(parser.ASTNode, 1);
    body[0] = return_stmt;
    
    const func = parser.ASTNode{
        .function_declaration = .{
            .name = try allocator.dupe(u8, "add"),
            .parameters = params,
            .return_type = parser.Type{ .int = .i32 },
            .body = body,
        },
    };
    
    return func;
} 