const std = @import("std");
const parser = @import("parser.zig");

pub const SimpleCodeGen = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayList(u8),
    variable_counter: u32,
    
    pub fn init(allocator: std.mem.Allocator) SimpleCodeGen {
        return SimpleCodeGen{
            .allocator = allocator,
            .output = std.ArrayList(u8).init(allocator),
            .variable_counter = 0,
        };
    }
    
    pub fn deinit(self: *SimpleCodeGen) void {
        self.output.deinit();
    }
    
    pub fn generate(self: *SimpleCodeGen, ast: parser.ASTNode) ![]const u8 {
        // LLVM IR header
        try self.output.appendSlice("; ModuleID = 'toy_program'\n");
        try self.output.appendSlice("target triple = \"x86_64-unknown-linux-gnu\"\n\n");
        
        // Generate code for the program
        switch (ast) {
            .program => |prog| {
                for (prog.statements) |stmt| {
                    try self.generateStatement(stmt);
                }
            },
            else => return error.InvalidTopLevelNode,
        }
        
        return self.output.items;
    }
    
    fn generateStatement(self: *SimpleCodeGen, node: parser.ASTNode) !void {
        switch (node) {
            .function_declaration => try self.generateFunction(node.function_declaration),
            else => {}, // Skip other statements for now
        }
    }
    
    fn generateFunction(self: *SimpleCodeGen, func: @TypeOf(@as(parser.ASTNode, undefined).function_declaration)) !void {
        // Function signature
        try self.output.appendSlice("define i64 @");
        try self.output.appendSlice(func.name);
        try self.output.appendSlice("(");
        
        // Parameters
        for (func.parameters, 0..) |param, i| {
            if (i > 0) try self.output.appendSlice(", ");
            try self.output.appendSlice("i64 %");
            try self.output.appendSlice(param);
        }
        
        try self.output.appendSlice(") {\n");
        try self.output.appendSlice("entry:\n");
        
        // Allocate stack space for parameters
        for (func.parameters) |param| {
            try self.output.appendSlice("  %");
            try self.output.appendSlice(param);
            try self.output.appendSlice(".addr = alloca i64, align 8\n");
            try self.output.appendSlice("  store i64 %");
            try self.output.appendSlice(param);
            try self.output.appendSlice(", i64* %");
            try self.output.appendSlice(param);
            try self.output.appendSlice(".addr, align 8\n");
        }
        
        // Generate function body
        var return_value: ?[]const u8 = null;
        for (func.body) |stmt| {
            const result = try self.generateFunctionStatement(stmt);
            if (result) |val| return_value = val;
        }
        
        // Return statement
        if (return_value) |val| {
            try self.output.appendSlice("  ret i64 ");
            try self.output.appendSlice(val);
            try self.output.appendSlice("\n");
        } else {
            try self.output.appendSlice("  ret i64 0\n");
        }
        
        try self.output.appendSlice("}\n\n");
    }
    
    fn generateFunctionStatement(self: *SimpleCodeGen, node: parser.ASTNode) !?[]const u8 {
        switch (node) {
            .variable_declaration => {
                try self.generateVariableDeclaration(node.variable_declaration);
                return null;
            },
            .return_statement => {
                return try self.generateReturnStatement(node.return_statement);
            },
            else => return null,
        }
    }
    
    fn generateVariableDeclaration(self: *SimpleCodeGen, var_decl: @TypeOf(@as(parser.ASTNode, undefined).variable_declaration)) !void {
        // Allocate stack space
        try self.output.appendSlice("  %");
        try self.output.appendSlice(var_decl.name);
        try self.output.appendSlice(".addr = alloca i64, align 8\n");
        
        // Generate the value expression
        const value_reg = try self.generateExpression(var_decl.value.*);
        
        // Store the value
        try self.output.appendSlice("  store i64 ");
        try self.output.appendSlice(value_reg);
        try self.output.appendSlice(", i64* %");
        try self.output.appendSlice(var_decl.name);
        try self.output.appendSlice(".addr, align 8\n");
    }
    
    fn generateReturnStatement(self: *SimpleCodeGen, ret: @TypeOf(@as(parser.ASTNode, undefined).return_statement)) !?[]const u8 {
        if (ret.value) |value| {
            return try self.generateExpression(value.*);
        }
        return null;
    }
    
    fn generateExpression(self: *SimpleCodeGen, node: parser.ASTNode) ![]const u8 {
        switch (node) {
            .number_literal => |num| {
                return try std.fmt.allocPrint(self.allocator, "{}", .{num.value});
            },
            .identifier => |ident| {
                const reg_name = try self.getNextRegister();
                try self.output.appendSlice("  ");
                try self.output.appendSlice(reg_name);
                try self.output.appendSlice(" = load i64, i64* %");
                try self.output.appendSlice(ident.name);
                try self.output.appendSlice(".addr, align 8\n");
                return reg_name;
            },
            .binary_expression => |bin_expr| {
                const left_reg = try self.generateExpression(bin_expr.left.*);
                const right_reg = try self.generateExpression(bin_expr.right.*);
                const result_reg = try self.getNextRegister();
                
                try self.output.appendSlice("  ");
                try self.output.appendSlice(result_reg);
                try self.output.appendSlice(" = ");
                
                switch (bin_expr.operator) {
                    .add => try self.output.appendSlice("add"),
                    .subtract => try self.output.appendSlice("sub"),
                    .multiply => try self.output.appendSlice("mul"),
                    .divide => try self.output.appendSlice("sdiv"),
                }
                
                try self.output.appendSlice(" i64 ");
                try self.output.appendSlice(left_reg);
                try self.output.appendSlice(", ");
                try self.output.appendSlice(right_reg);
                try self.output.appendSlice("\n");
                
                return result_reg;
            },
            .call_expression => |call| {
                const callee_name = switch (call.callee.*) {
                    .identifier => |ident| ident.name,
                    else => return error.InvalidCallee,
                };
                
                var args = std.ArrayList([]const u8).init(self.allocator);
                defer args.deinit();
                
                for (call.arguments) |arg| {
                    const arg_reg = try self.generateExpression(arg);
                    try args.append(arg_reg);
                }
                
                const result_reg = try self.getNextRegister();
                try self.output.appendSlice("  ");
                try self.output.appendSlice(result_reg);
                try self.output.appendSlice(" = call i64 @");
                try self.output.appendSlice(callee_name);
                try self.output.appendSlice("(");
                
                for (args.items, 0..) |arg, i| {
                    if (i > 0) try self.output.appendSlice(", ");
                    try self.output.appendSlice("i64 ");
                    try self.output.appendSlice(arg);
                }
                
                try self.output.appendSlice(")\n");
                return result_reg;
            },
            else => return error.InvalidExpression,
        }
    }
    
    fn getNextRegister(self: *SimpleCodeGen) ![]const u8 {
        self.variable_counter += 1;
        return try std.fmt.allocPrint(self.allocator, "%{}", .{self.variable_counter});
    }
    
    pub fn writeToFile(self: *SimpleCodeGen, filename: []const u8) !void {
        const file = try std.fs.cwd().createFile(filename, .{});
        defer file.close();
        try file.writeAll(self.output.items);
    }
}; 