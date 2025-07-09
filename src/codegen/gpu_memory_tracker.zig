const std = @import("std");
pub const LLVM = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/Support.h");
    @cInclude("llvm-c/BitReader.h");
    @cInclude("llvm-c/Object.h");
});

/// Tracks the location of variables (CPU or GPU) to optimize memory transfers
pub const GpuMemoryTracker = struct {
    /// Represents the location of a variable
    pub const Location = enum {
        cpu,
        gpu,
        both, // Data is synchronized on both CPU and GPU
    };

    /// Information about a tracked variable
    pub const VarInfo = struct {
        location: Location,
        gpu_ptr: ?LLVM.LLVMValueRef, // GPU pointer if allocated
        cpu_ptr: LLVM.LLVMValueRef, // CPU pointer (always exists)
        size: u64, // Size in bytes
        last_modified: Location, // Where was it last modified
    };

    allocator: std.mem.Allocator,
    variables: std.StringHashMap(VarInfo),
    verbose: bool,

    pub fn init(allocator: std.mem.Allocator, verbose: bool) GpuMemoryTracker {
        return .{
            .allocator = allocator,
            .variables = std.StringHashMap(VarInfo).init(allocator),
            .verbose = verbose,
        };
    }

    pub fn deinit(self: *GpuMemoryTracker) void {
        self.variables.deinit();
    }

    /// Register a new variable as being on CPU
    pub fn registerVariable(self: *GpuMemoryTracker, name: []const u8, cpu_ptr: LLVM.LLVMValueRef, size: u64) !void {
        try self.variables.put(name, .{
            .location = .cpu,
            .gpu_ptr = null,
            .cpu_ptr = cpu_ptr,
            .size = size,
            .last_modified = .cpu,
        });

        if (self.verbose) {
            std.debug.print("📍 Registered variable '{s}' on CPU (size: {} bytes)\n", .{ name, size });
        }
    }

    /// Mark that a variable has been allocated on GPU
    pub fn markGpuAllocated(self: *GpuMemoryTracker, name: []const u8, gpu_ptr: LLVM.LLVMValueRef) !void {
        if (self.variables.getPtr(name)) |info| {
            info.gpu_ptr = gpu_ptr;
            if (self.verbose) {
                std.debug.print("🎮 Allocated GPU memory for variable '{s}'\n", .{name});
            }
        }
    }

    /// Mark that data has been copied from CPU to GPU
    pub fn markCopiedToGpu(self: *GpuMemoryTracker, name: []const u8) !void {
        if (self.variables.getPtr(name)) |info| {
            info.location = .both;
            if (self.verbose) {
                std.debug.print("⬆️  Copied variable '{s}' to GPU\n", .{name});
            }
        }
    }

    /// Mark that data has been copied from GPU to CPU
    pub fn markCopiedToCpu(self: *GpuMemoryTracker, name: []const u8) !void {
        if (self.variables.getPtr(name)) |info| {
            info.location = .both;
            if (self.verbose) {
                std.debug.print("⬇️  Copied variable '{s}' to CPU\n", .{name});
            }
        }
    }

    /// Mark that a variable has been modified on GPU
    pub fn markModifiedOnGpu(self: *GpuMemoryTracker, name: []const u8) !void {
        if (self.variables.getPtr(name)) |info| {
            info.location = .gpu;
            info.last_modified = .gpu;
            if (self.verbose) {
                std.debug.print("✏️  Variable '{s}' modified on GPU (CPU copy now stale)\n", .{name});
            }
        }
    }

    /// Mark that a variable has been modified on CPU
    pub fn markModifiedOnCpu(self: *GpuMemoryTracker, name: []const u8) !void {
        if (self.variables.getPtr(name)) |info| {
            info.location = .cpu;
            info.last_modified = .cpu;
            if (self.verbose) {
                std.debug.print("✏️  Variable '{s}' modified on CPU (GPU copy now stale)\n", .{name});
            }
        }
    }

    /// Check if a variable needs to be transferred to GPU
    pub fn needsTransferToGpu(self: *GpuMemoryTracker, name: []const u8) bool {
        if (self.variables.get(name)) |info| {
            // Need transfer if:
            // 1. Data is only on CPU
            // 2. Data is on both but was last modified on CPU
            return info.location == .cpu or (info.location == .both and info.last_modified == .cpu);
        }
        return true; // Conservative: if unknown, assume transfer needed
    }

    /// Check if a variable needs to be transferred to CPU
    pub fn needsTransferToCpu(self: *GpuMemoryTracker, name: []const u8) bool {
        if (self.variables.get(name)) |info| {
            return info.location == .gpu and info.last_modified == .gpu;
        }
        return false;
    }

    /// Get GPU pointer for a variable (null if not allocated)
    pub fn getGpuPtr(self: *GpuMemoryTracker, name: []const u8) ?LLVM.LLVMValueRef {
        if (self.variables.get(name)) |info| {
            return info.gpu_ptr;
        }
        return null;
    }

    /// Get CPU pointer for a variable
    pub fn getCpuPtr(self: *GpuMemoryTracker, name: []const u8) ?LLVM.LLVMValueRef {
        if (self.variables.get(name)) |info| {
            return info.cpu_ptr;
        }
        return null;
    }

    /// Get the size of a variable
    pub fn getSize(self: *GpuMemoryTracker, name: []const u8) ?u64 {
        if (self.variables.get(name)) |info| {
            return info.size;
        }
        return null;
    }

    /// Print current state of all tracked variables
    pub fn printState(self: *GpuMemoryTracker) void {
        std.debug.print("=== GPU Memory Tracker State ===\n", .{});
        var it = self.variables.iterator();
        while (it.next()) |entry| {
            const info = entry.value_ptr.*;
            std.debug.print("  {s}: location={s}, gpu_allocated={}, last_modified={s}\n", .{
                entry.key_ptr.*,
                @tagName(info.location),
                info.gpu_ptr != null,
                @tagName(info.last_modified),
            });
        }
        std.debug.print("================================\n", .{});
    }
};

