gpu.module @kernels {
  gpu.func @gpu_add_kernel(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>) kernel {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    
    // Calculate global thread index: blockIdx.x * blockDim.x + threadIdx.x
    %block_id = gpu.block_id x
    %block_dim = gpu.block_dim x
    %thread_id = gpu.thread_id x
    %block_offset = arith.muli %block_id, %block_dim : index
    %global_id = arith.addi %block_offset, %thread_id : index
    
    // Bounds check: if (global_id >= 1024) return
    %cond = arith.cmpi ult, %global_id, %c1024 : index
    scf.if %cond {
      // Load values: val1 = a[i], val2 = b[i]
      %val1 = memref.load %arg0[%global_id] : memref<1024xf32>
      %val2 = memref.load %arg1[%global_id] : memref<1024xf32>
      
      // Perform addition: result = a[i] + b[i]
      %sum = arith.addf %val1, %val2 : f32
      
      // Store result back to a[i] (in-place)
      memref.store %sum, %arg0[%global_id] : memref<1024xf32>
    }
    gpu.return
  }
} 