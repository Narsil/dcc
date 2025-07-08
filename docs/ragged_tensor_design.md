# Ragged Tensor Design for DCC

## Overview

This document describes the design for supporting ragged (variable-length) tensors in DCC, with a focus on enabling efficient implementations of algorithms like Flash Attention that operate on sequences of varying lengths.

## Motivation

Many modern ML workloads, particularly in NLP, require operating on batches of sequences with different lengths. Current approaches typically either:
1. Pad all sequences to maximum length (wasteful)
2. Pack sequences into flat arrays with separate metadata (error-prone)

We propose a type system extension that makes ragged tensors first-class citizens, ensuring type safety while maintaining DCC's philosophy of element-wise operations and compiler-driven optimization.

## Core Design: The `coupling` Keyword

### Basic Syntax

```toy
type PackedSequences = coupling {
    data: [b: B][seqlen[b], H, D]f32,
    seqlen: [B]i32
}
```

The `coupling` keyword creates a type that bundles:
- A ragged tensor with variable-length dimensions
- The metadata that defines those dimensions

### Key Properties

1. **Type Safety**: The compiler ensures `data` and `seqlen` remain consistent
2. **Single Unit**: Pass one argument instead of multiple dependent ones
3. **Clear Dependencies**: The relationship between fields is explicit in the type

## Usage Examples

### Flash Attention

```toy
type PackedQueries = coupling {
    data: [b: B][seqlen[b], H, D]f32,
    seqlen: [B]i32
}

type PackedKV = coupling {
    data: [b: B][seqlen[b], H, D]f32,
    seqlen: [B]i32
}

fn flash_attention(q: PackedQueries, kv: PackedKV) PackedQueries {
    // Compute attention scores for each sequence
    let scores: [b: B][q.seqlen[b], kv.seqlen[b]]f32;
    
    scores[b][i, j] = reduce(+, 
        q.data[b][i, h, d] * kv.data[b][j, h, d]
    ) / sqrt(D);
    
    // Softmax over each sequence
    let max_scores: [b: B][q.seqlen[b]]f32;
    max_scores[b][i] = reduce(max, scores[b][i, j]);
    
    let exp_scores: [b: B][q.seqlen[b], kv.seqlen[b]]f32;
    exp_scores[b][i, j] = exp(scores[b][i, j] - max_scores[b][i]);
    
    let sum_exp: [b: B][q.seqlen[b]]f32;
    sum_exp[b][i] = reduce(+, exp_scores[b][i, j]);
    
    let attn_weights: [b: B][q.seqlen[b], kv.seqlen[b]]f32;
    attn_weights[b][i, j] = exp_scores[b][i, j] / sum_exp[b][i];
    
    // Apply attention to values
    let output: PackedQueries = PackedQueries {
        data: [b: B][q.seqlen[b], H, D]f32,
        seqlen: q.seqlen
    };
    
    output.data[b][i, h, d] = reduce(+, 
        attn_weights[b][i, j] * kv.data[b][j, h, d]
    );
    
    return output;
}
```

### Construction

```toy
fn pack_sequences(sequences: list[[?, H, D]f32]) PackedSequences {
    let B = len(sequences);
    let seqlen: [B]i32;
    
    // Compute lengths
    seqlen[b] = len(sequences[b]);
    
    // Create coupled type - compiler validates consistency
    let packed = PackedSequences {
        data: [b: B][seqlen[b], H, D]f32,
        seqlen: seqlen
    };
    
    // Copy data
    packed.data[b][i, h, d] = sequences[b][i, h, d];
    
    return packed;
}
```

### Complex Coupling

```toy
type AttentionData = coupling {
    queries: [b: B][q_seqlen[b], H, D]f32,
    keys: [b: B][kv_seqlen[b], H, D]f32,
    values: [b: B][kv_seqlen[b], H, D]f32,
    q_seqlen: [B]i32,
    kv_seqlen: [B]i32,
    attention_mask: [b: B][q_seqlen[b], kv_seqlen[b]]bool
}
```

## Implementation Considerations

### Memory Layout

The compiler must decide how to store coupled data:

1. **Separate Arrays**: Store each field independently
   - Pros: Simple, allows independent access
   - Cons: May have poor cache locality

2. **Packed Format**: Flatten ragged data into single array with offsets
   - Pros: Better memory efficiency, single allocation
   - Cons: More complex indexing

3. **Hybrid**: Let compiler choose based on usage patterns

### Type Checking

The compiler must verify:
- Dimension references (`seqlen[b]`) are valid
- All uses of `b` index are consistent
- Construction provides compatible data

### Code Generation

For GPU execution, the compiler can:
- Generate efficient kernels that process each sequence independently
- Avoid padding overhead
- Use block sizes adapted to actual sequence lengths
- Leverage warp-level operations for sequences that fit in a warp

## Advantages

1. **Type Safety**: Impossible to pass mismatched data and lengths
2. **Clarity**: Dependencies are explicit
3. **Performance**: Compiler has full information for optimization
4. **Composability**: Coupling types work with existing DCC operations

## Alternative Syntax Considered

We also considered:
- Jagged arrays: `[[_, H, D]]f32` 
- Sparse tensors with block structure
- Masked operations on padded tensors

The `coupling` approach was chosen for its explicitness and ability to handle complex relationships between multiple arrays.

## Future Extensions

1. **Constraints**: Add validation rules
   ```toy
   type ValidatedPacked = coupling {
       data: [b: B][seqlen[b], H, D]f32,
       seqlen: [B]i32
   } where {
       all(seqlen[b] > 0),
       sum(seqlen) <= MAX_TOKENS
   }
   ```

2. **Layout Hints**: Control memory layout
   ```toy
   type PackedTensor = coupling {
       data: [b: B][seqlen[b], H, D]f32,
       seqlen: [B]i32
   } layout(interleaved)
   ```

3. **Automatic Packing**: Compiler-generated packing/unpacking
   ```toy
   fn auto_pack(seqs: [[?, H, D]]f32) -> auto coupling
   ```

## Conclusion

The `coupling` keyword provides a clean, type-safe way to express ragged tensors while maintaining DCC's philosophy of declarative, element-wise operations. It enables efficient implementations of complex algorithms like Flash Attention without forcing users to manually manage indices and offsets.