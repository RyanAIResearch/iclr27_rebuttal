# Register-lifetime change in the SR backward leaf kernel

This page shows the whole single-reconstruction (SR) backward leaf kernel from [`sr_backward_leaf_kernel.cu`](sr_backward_leaf_kernel.cu). The plain patch is in [`register_lifetime.diff`](register_lifetime.diff). Only the highlighted lines belong to the register-lifetime change; every other line is unchanged.

- Green (`+`): code after the change.
- Red (`-`): code it replaced.
- No marker: unchanged code.

Changed lines: 28 added, 26 removed. Arithmetic, the dBias reduction order and the dK/dV accumulation order are unchanged.

| # | Lines in `sr_backward_leaf_kernel.cu` | Change | Added / removed |
|---|---|---|---|
| 1 | 155–162 | slot validity | +8 / −5 |
| 2 | 206 | slot validity | +1 / −1 |
| 3 | 365 | slot validity | +1 / −1 |
| 4 | 369–371 | BF16 packing | +3 / −0 |
| 5 | 414–417 | BF16 packing | +4 / −0 |
| 6 | 422 | BF16 packing | +1 / −1 |
| 7 | 427–434 | BF16 packing | +8 / −16 |
| 8 | 478 | slot validity | +1 / −1 |
| 9 | 540 | slot validity | +1 / −1 |

**slot validity.** Per-slot validity is computed where it is used (`slot_is_active`) instead of a float array `outer_mult[kOuterLeaf]` that stayed live for the whole kernel. For O == N it is the compile-time constant `true`.

**BF16 packing.** P and dS are packed into BF16 pairs (`p_packed`, `ds_packed`) right after the FP32 reconstruction; the dK/dV MMA loop reads the packed values, so the FP32 `dp`/`score` accumulators are no longer live across that loop.

```diff
 // ---------------------------------------------------------------------------
 // SR (single-reconstruction) backward leaf: ONE kernel reconstructs
 // {S^T, P^T, dP^T, dS^T} per (outer, q-tile) and feeds all four gradients.
 //   * CTA = (64-key block, head, 4-outer leaf), 256 threads = 2 warpgroups
 //     owning slot pairs {0,1}/{2,3} (the proven dq-wg platform, mirrored to
 //     K residency).  K/V and the additive mask are key-invariant per CTA and
 //     staged once; the CTA sweeps the query slab in 32-row tiles.
 //   * dK/dV accumulate in registers with the dkdv chain order (A = packed
 //     dS^T/P^T accumulators, B = rowtrans over the staged q/dO tiles) --
 //     bit-identical to the two-pass dkdv by construction.
 //   * dBias keeps the exact workspace contract: per q-tile the two
 //     warpgroups fold (o0+o1)/(o2+o3) into FP32 tree tiles, combine, and
 //     single-write the leaf plane cells [q, kb-keys]; leaf-wave residency
 //     and the finalize kernel are untouched.
 //   * dQ is the one scattered gradient: per (kb, q-tile, slot) the dS tile
 //     is transposed through shared memory, multiplied by K, and written to
 //     a flat FP32 partial [kb][plane][q][D]; a fixed ascending-kb reduce
 //     folds partials into BF16 dQ.  delta = classic dO.O (v1 contract).
 constexpr int kSrKeyBlock = 64;
 constexpr int kSrQueryTile = 32;
 constexpr int kSrTreeStride = kSrKeyBlock + 1;
 constexpr int kSrBiasStride = kSrKeyBlock + 8;
 constexpr int kSrDsStride = kSrKeyBlock + 8;

 __device__ __forceinline__ void sr_cp_async_16(
     __nv_bfloat16* dst, const __nv_bfloat16* src) {
   asm volatile(
       "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_u32(dst)),
       "l"(src));
 }

 __device__ __forceinline__ void sr_cp_async_commit() {
   asm volatile("cp.async.commit_group;\n");
 }

 __device__ __forceinline__ void sr_cp_async_wait_all() {
   asm volatile("cp.async.wait_group 0;\n");
 }

 template <int HeadDim>
 __device__ __forceinline__ void ldmatrix_b_rowtrans_sw_sr(
     const __nv_bfloat16* tile, int row_start, int dim_start, int lane,
     uint32_t (&frag)[2]) {
   const int row = row_start + (lane & 15);
   const uint32_t addr = shared_u32(
       tile + row * HeadDim + wg_swz_col<HeadDim>(row, dim_start));
   asm volatile(
       "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
       : "=r"(frag[0]), "=r"(frag[1])
       : "r"(addr));
 }

 template <int HeadDim, bool TaskInGridZ, bool OuterEqSeq>
 __global__ __launch_bounds__(kThreadsWg, 2) void turbopairformer_sr_leaf_kernel(
     const __nv_bfloat16* __restrict__ q,
     const __nv_bfloat16* __restrict__ k,
     const __nv_bfloat16* __restrict__ v,
     const __nv_bfloat16* __restrict__ d_output,
     const float* __restrict__ logsumexp,
     const float* __restrict__ delta,
     const float* __restrict__ mask,
     const __nv_bfloat16* __restrict__ bias,
     __nv_bfloat16* __restrict__ out_dk,
     __nv_bfloat16* __restrict__ out_dv,
     float* __restrict__ dk_continuation,
     float* __restrict__ dv_continuation,
     __nv_bfloat16* __restrict__ dq_partials,
     __nv_bfloat16* __restrict__ workspace,
     int sequence,
     int outer_count,
     int total_planes,
     int heads,
     int batch_base,
     int active_batch,
     int batch_capacity,
     int query_base,
     int query_rows,
     int query_capacity,
     int leaf_base,
     int io_dim,
     float scale) {
   const int lane = threadIdx.x & (kWarpSize - 1);
   const int warp = threadIdx.x / kWarpSize;
   const int warpgroup = warp / kWarpsPerBlock;
   const int warp_in_group = warp & (kWarpsPerBlock - 1);
   const int wave_task = TaskInGridZ ? blockIdx.z : blockIdx.x;
   const int local_batch = wave_task % active_batch;
   const int batch_index = batch_base + local_batch;
   const int wave_leaf = wave_task / active_batch;
   const int key_block = TaskInGridZ ? blockIdx.x : blockIdx.z;
   const int key_start = key_block * kSrKeyBlock;
   const int head = blockIdx.y;
   const int leaf = leaf_base + wave_leaf;
   const int group = lane >> 2;
   const int thread_in_group = lane & 3;
   const int warp_key = warp_in_group * kQueryTile;

   // q/dO/bias staging is double-buffered behind cp.async so the global
   // latency of the next query tile hides under the current tile's math.
   extern __shared__ __align__(1024) unsigned char smem_raw[];
   __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem_raw);
   __nv_bfloat16* sv = sk + kOuterLeaf * kSrKeyBlock * HeadDim;
   __nv_bfloat16* sq = sv + kOuterLeaf * kSrKeyBlock * HeadDim;
   __nv_bfloat16* sdo = sq + 2 * kOuterLeaf * kSrQueryTile * HeadDim;
   __nv_bfloat16* sds = sdo + 2 * kOuterLeaf * kSrQueryTile * HeadDim;
   __nv_bfloat16* sbias_tile =
       sds + kOuterLeaf * kSrQueryTile * kSrDsStride;
   float* smask = reinterpret_cast<float*>(
       sbias_tile + 2 * kSrQueryTile * kSrBiasStride);
   float* sbias_acc = smask + kOuterLeaf * kSrKeyBlock;

   const int64_t plane_base =
       (((static_cast<int64_t>(wave_leaf) * batch_capacity + local_batch) *
             heads +
         head) *
            static_cast<int64_t>(query_capacity)) *
       sequence;
   const int64_t bias_base =
       ((static_cast<int64_t>(batch_index) * heads + head) * sequence) *
       static_cast<int64_t>(sequence);
   const bool single_outer = !OuterEqSeq && outer_count == 1;

   const int oc = OuterEqSeq ? sequence : outer_count;
   const auto slot_plane = [&](int slot) -> int64_t {
     const int outer = leaf * kOuterLeaf + slot;
     const int outer_load = outer < oc ? outer : oc - 1;
     return static_cast<int64_t>(batch_index) * oc + outer_load;
   };
-  float outer_mult[kOuterLeaf];
-#pragma unroll
-  for (int slot = 0; slot < kOuterLeaf; ++slot) {
-    outer_mult[slot] = leaf * kOuterLeaf + slot < oc ? 1.0F : 0.0F;
-  }
+  const auto slot_is_active = [&](int slot) {
+    if constexpr (OuterEqSeq) {
+      // The SR launch requires N % 64 == 0, so O == N has full leaves.
+      return true;
+    } else {
+      return leaf * kOuterLeaf + slot < oc;
+    }
+  };

   // K/V and the additive mask are key-invariant: staged once.
 #pragma unroll 1
   const int staged_slots = single_outer ? 1 : kOuterLeaf;
   for (int slot = 0; slot < staged_slots; ++slot) {
     __nv_bfloat16* dst_k = sk + slot * kSrKeyBlock * HeadDim;
     __nv_bfloat16* dst_v = sv + slot * kSrKeyBlock * HeadDim;
     const int64_t tensor_base = (slot_plane(slot) * heads + head) *
         static_cast<int64_t>(sequence) * io_dim;
     for (int element = threadIdx.x; element < kSrKeyBlock * HeadDim;
          element += kThreadsWg) {
       const int row = element / HeadDim;
       const int col = element - row * HeadDim;
       const int64_t src = tensor_base +
           static_cast<int64_t>(key_start + row) * io_dim + col;
       const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
       dst_k[dst_index] =
           col < io_dim ? k[src] : __float2bfloat16_rn(0.0F);
       dst_v[dst_index] =
           col < io_dim ? v[src] : __float2bfloat16_rn(0.0F);
     }
   }
   for (int element = threadIdx.x; element < staged_slots * kSrKeyBlock;
        element += kThreadsWg) {
     const int slot = element / kSrKeyBlock;
     const int key = element - slot * kSrKeyBlock;
     smask[element] = mask[slot_plane(slot) * sequence + key_start + key];
   }

   Acc dk_acc[kSlotsPerWarpgroup][HeadDim / 8];
   Acc dv_acc[kSlotsPerWarpgroup][HeadDim / 8];
 #pragma unroll
   for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
 #pragma unroll
     for (int tile = 0; tile < HeadDim / 8; ++tile) {
       dk_acc[local][tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
       dv_acc[local][tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
     }
   }
   if (query_base != 0) {
 #pragma unroll
     for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
       const int slot = warpgroup * kSlotsPerWarpgroup + local;
-      if (outer_mult[slot] == 0.0F) {
+      if (!slot_is_active(slot)) {
         continue;
       }
       const int64_t out_base = (slot_plane(slot) * heads + head) *
               static_cast<int64_t>(sequence) * io_dim +
           static_cast<int64_t>(key_start + warp_key) * io_dim;
 #pragma unroll
       for (int tile = 0; tile < HeadDim / 8; ++tile) {
 #pragma unroll
         for (int half = 0; half < 2; ++half) {
           const int row = group + half * 8;
           const int column = tile * 8 + thread_in_group * 2;
           if (column >= io_dim) {
             continue;
           }
           const int64_t index =
               out_base + static_cast<int64_t>(row) * io_dim + column;
           const float2 dv_pair =
               *reinterpret_cast<const float2*>(dv_continuation + index);
           const float2 dk_pair =
               *reinterpret_cast<const float2*>(dk_continuation + index);
           dv_acc[local][tile].v[half * 2] = dv_pair.x;
           dv_acc[local][tile].v[half * 2 + 1] = dv_pair.y;
           dk_acc[local][tile].v[half * 2] = dk_pair.x;
           dk_acc[local][tile].v[half * 2 + 1] = dk_pair.y;
         }
       }
     }
   }

   constexpr int kSrChunks = HeadDim / 8;
   const auto stage_qdo_and_bias = [&](int query_start, int parity) {
     __nv_bfloat16* buf_q =
         sq + parity * kOuterLeaf * kSrQueryTile * HeadDim;
     __nv_bfloat16* buf_do =
         sdo + parity * kOuterLeaf * kSrQueryTile * HeadDim;
     __nv_bfloat16* buf_bias =
         sbias_tile + parity * kSrQueryTile * kSrBiasStride;
 #pragma unroll 1
     for (int slot = 0; slot < staged_slots; ++slot) {
       __nv_bfloat16* dst_q = buf_q + slot * kSrQueryTile * HeadDim;
       __nv_bfloat16* dst_do = buf_do + slot * kSrQueryTile * HeadDim;
       const int64_t plane = slot_plane(slot);
       const int64_t tensor_base = (plane * heads + head) *
           static_cast<int64_t>(sequence) * io_dim;
       const int64_t dout_base =
           plane * static_cast<int64_t>(sequence) * heads * io_dim +
           static_cast<int64_t>(head) * io_dim;
       for (int chunk = threadIdx.x; chunk < kSrQueryTile * kSrChunks;
            chunk += kThreadsWg) {
         const int row = chunk / kSrChunks;
         const int col = (chunk - row * kSrChunks) * 8;
         const int gq = query_start + row;
         const int dst_index = row * HeadDim + wg_swz_col<HeadDim>(row, col);
         if (col < io_dim) {
           sr_cp_async_16(
               dst_q + dst_index,
               q + tensor_base + static_cast<int64_t>(gq) * io_dim + col);
           sr_cp_async_16(
               dst_do + dst_index,
               d_output + dout_base +
                   static_cast<int64_t>(gq) * heads * io_dim + col);
         } else {
           *reinterpret_cast<uint4*>(dst_q + dst_index) =
               uint4{0U, 0U, 0U, 0U};
           *reinterpret_cast<uint4*>(dst_do + dst_index) =
               uint4{0U, 0U, 0U, 0U};
         }
       }
     }
     for (int chunk = threadIdx.x; chunk < kSrQueryTile * (kSrKeyBlock / 8);
          chunk += kThreadsWg) {
       const int row = chunk / (kSrKeyBlock / 8);
       const int key = (chunk - row * (kSrKeyBlock / 8)) * 8;
       sr_cp_async_16(
           buf_bias + row * kSrBiasStride + key,
           bias + bias_base +
               static_cast<int64_t>(query_start + row) * sequence +
               key_start + key);
     }
     sr_cp_async_commit();
   };

   stage_qdo_and_bias(query_base, 0);

   const int query_end = query_base + query_rows;
 #pragma unroll 1
   for (int query_start = query_base; query_start < query_end;
        query_start += kSrQueryTile) {
     const int parity = ((query_start - query_base) / kSrQueryTile) & 1;
     sr_cp_async_wait_all();
     wg_proxy_fence();
     __syncthreads();
     if (query_start + kSrQueryTile < query_end) {
       stage_qdo_and_bias(query_start + kSrQueryTile, parity ^ 1);
     }
     const __nv_bfloat16* cur_q =
         sq + parity * kOuterLeaf * kSrQueryTile * HeadDim;
     const __nv_bfloat16* cur_do =
         sdo + parity * kOuterLeaf * kSrQueryTile * HeadDim;
     const __nv_bfloat16* cur_bias =
         sbias_tile + parity * kSrQueryTile * kSrBiasStride;
 #pragma unroll
     for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
       const int slot = warpgroup * kSlotsPerWarpgroup + local;
       if (single_outer && slot != 0) {
         continue;
       }
       const __nv_bfloat16* tile_k = sk + slot * kSrKeyBlock * HeadDim;
       const __nv_bfloat16* tile_v = sv + slot * kSrKeyBlock * HeadDim;
       const __nv_bfloat16* tile_q = cur_q + slot * kSrQueryTile * HeadDim;
       const __nv_bfloat16* tile_do = cur_do + slot * kSrQueryTile * HeadDim;

       // S^T = K.Q^T and dP^T = V.dO^T: A = the slot's 64-key K/V tile,
       // B = its 32-query q/dO tile (the dkdv ss encoding, proven bitwise
       // vs the mma.sync chain).  Accumulator rows = this warp's 16 keys.
       constexpr uint64_t kSsMode =
           HeadDim == 16 ? 3 : (HeadDim == 32 ? 2 : 1);
       constexpr uint64_t kSsLbo = 64;
       constexpr uint64_t kSsSbo = HeadDim;
       Acc score[kSrQueryTile / 8];
       Acc dp[kSrQueryTile / 8];
 #pragma unroll
       for (int column = 0; column < kSrQueryTile / 8; ++column) {
         score[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
         dp[column] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
       }
       wg_wgmma_fence();
 #pragma unroll
       for (int chain = 0; chain < HeadDim / 16; ++chain) {
         const uint64_t chain_off = static_cast<uint64_t>(chain) * 32;
         const uint64_t desc_k =
             ((static_cast<uint64_t>(shared_u32(tile_k)) + chain_off) >> 4) |
             (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
         const uint64_t desc_q =
             ((static_cast<uint64_t>(shared_u32(tile_q)) + chain_off) >> 4) |
             (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
         wg_ss_wgmma_m64n32k16(score, desc_k, desc_q, chain != 0);
       }
 #pragma unroll
       for (int chain = 0; chain < HeadDim / 16; ++chain) {
         const uint64_t chain_off = static_cast<uint64_t>(chain) * 32;
         const uint64_t desc_v =
             ((static_cast<uint64_t>(shared_u32(tile_v)) + chain_off) >> 4) |
             (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
         const uint64_t desc_do =
             ((static_cast<uint64_t>(shared_u32(tile_do)) + chain_off) >> 4) |
             (kSsLbo << 16) | (kSsSbo << 32) | (kSsMode << 62);
         wg_ss_wgmma_m64n32k16(dp, desc_v, desc_do, chain != 0);
       }
       wg_wgmma_commit();
       wg_wgmma_wait_zero();

       // Elementwise recompute in the [key, query] accumulator layout:
       // P^T = exp(S^T*scale + mask[key] + bias[q,k] - LSE[q]);
       // dS^T = P^T * (dP^T - delta[q]).  Column index = query.
       float* tree = sbias_acc +
           warpgroup * (kSrQueryTile * kSrTreeStride);
       __nv_bfloat16* ds_tile = sds + slot * kSrQueryTile * kSrDsStride;
-      const float slot_mult = outer_mult[slot];
+      const float slot_mult = slot_is_active(slot) ? 1.0F : 0.0F;
       const float* slot_mask = smask + slot * kSrKeyBlock;
       const int64_t stats_base = (slot_plane(slot) * heads + head) *
           static_cast<int64_t>(sequence);
+      // Once FP32 dS is formed, retain only the BF16 operands used by MMA.
+      uint32_t p_packed[kSrQueryTile / 8][2];
+      uint32_t ds_packed[kSrQueryTile / 8][2];
 #pragma unroll
       for (int column = 0; column < kSrQueryTile / 8; ++column) {
 #pragma unroll
         for (int half = 0; half < 2; ++half) {
           const int key_row = warp_key + group + half * 8;
           const float mask_value = slot_mask[key_row];
 #pragma unroll
           for (int pair = 0; pair < 2; ++pair) {
             const int query_in_tile = column * 8 + thread_in_group * 2 +
                 pair;
             const int query = query_start + query_in_tile;
             const int index = half * 2 + pair;
             const int64_t stats = stats_base + query;
             const float lse = logsumexp[stats];
             const float value = score[column].v[index] * scale +
                 mask_value +
                 __bfloat162float(
                     cur_bias[query_in_tile * kSrBiasStride + key_row]);
             const float probability =
                 lse == -std::numeric_limits<float>::infinity()
                 ? 0.0F
                 : __expf(value - lse);
             float d_score =
                 probability * (dp[column].v[index] - delta[stats]);
             if constexpr (!OuterEqSeq) {
               d_score *= slot_mult;
             }
             float p_value = probability;
             if constexpr (!OuterEqSeq) {
               p_value *= slot_mult;
             }
             dp[column].v[index] = p_value;
             score[column].v[index] = d_score;
             const int cell = query_in_tile * kSrTreeStride + key_row;
             if (local == 0) {
               tree[cell] = d_score;
             } else {
               tree[cell] = __fadd_rn(tree[cell], d_score);
             }
             ds_tile[query_in_tile * kSrDsStride + key_row] =
                 __float2bfloat16_rn(d_score);
           }
+          p_packed[column][half] = pack_bf16_pair(
+              dp[column].v[half * 2], dp[column].v[half * 2 + 1]);
+          ds_packed[column][half] = pack_bf16_pair(
+              score[column].v[half * 2], score[column].v[half * 2 + 1]);
         }
       }

       // dV += P^T @ dO; dK += dS^T @ Q -- contraction over the 32 queries,
-      // the exact dkdv chain (A = packed accumulators, B = rowtrans).
+      // the exact dkdv chain (A = packed BF16 operands, B = rowtrans).
 #pragma unroll
       for (int chunk = 0; chunk < kSrQueryTile / 16; ++chunk) {
         uint32_t p_frag[4];
         uint32_t ds_frag[4];
-        p_frag[0] = pack_bf16_pair(
-            dp[chunk * 2].v[0], dp[chunk * 2].v[1]);
-        p_frag[1] = pack_bf16_pair(
-            dp[chunk * 2].v[2], dp[chunk * 2].v[3]);
-        p_frag[2] = pack_bf16_pair(
-            dp[chunk * 2 + 1].v[0], dp[chunk * 2 + 1].v[1]);
-        p_frag[3] = pack_bf16_pair(
-            dp[chunk * 2 + 1].v[2], dp[chunk * 2 + 1].v[3]);
-        ds_frag[0] = pack_bf16_pair(
-            score[chunk * 2].v[0], score[chunk * 2].v[1]);
-        ds_frag[1] = pack_bf16_pair(
-            score[chunk * 2].v[2], score[chunk * 2].v[3]);
-        ds_frag[2] = pack_bf16_pair(
-            score[chunk * 2 + 1].v[0], score[chunk * 2 + 1].v[1]);
-        ds_frag[3] = pack_bf16_pair(
-            score[chunk * 2 + 1].v[2], score[chunk * 2 + 1].v[3]);
+        p_frag[0] = p_packed[chunk * 2][0];
+        p_frag[1] = p_packed[chunk * 2][1];
+        p_frag[2] = p_packed[chunk * 2 + 1][0];
+        p_frag[3] = p_packed[chunk * 2 + 1][1];
+        ds_frag[0] = ds_packed[chunk * 2][0];
+        ds_frag[1] = ds_packed[chunk * 2][1];
+        ds_frag[2] = ds_packed[chunk * 2 + 1][0];
+        ds_frag[3] = ds_packed[chunk * 2 + 1][1];
 #pragma unroll
         for (int tile = 0; tile < HeadDim / 8; ++tile) {
           uint32_t b_frag[2];
           ldmatrix_b_rowtrans_sw_sr<HeadDim>(
               tile_do, chunk * 16, tile * 8, lane, b_frag);
           mma_m16n8k16(dv_acc[local][tile], p_frag, b_frag);
           ldmatrix_b_rowtrans_sw_sr<HeadDim>(
               tile_q, chunk * 16, tile * 8, lane, b_frag);
           mma_m16n8k16(dk_acc[local][tile], ds_frag, b_frag);
         }
       }
     }

     // Tree tiles and dS tiles complete for this q-tile.
     __syncthreads();

     // dBias: combine the warpgroup pair trees and single-write the leaf
     // plane cells [q, kb-keys] (same values, cells, and order as the
     // two-pass kernel's fixed tree).
     const int64_t plane_tile_base = plane_base +
         static_cast<int64_t>(query_start - query_base) * sequence +
         key_start;
     for (int element = threadIdx.x; element < kSrQueryTile * kSrKeyBlock;
          element += kThreadsWg) {
       const int row = element / kSrKeyBlock;
       const int key = element - row * kSrKeyBlock;
       const int cell = row * kSrTreeStride + key;
       const float tile_sum = single_outer
           ? sbias_acc[cell]
           : __fadd_rn(
                 sbias_acc[cell],
                 sbias_acc[kSrQueryTile * kSrTreeStride + cell]);
       workspace[plane_tile_base + static_cast<int64_t>(row) * sequence +
                 key] = __float2bfloat16_rn(tile_sum);
     }

     // dQ partial for this (kb, q-tile): dq += dS @ K over the CTA's 64
     // keys.  Each warpgroup serves its own slot pair; warps 0/1 take the
     // first slot's row halves, warps 2/3 the second's.
     {
       const int local = warp_in_group >> 1;
       const int slot = warpgroup * kSlotsPerWarpgroup + local;
       const int row_half = warp_in_group & 1;
-      if (outer_mult[slot] != 0.0F) {
+      if (slot_is_active(slot)) {
         const __nv_bfloat16* ds_tile =
             sds + slot * kSrQueryTile * kSrDsStride +
             row_half * kQueryTile * kSrDsStride;
         const __nv_bfloat16* tile_k = sk + slot * kSrKeyBlock * HeadDim;
         Acc dq_acc[HeadDim / 8];
 #pragma unroll
         for (int tile = 0; tile < HeadDim / 8; ++tile) {
           dq_acc[tile] = Acc{{0.0F, 0.0F, 0.0F, 0.0F}};
         }
 #pragma unroll
         for (int chunk = 0; chunk < kSrKeyBlock / 16; ++chunk) {
           uint32_t a_frag[4];
           ldmatrix_a16x16(ds_tile, kSrDsStride, chunk * 16, lane, a_frag);
           #pragma unroll
           for (int tile = 0; tile < HeadDim / 8; ++tile) {
             uint32_t k_frag[2];
             ldmatrix_b_keytrans_sw<HeadDim>(
                 tile_k, chunk * 16, tile * 8, lane, k_frag);
             mma_m16n8k16(dq_acc[tile], a_frag, k_frag);
           }
         }
         // Partial layout matches the contiguous [B,O,H,N,D] output per
         // key block: idx = ((kb * total_planes + plane) * heads + head)
         //                  * sequence + q) * io_dim + col
         const int64_t plane = slot_plane(slot);
         const int64_t idx_base =
             (((static_cast<int64_t>(key_block) * total_planes + plane) *
                   heads +
               head) *
                  static_cast<int64_t>(sequence) +
              (query_start + row_half * kQueryTile)) *
             io_dim;
 #pragma unroll
         for (int tile = 0; tile < HeadDim / 8; ++tile) {
 #pragma unroll
           for (int half = 0; half < 2; ++half) {
             const int row = group + half * 8;
             const int column = tile * 8 + thread_in_group * 2;
             if (column >= io_dim) {
               continue;
             }
             const int64_t index =
                 idx_base + static_cast<int64_t>(row) * io_dim + column;
             *reinterpret_cast<uint32_t*>(dq_partials + index) =
                 pack_bf16_pair(
                     dq_acc[tile].v[half * 2],
                     dq_acc[tile].v[half * 2 + 1]);
           }
         }
       }
     }

   }

   // Epilogue: dK/dV for this warp's 16 key rows, both slots of its
   // warpgroup.  Intermediate query stripes preserve the raw FP32 MMA
   // accumulators; the final stripe scales dK once and stores both BF16.
   const bool final_query_stripe = query_base + query_rows == sequence;
 #pragma unroll
   for (int local = 0; local < kSlotsPerWarpgroup; ++local) {
     const int slot = warpgroup * kSlotsPerWarpgroup + local;
-    if (outer_mult[slot] == 0.0F) {
+    if (!slot_is_active(slot)) {
       continue;
     }
     const int64_t out_base = (slot_plane(slot) * heads + head) *
         static_cast<int64_t>(sequence) * io_dim +
         static_cast<int64_t>(key_start + warp_key) * io_dim;
 #pragma unroll
     for (int tile = 0; tile < HeadDim / 8; ++tile) {
 #pragma unroll
       for (int half = 0; half < 2; ++half) {
         const int row = group + half * 8;
         const int column = tile * 8 + thread_in_group * 2;
         if (column >= io_dim) {
           continue;
         }
         const int64_t index =
             out_base + static_cast<int64_t>(row) * io_dim + column;
         if (final_query_stripe) {
           *reinterpret_cast<uint32_t*>(out_dv + index) = pack_bf16_pair(
               dv_acc[local][tile].v[half * 2],
               dv_acc[local][tile].v[half * 2 + 1]);
           *reinterpret_cast<uint32_t*>(out_dk + index) = pack_bf16_pair(
               dk_acc[local][tile].v[half * 2] * scale,
               dk_acc[local][tile].v[half * 2 + 1] * scale);
         } else {
           // Preserve the audited scalar continuation contract exactly:
           //   dv_continuation[index] = {dv0, dv1}
           //   dk_continuation[index] = {dk0, dk1}
           // The aligned float2 stores only combine each adjacent pair into
           // one transaction; they do not change values or stripe ordering.
           *reinterpret_cast<float2*>(dv_continuation + index) = make_float2(
               dv_acc[local][tile].v[half * 2],
               dv_acc[local][tile].v[half * 2 + 1]);
           *reinterpret_cast<float2*>(dk_continuation + index) = make_float2(
               dk_acc[local][tile].v[half * 2],
               dk_acc[local][tile].v[half * 2 + 1]);
         }
       }
     }
   }
 }
```
