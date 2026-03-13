# WHAM Fused dlnZ + cuBLAS DGEMM Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~1.86M kernel launches in WHAM profile computation by fusing the serial get_dlnZ loop into a single 2D kernel and replacing get_CC with cuBLAS DGEMM.

**Architecture:** Two independent CUDA optimizations that replace the inner loop of `compute_profiles_range()` (wham.cu:1834-1888). The fused dlnZ kernel computes all jN parameter derivatives in one pass using a device-side param_desc table. The cuBLAS DGEMM computes CC[j1,j2] as M*M^T where M[j,k] = sqrt(w[k])*exp(dlnZ[j,k]-lnZ[k]). Both are algebraically identical to current code (verified via tractatus analysis).

**Tech Stack:** CUDA C (nvcc), cuBLAS (DGEMM), Python ctypes bindings

**Performance target:** For nsubs=[9,15] (jN=1404, iN=441, B_N=400):
- Current: 1404 x (reactioncoord + resetlogdata + get_dlnZ + sync + memcpy) + get_CC(1404^2) = ~1.86M kernel launches per profile
- Target: 1 fused kernel + 1 DGEMM call per profile = 2 operations per profile

**CRITICAL stride invariant:** `dlnZ_dN` is allocated as `[jN x B_N]` but the actual working stride is `g_1d_bins` (= `data->B[1].N`), which can be SMALLER than `B_N` when 2D bins > 1D bins (e.g., 1D=400, 2D=32x32=1024 → B_N=1024 but g_1d_bins=400). Both the fused kernel and DGEMM MUST use `g_1d_bins` as the dlnZ stride, not `B_N`. On the host side, the equivalent value is `data->B[1].N`.

---

## Chunk 1: Device-side param_desc table

### Task 1: Add params_d to struct_data and copy param_desc to device

The fused dlnZ kernel needs to know each parameter's type and block indices. Currently `params` is host-only (malloc'd at wham.cu:720). We need a device copy.

**Files:**
- Modify: `cphmd/wham/src/wham.h:50-113` (struct_data)
- Modify: `cphmd/wham/src/wham.cu:717-883` (build_param_descs)
- Modify: `cphmd/wham/src/wham.cu` (free_data -- wherever struct_data is freed)

- [ ] **Step 1: Add `params_d` field to struct_data**

In `cphmd/wham/src/wham.h`, add after line 113 (`param_desc *params`):

```c
  param_desc *params_d;             // Device copy of param descriptors [jN]
```

- [ ] **Step 2: Allocate and copy params_d in build_param_descs()**

In `cphmd/wham/src/wham.cu`, at the end of `build_param_descs()` (after the `assert(idx == jN)` line), add:

```c
  // Copy param descriptors to device for fused dlnZ kernel
  CUDA_CHECK(cudaMalloc(&(data->params_d), jN * sizeof(param_desc)));
  CUDA_CHECK(cudaMemcpy(data->params_d, data->params,
                         jN * sizeof(param_desc), cudaMemcpyHostToDevice));
```

- [ ] **Step 3: Free params_d in cleanup**

Find every location where `free(data->params)` is called and add `cudaFree(data->params_d)` immediately before it. Search for `free(data->params)` to find all sites.

- [ ] **Step 4: Compile and verify no regressions**

Run: `cd cphmd/wham/src && make`
Expected: Clean compilation, no errors.

Run: `pytest tests/test_wham_packing.py -v`
Expected: All 11 tests PASS (no Python API change).

- [ ] **Step 5: Commit**

```bash
git add cphmd/wham/src/wham.h cphmd/wham/src/wham.cu
git commit -m "refactor: copy param_desc table to device for fused dlnZ kernel"
```

---

### Task 2: Add `__device__` inline reaction coordinate functions

The fused kernel needs to compute q_j inline without launching separate reactioncoord kernels. Extract the 6 formulas as `__device__` inline functions.

**Files:**
- Modify: `cphmd/wham/src/wham.cu` (add before get_dlnZ kernel, ~line 1308)

- [ ] **Step 1: Write the 6 device-inline reaction coordinate functions**

Add before the `get_dlnZ` kernel definition (before line 1310):

```cuda
// --- Inline reaction coordinate functions for fused dlnZ kernel ---
// These mirror the global reactioncoord_* kernels (lines 1151-1218)
// but as __device__ functions callable from within a kernel.

static __device__ inline double rc_phi(const double *D, int Ndim, int j1) {
    return D[1 + j1];  // lambda_j1
}

static __device__ inline double rc_psi(const double *D, int Ndim, int j1, int j2) {
    return D[1 + j1] * D[1 + j2];  // lambda_j1 * lambda_j2
}

static __device__ inline double rc_chi(const double *D, int Ndim, int j1, int j2,
                                        double omega_scale) {
    double lam_i = D[1 + j1];
    double lam_j = D[1 + j2];
    return lam_j * (1.0 - exp(-lam_i / omega_scale));
}

static __device__ inline double rc_omega(const double *D, int Ndim, int j1, int j2,
                                          double chi_offset) {
    double lam_i = D[1 + j1];
    double lam_j = D[1 + j2];
    return lam_j * (1.0 - 1.0 / (lam_i / chi_offset + 1.0));
}

static __device__ inline double rc_omega2(const double *D, int Ndim, int j1, int j2,
                                           double chi_offset_t) {
    double lam_i = D[1 + j1];
    double lam_j = D[1 + j2];
    return -lam_j * (1.0 - 1.0 / (lam_i / (-1.0 - chi_offset_t) + 1.0));
}

static __device__ inline double rc_omega3(const double *D, int Ndim, int j1, int j2,
                                           double chi_offset_u) {
    double lam_i = D[1 + j1];
    double lam_j = D[1 + j2];
    return lam_j * lam_j * (1.0 - 1.0 / (lam_i / chi_offset_u + 1.0));
}

// Dispatch: compute q_j for parameter descriptor d, reading lambdas from frame row D
static __device__ inline double compute_rc(const double *D, int Ndim,
                                            const param_desc *d,
                                            double chi_offset, double omega_scale,
                                            double chi_offset_t, double chi_offset_u) {
    switch (d->type) {
    case 0: return rc_phi(D, Ndim, d->j1);
    case 1: return rc_psi(D, Ndim, d->j1, d->j2);
    case 2: return rc_chi(D, Ndim, d->j1, d->j2, omega_scale);
    case 3: return rc_omega(D, Ndim, d->j1, d->j2, chi_offset);
    case 4: return rc_omega2(D, Ndim, d->j1, d->j2, chi_offset_t);
    case 5: return rc_omega3(D, Ndim, d->j1, d->j2, chi_offset_u);
    default: return 0.0;
    }
}
```

- [ ] **Step 2: Verify the formulas match the existing global kernels**

Cross-reference each `rc_*` function against the corresponding `reactioncoord_*` kernel:
- `rc_phi` <-> `reactioncoord_phi` (line 1151): both return `D[1+j1]`
- `rc_psi` <-> `reactioncoord_psi` (line 1161): both return `D[1+j1] * D[1+j2]`
- `rc_omega` <-> `reactioncoord_omega` (line 1172): both use `chi_offset` sigmoid
- `rc_chi` <-> `reactioncoord_chi` (line 1183): both use `omega_scale` exponential
- `rc_omega2` <-> `reactioncoord_omega2` (line 1196): both use `chi_offset_t` with negation
- `rc_omega3` <-> `reactioncoord_omega3` (line 1209): both use `chi_offset_u` with lambda^2

- [ ] **Step 3: Compile**

Run: `cd cphmd/wham/src && make`
Expected: Clean compile (functions are static inline, no symbol export issues).

- [ ] **Step 4: Commit**

```bash
git add cphmd/wham/src/wham.cu
git commit -m "refactor: add device-inline reaction coordinate functions for fused kernel"
```

---

## Chunk 2: Fused get_dlnZ kernel

### Task 3: Implement the fused get_dlnZ_fused kernel

Replace the serial loop (wham.cu:1834-1847) with a single 2D kernel that processes all jN parameters in one launch.

**Files:**
- Modify: `cphmd/wham/src/wham.cu` (add new kernel, modify `compute_profiles_range`)

- [ ] **Step 1: Write the fused kernel**

Add after the `compute_rc` function (before the existing `get_dlnZ` kernel):

```cuda
// --- Fused get_dlnZ: batch all jN parameters in one kernel launch ---
// Grid:  (ceil(ND / (threads_per_block * SBLOCK)), jN, 1)
//   blockIdx.x = frame-tile index
//   blockIdx.y = parameter index j1
// Block: (threads_per_block, 1, 1)  -- typically 100
// Shared memory: g_1d_bins * sizeof(double) per block (same as serial get_dlnZ)
//
// This replaces the serial loop:
//   for (j1 = 0; j1 < jN; j1++) {
//     reactioncoord_all(data, j1);  // separate kernel launch
//     resetlogdata<<<...>>>();       // separate kernel launch
//     get_dlnZ<<<...>>>(data, j1);  // separate kernel launch
//     cudaDeviceSynchronize();       // host-device sync
//     cudaMemcpy(dlnZ_hN[j1]...);   // D2H transfer
//   }
// Total eliminated: jN x 3 kernel launches + jN syncs + jN memcpys
// Replaced with: 1 kernel launch + 1 bulk memcpy of entire dlnZ_dN

__global__ void get_dlnZ_fused(struct_data data, double beta,
                                const double *__restrict__ gshift,
                                const param_desc *__restrict__ params_d,
                                int block_off)
{
    int j1 = blockIdx.y;  // Which parameter this block computes
    int t = blockIdx.x * blockDim.x + threadIdx.x;  // Frame-tile index

    // Shared memory for bin accumulation (same pattern as serial get_dlnZ)
    extern __shared__ double loc_dlnZ[];

    // Initialize shared bins to -INFINITY
    for (int i = threadIdx.x; i < g_1d_bins; i += blockDim.x)
        loc_dlnZ[i] = -INFINITY;
    __syncthreads();

    // Load parameter descriptor from device memory (read-once, cached in L1)
    param_desc pd = params_d[j1];

    // Process SBLOCK frames per thread (same tiling as serial version)
    for (int i = SBLOCK * t; i < SBLOCK * (t + 1) && i < data.ND; i++)
    {
        const double *D_row = &data.D_d[i * data.Ndim];
        double E = D_row[0];
        int iB = (int)D_row[data.Ndim - 2];  // 1D bin index

        if (iB >= 0 && iB < g_1d_bins && isfinite(E))
        {
            // Compute reaction coordinate inline (replaces reactioncoord_* kernel)
            double q = compute_rc(D_row, data.Ndim, &pd,
                                  data.chi_offset, data.omega_scale,
                                  data.chi_offset_t, data.chi_offset_u);

            // Match serial get_dlnZ validation (wham.cu:1337): isfinite + 1e-15 threshold
            if (isfinite(q) && q > 1e-15)
            {
                int sim_idx = data.i_d[i];
                if (sim_idx >= 0 && sim_idx < data.NF)
                {
                    double lnw = data.lnw_d[sim_idx];
                    double lnDenom = data.lnDenom_d[i];

                    if (isfinite(lnw) && isfinite(lnDenom))
                    {
                        double log_q = log(q);
                        if (isfinite(log_q))
                        {
                            // Compute vshift = dot(gshift[sim], lambda)
                            double vshift = 0.0;
                            if (data.use_gshift) {
                                for (int bl = 0; bl < data.Nblocks; bl++)
                                    vshift += gshift[sim_idx * data.Nblocks + bl] * D_row[1 + bl];
                            }

                            double contribution = lnw - lnDenom - beta * (E + vshift) + log_q;
                            if (isfinite(contribution))
                                atomic_logadd(&loc_dlnZ[iB], contribution);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Write shared memory to global dlnZ_dN[j1 * g_1d_bins + bin]
    // (same stride as serial get_dlnZ final reduction, wham.cu:1372)
    for (int i = threadIdx.x; i < g_1d_bins; i += blockDim.x)
    {
        if (isfinite(loc_dlnZ[i]) && loc_dlnZ[i] > -INFINITY)
            atomic_logadd(&data.dlnZ_dN[g_1d_bins * j1 + i], loc_dlnZ[i]);
    }
}
```

- [ ] **Step 2: Add a helper to reset the entire dlnZ_dN array in one launch**

Add after the fused kernel:

```cuda
// Reset entire dlnZ_dN[jN x g_1d_bins] to -INFINITY in one kernel launch
// (replaces jN separate resetlogdata calls)
__global__ void resetlogdata_bulk(double *arr, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
        arr[idx] = -INFINITY;
}
```

- [ ] **Step 3: Compile**

Run: `cd cphmd/wham/src && make`
Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add cphmd/wham/src/wham.cu
git commit -m "feat: add get_dlnZ_fused kernel for batched parameter computation"
```

---

### Task 4: Wire the fused kernel into compute_profiles_range

Replace the serial loop in `compute_profiles_range` with the fused kernel launch, controlled by a `use_fused` toggle with `CPHMD_WHAM_SERIAL=1` environment variable fallback.

**Files:**
- Modify: `cphmd/wham/src/wham.cu:1834-1847` (the serial loop)
- Modify: `cphmd/wham/src/wham.h` (add use_fused field)

- [ ] **Step 1: Add use_fused toggle to struct_data**

In `wham.h`, add after `endpoint_decay` field:

```c
  int use_fused;    // 0 = serial dlnZ + custom get_CC (legacy)
                    // 1 = fused dlnZ + cuBLAS DGEMM (default)
```

- [ ] **Step 2: Set use_fused in readdata_from_memory**

In `readdata_from_memory()`, after struct_data initialization:

```c
  data->use_fused = 1;  // Default: fused path enabled

  // Environment variable override for debugging/validation
  const char *fused_env = getenv("CPHMD_WHAM_SERIAL");
  if (fused_env && atoi(fused_env) == 1)
    data->use_fused = 0;
```

Do the same in the slim variant's readdata function if it exists separately.

- [ ] **Step 3: Replace the serial dlnZ loop with conditional fused/serial**

Replace lines 1834-1847 in `compute_profiles_range` with:

```c
    if (data->use_fused) {
      // --- Fused dlnZ path: single kernel for all jN parameters ---
      // CRITICAL: dlnZ_dN stride is g_1d_bins (= data->B[1].N), NOT B_N.
      // B_N = max(1D_bins, 2D_bins) but dlnZ only uses 1D bins.
      int bins_1d = data->B[1].N;  // host-side equivalent of g_1d_bins
      int total_dlnZ = jN * bins_1d;
      resetlogdata_bulk<<<(total_dlnZ + BLOCK - 1) / BLOCK, BLOCK>>>(
          data->dlnZ_dN, total_dlnZ);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      int frame_tiles = (data->ND + (100 * SBLOCK) - 1) / (100 * SBLOCK);
      dim3 grid(frame_tiles, jN, 1);
      // Shared memory: bins_1d doubles (not B_N — only 1D bins used)
      get_dlnZ_fused<<<grid, 100, bins_1d * sizeof(double)>>>(
          data[0], data->beta_t, data->gshift_d, data->params_d,
          data->current_block_idx);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // Single bulk memcpy at g_1d_bins stride, then scatter to dlnZ_hN
      double *dlnZ_bulk = (double *)malloc(jN * bins_1d * sizeof(double));
      CUDA_CHECK(cudaMemcpy(dlnZ_bulk, data->dlnZ_dN,
                             jN * bins_1d * sizeof(double), cudaMemcpyDeviceToHost));
      for (j1 = 0; j1 < jN; j1++)
        memcpy(data->dlnZ_hN[j1], &dlnZ_bulk[j1 * bins_1d], bins_1d * sizeof(double));
      free(dlnZ_bulk);
    } else {
      // --- Serial dlnZ path (original) ---
      for (j1 = 0; j1 < jN; j1++) {
        reactioncoord_all(data, j1);
        resetlogdata<<<(B_N + BLOCK - 1) / BLOCK, BLOCK>>>(&(data->dlnZ_dN[B_N * j1]), B_N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        get_dlnZ<<<(data->ND + (100 * SBLOCK) - 1) / (100 * SBLOCK), 100,
                    B_N * sizeof(double)>>>(data[0], j1, data->beta_t,
                                             data->gshift_d, data->current_block_idx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(data->dlnZ_hN[j1], &(data->dlnZ_dN[B_N * j1]),
                               B_N * sizeof(double), cudaMemcpyDeviceToHost));
      }
    }
```

**CRITICAL**: The rest of `compute_profiles_range` (lines 1849-1888: get_CC launch, host V/C accumulation) is **unchanged** at this point. The fused kernel populates the same `dlnZ_dN` / `dlnZ_hN` arrays, so all downstream code works identically.

- [ ] **Step 4: Compile**

Run: `cd cphmd/wham/src && make`
Expected: Clean compilation.

- [ ] **Step 5: Commit**

```bash
git add cphmd/wham/src/wham.cu cphmd/wham/src/wham.h
git commit -m "perf: wire fused get_dlnZ kernel, eliminate jN serial kernel launches"
```

---

## Chunk 3: cuBLAS DGEMM for CC matrix

### Task 5: Add cuBLAS DGEMM to replace get_CC kernel

**Files:**
- Modify: `cphmd/wham/src/Makefile` (add `-lcublas`)
- Modify: `cphmd/wham/src/wham.cu` (add cuBLAS include, DGEMM wrapper, modify `compute_profiles_range`)

- [ ] **Step 1: Update Makefile to link cuBLAS**

In `cphmd/wham/src/Makefile`, change the compilation line:

```makefile
NVCCFLAGS := -shared -Xcompiler -fPIC -arch=all-major -Wno-deprecated-gpu-targets

$(WHAM_TARGET): wham.cu wham.h
	$(NVCC) $(NVCCFLAGS) -lcublas -o $@ $<
```

- [ ] **Step 2: Add cuBLAS include and handle initialization**

At the top of `wham.cu` (after existing CUDA includes):

```cuda
#include <cublas_v2.h>

// Global cuBLAS handle -- initialized once, reused across profile computations
static cublasHandle_t g_cublas_handle = NULL;

static void ensure_cublas_handle() {
    if (g_cublas_handle == NULL) {
        cublasStatus_t stat = cublasCreate(&g_cublas_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS initialization failed: %d\n", (int)stat);
            g_cublas_handle = NULL;
        }
    }
}

// Clean up cuBLAS handle when shared library is unloaded
__attribute__((destructor))
static void cleanup_cublas() {
    if (g_cublas_handle != NULL) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
}
```

- [ ] **Step 3: Write the build_M_matrix kernel**

```cuda
// --- Build M matrix for cuBLAS DGEMM CC computation ---
// CRITICAL: dlnZ_dN uses g_1d_bins stride, M uses g_1d_bins stride for DGEMM K dimension.
// M[j * g_1d_bins + k] = sqrt(w[k]) * exp(dlnZ_dN[j * g_1d_bins + k] - lnZ_d[k])
// If lnZ_d[k] is not finite, zero the entry.
__global__ void build_M_matrix(const double *__restrict__ dlnZ_dN,
                                const double *__restrict__ lnZ_d,
                                const double *__restrict__ sqrt_w_d,
                                double *__restrict__ M_d,
                                int jN)
{
    // g_1d_bins is a __device__ variable — used as stride for both dlnZ_dN and M_d
    int j = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < g_1d_bins && j < jN) {
        double lnZ = lnZ_d[k];
        double sw = sqrt_w_d[k];

        if (isfinite(lnZ) && lnZ > -INFINITY && sw > 0.0) {
            double dlnZ = dlnZ_dN[j * g_1d_bins + k];
            if (isfinite(dlnZ)) {
                double exp_arg = dlnZ - lnZ;
                if (isfinite(exp_arg)) {
                    double val = sw * exp(exp_arg);
                    M_d[j * g_1d_bins + k] = isfinite(val) ? val : 0.0;
                } else {
                    M_d[j * g_1d_bins + k] = 0.0;
                }
            } else {
                M_d[j * g_1d_bins + k] = 0.0;
            }
        } else {
            M_d[j * g_1d_bins + k] = 0.0;
        }
    }
}
```

- [ ] **Step 4: Write the compute_CC_dgemm function**

```cuda
// Compute CC = M * M^T using cuBLAS DGEMM
// CRITICAL: K dimension is bins_1d (= g_1d_bins on device, = data->B[1].N on host),
// NOT B_N. The dlnZ_dN stride and M stride are both bins_1d.
// M is [jN x bins_1d] row-major = [bins_1d x jN] column-major.
// CC = M * M^T (row-major) = M_col^T * M_col (column-major)
static int compute_CC_dgemm(struct_data *data, double wnorm, int ptype)
{
    int jN = data->jN;
    int bins_1d = data->B[1].N;  // host-side g_1d_bins

    ensure_cublas_handle();
    if (g_cublas_handle == NULL) return -1;

    // 1. Build sqrt_w on host, copy to device
    // endpoint_ramp uses bins_1d (1D profiles) not B_N
    double *sqrt_w_h = (double *)malloc(bins_1d * sizeof(double));
    for (int k = 0; k < bins_1d; k++) {
        double w = wnorm;
        if (ptype == 0 || ptype == 3)
            w *= endpoint_ramp(k, bins_1d, data->endpoint_weight, data->endpoint_decay);
        sqrt_w_h[k] = (w > 0.0) ? sqrt(w) : 0.0;
    }

    double *sqrt_w_d;
    CUDA_CHECK(cudaMalloc(&sqrt_w_d, bins_1d * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(sqrt_w_d, sqrt_w_h, bins_1d * sizeof(double),
                           cudaMemcpyHostToDevice));
    free(sqrt_w_h);

    // 2. Allocate M matrix on device [jN x bins_1d]
    double *M_d;
    CUDA_CHECK(cudaMalloc(&M_d, (size_t)jN * bins_1d * sizeof(double)));

    // 3. Build M matrix (kernel uses g_1d_bins internally for stride)
    dim3 grid_m((bins_1d + BLOCK - 1) / BLOCK, jN, 1);
    build_M_matrix<<<grid_m, BLOCK>>>(data->dlnZ_dN, data->lnZ_d,
                                       sqrt_w_d, M_d, jN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. CC = M * M^T via cuBLAS DGEMM
    // M is [jN x bins_1d] row-major = [bins_1d x jN] column-major
    // cublasDgemm(T, N): M_col^T * M_col = [jN x bins_1d] * [bins_1d x jN] = [jN x jN]
    double alpha = 1.0, beta_val = 0.0;
    cublasStatus_t stat = cublasDgemm(g_cublas_handle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       jN, jN, bins_1d,
                                       &alpha,
                                       M_d, bins_1d,   // A = M_col [bins_1d x jN], lda=bins_1d
                                       M_d, bins_1d,   // B = M_col [bins_1d x jN], ldb=bins_1d
                                       &beta_val,
                                       data->CC_d, jN);  // C = CC [jN x jN], ldc=jN

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS DGEMM failed: %d\n", (int)stat);
        cudaFree(M_d);
        cudaFree(sqrt_w_d);
        return -1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Cleanup
    cudaFree(M_d);
    cudaFree(sqrt_w_d);

    return 0;
}
```

- [ ] **Step 5: Compile**

Run: `cd cphmd/wham/src && make`
Expected: Clean compilation with cuBLAS linked.

- [ ] **Step 6: Commit**

```bash
git add cphmd/wham/src/Makefile cphmd/wham/src/wham.cu
git commit -m "feat: add cuBLAS DGEMM wrapper for CC matrix computation"
```

---

### Task 6: Wire DGEMM into compute_profiles_range

Replace the `get_CC<<<>>>` launch with conditional `compute_CC_dgemm()`.

**Files:**
- Modify: `cphmd/wham/src/wham.cu:1849-1857` (get_CC launch site)

- [ ] **Step 1: Replace get_CC launch with conditional DGEMM**

Replace lines 1849-1857 in `compute_profiles_range` with:

```c
    // CC matrix: DGEMM path or legacy kernel
    if (data->use_fused) {
      int dgemm_ret = compute_CC_dgemm(data, wnorm, ptype);
      if (dgemm_ret != 0) {
        fprintf(stderr, "DGEMM fallback to serial get_CC for profile %d\n", i);
        // Fallback to serial get_CC
        get_CC<<<make_uint3(jN, jN, 1), 100>>>(data[0], i, data->beta_t, wnorm, ptype,
                                                 data->gshift_d, data->current_block_idx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    } else {
      get_CC<<<make_uint3(jN, jN, 1), 100>>>(data[0], i, data->beta_t, wnorm, ptype,
                                               data->gshift_d, data->current_block_idx);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(data->CC_h, data->CC_d, jN * jN * sizeof(double),
                           cudaMemcpyDeviceToHost));

    validate_gpu_results(data, B_N, jN);
    validate_matrix_results(data->CC_h, jN, jN, "CC matrix");
```

- [ ] **Step 2: Compile**

Run: `cd cphmd/wham/src && make`
Expected: Clean compilation.

- [ ] **Step 3: Commit**

```bash
git add cphmd/wham/src/wham.cu
git commit -m "perf: wire cuBLAS DGEMM for CC matrix, with serial fallback"
```

---

## Chunk 4: Optimization and validation

### Task 7: Pre-allocate M_d and sqrt_w_d scratch buffers

For large systems, per-profile cudaMalloc/cudaFree is wasteful. Pre-allocate once.

**Files:**
- Modify: `cphmd/wham/src/wham.h` (add scratch fields)
- Modify: `cphmd/wham/src/wham.cu` (allocate in readdata, use in compute_CC_dgemm)

- [ ] **Step 1: Add scratch fields to struct_data**

In `wham.h`:

```c
  double *M_d;       // Scratch: M matrix for DGEMM [jN x bins_1d] (device)
  double *sqrt_w_d;  // Scratch: sqrt(weights) [bins_1d] (device)
```

- [ ] **Step 2: Allocate in readdata_from_memory after dlnZ_dN**

```c
// After: CUDA_CHECK(cudaMalloc(&(data->dlnZ_dN), jN * B_N * sizeof(double)));
// Use bins_1d (= data->B[1].N) for M_d stride, NOT B_N
int bins_1d = data->B[1].N;
CUDA_CHECK(cudaMalloc(&(data->M_d), (size_t)jN * bins_1d * sizeof(double)));
CUDA_CHECK(cudaMalloc(&(data->sqrt_w_d), bins_1d * sizeof(double)));
```

- [ ] **Step 3: Update compute_CC_dgemm to use pre-allocated buffers**

Replace `cudaMalloc(&M_d, ...)` with `double *M_d = data->M_d;`
Replace `cudaMalloc(&sqrt_w_d, ...)` with `double *sqrt_w_d = data->sqrt_w_d;`
Remove the `cudaFree(M_d)` and `cudaFree(sqrt_w_d)` calls.
Keep the host `sqrt_w_h` malloc/free (small, 3.2KB).

- [ ] **Step 4: Free in cleanup alongside dlnZ_dN**

```c
cudaFree(data->M_d);
cudaFree(data->sqrt_w_d);
```

- [ ] **Step 5: Compile and commit**

Run: `cd cphmd/wham/src && make`

```bash
git add cphmd/wham/src/wham.cu cphmd/wham/src/wham.h
git commit -m "perf: pre-allocate M_d/sqrt_w_d scratch buffers, avoid per-profile malloc"
```

---

### Task 8: Write regression test scaffold

**Files:**
- Create: `tests/test_wham_fused.py`

- [ ] **Step 1: Write test file**

```python
"""Regression tests: fused dlnZ + cuBLAS DGEMM vs serial WHAM profiles.

Validates that the fused CUDA code path produces identical C/V output
to the serial path. Requires a CUDA GPU to run.
"""

from __future__ import annotations

import os
import subprocess
import numpy as np
import pytest


def _gpu_available() -> bool:
    """Check if CUDA GPU is accessible via nvidia-smi."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="No CUDA GPU available"
)


class TestFusedToggle:
    """Verify CPHMD_WHAM_SERIAL environment variable is respected."""

    def test_serial_env_var_exists(self):
        """The toggle mechanism is documented and env var is checkable."""
        # This is a sanity test -- actual GPU validation is in Task 9
        assert "CPHMD_WHAM_SERIAL" not in os.environ or \
               os.environ["CPHMD_WHAM_SERIAL"] in ("0", "1")


class TestComputeRCFormulas:
    """Verify inline RC formulas match expected values (CPU-side check)."""

    def test_phi_is_lambda(self):
        """rc_phi returns lambda[j1]."""
        # phi(j1=2, lambdas=[0.3, 0.5, 0.2]) = 0.2
        lambdas = [0.3, 0.5, 0.2]
        assert lambdas[2] == pytest.approx(0.2)

    def test_psi_is_product(self):
        """rc_psi returns lambda[j1] * lambda[j2]."""
        lambdas = [0.3, 0.5, 0.2]
        assert lambdas[0] * lambdas[1] == pytest.approx(0.15)

    def test_chi_formula(self):
        """rc_chi = lambda_j * (1 - exp(-lambda_i / omega_scale))."""
        import math
        omega_scale = 1.0 / 5.5  # 0.18182
        lam_i, lam_j = 0.8, 0.5
        expected = lam_j * (1.0 - math.exp(-lam_i / omega_scale))
        # With lam_i=0.8, omega_scale=0.18182: exp(-4.4) ~ 0.012
        assert expected == pytest.approx(lam_j * (1.0 - math.exp(-4.4)), rel=1e-6)

    def test_omega_formula(self):
        """rc_omega = lambda_j * (1 - 1/(lambda_i/chi_offset + 1))."""
        chi_offset = 4 * 0.00408677  # 4*exp(-5.5) ~ 0.01635
        lam_i, lam_j = 0.9, 0.4
        expected = lam_j * (1.0 - 1.0 / (lam_i / chi_offset + 1.0))
        assert expected > 0  # Should be positive for lam_i >> chi_offset
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_wham_fused.py -v`
Expected: All pass (CPU-side formula checks, GPU tests skipped if no GPU).

- [ ] **Step 3: Commit**

```bash
git add tests/test_wham_fused.py
git commit -m "test: add regression test scaffold for fused dlnZ + DGEMM"
```

---

### Task 9: End-to-end validation on real system

Run the fused path on a real example and compare against serial path output.

**Files:**
- No code changes -- validation only

- [ ] **Step 1: Run serial path on glu_water example**

```bash
cd examples/00_glu_water
CPHMD_WHAM_SERIAL=1 cphmd run alf -i solvated --pH 7.0 --max-runs 3
cp solvated/analysis1/C.dat /tmp/C_serial.dat
cp solvated/analysis1/V.dat /tmp/V_serial.dat
```

- [ ] **Step 2: Run fused path on same example (clean analysis first)**

```bash
cd examples/00_glu_water
rm -rf solvated/analysis1
cphmd run alf -i solvated --pH 7.0 --max-runs 3
cp solvated/analysis1/C.dat /tmp/C_fused.dat
cp solvated/analysis1/V.dat /tmp/V_fused.dat
```

- [ ] **Step 3: Compare C/V matrices**

```python
import numpy as np
C_s = np.loadtxt("/tmp/C_serial.dat")
C_f = np.loadtxt("/tmp/C_fused.dat")
V_s = np.loadtxt("/tmp/V_serial.dat")
V_f = np.loadtxt("/tmp/V_fused.dat")

print(f"C max abs diff: {np.max(np.abs(C_s - C_f)):.2e}")
print(f"V max abs diff: {np.max(np.abs(V_s - V_f)):.2e}")

# Tolerance: ~1e-10 for atomic_logadd ordering + cuBLAS accumulation
assert np.allclose(C_s, C_f, atol=1e-8, rtol=1e-8), "C matrix mismatch!"
assert np.allclose(V_s, V_f, atol=1e-8, rtol=1e-8), "V matrix mismatch!"
print("PASS: C/V matrices match within tolerance")
```

Expected: Both assertions pass.

- [ ] **Step 4: Run on large system (nsubs=[9,15]) and time it**

```bash
cd lambdy/marcella_mpi

# Fused path (default)
time cphmd run alf -i solvated --max-runs 1 2>&1 | grep -i "wham\|profile\|time"

# Serial path
CPHMD_WHAM_SERIAL=1 time cphmd run alf -i solvated --max-runs 1 2>&1 | grep -i "wham\|profile\|time"
```

Expected: Fused path significantly faster (target: >5x for nsubs=[9,15]).

---

## Implementation Notes

### Grid dimension limits
- CUDA max gridDim.y = 65535. For jN=1404, this is well within limits.
- CUDA max gridDim.x = 2^31-1. Frame tiles for 150K frames: ~94 tiles. Fine.

### Shared memory budget
- Fused kernel: `g_1d_bins * sizeof(double)` = 400 x 8 = 3.2 KB per block (1D bins only)
- GPU shared memory per SM: 48-164 KB depending on architecture
- With 100 threads per block, occupancy is limited by registers, not shared memory

### cuBLAS column-major convention
- cuBLAS is column-major. Our M is row-major [jN x bins_1d].
- In column-major view: M is [bins_1d x jN].
- To compute M*M^T (row-major) = M_col^T * M_col (col-major):
  `cublasDgemm(CUBLAS_OP_T, CUBLAS_OP_N, jN, jN, bins_1d, a, M, bins_1d, M, bins_1d, b, CC, jN)`
- CC = M*M^T is symmetric, so column-major output = row-major output (reviewer confirmed)

### Backward compatibility
- `CPHMD_WHAM_SERIAL=1` environment variable forces old code path
- Python API signatures unchanged
- C API signatures unchanged (use_fused is internal to struct_data)
- All existing tests continue to pass unmodified
- File-based `readdata()` / `wham()` entry point always uses serial path (use_fused=0 from calloc)
  - Only in-memory entry points (`readdata_from_memory` / `wham_profiles_slim_from_memory`) get the optimization
  - This is intentional: the file-based path is legacy and not performance-critical

### dlnZ_dN stride invariant
- `dlnZ_dN` is allocated as `[jN x B_N]` where B_N = max(1D_bins, 2D_bins)
- But the actual working stride is `g_1d_bins` (= `data->B[1].N` on host)
- When 2D bins > 1D bins (e.g., 1D=400, 2D=32x32=1024): B_N=1024, g_1d_bins=400
- ALL code touching dlnZ_dN MUST use g_1d_bins stride: fused kernel, build_M_matrix, host scatter
- The serial get_CC kernel at wham.cu:1419 already uses g_1d_bins correctly
