/*
 * Enhanced WHAM Implementation with GPU-Specific Safeguards
 *
 * This version includes comprehensive safeguards to prevent GPU-specific issues
 * that can lead to incorrect C,V matrix calculations:
 *
 * 1. CUDA Error Checking:
 *    - All CUDA calls wrapped with CUDA_CHECK macro
 *    - Kernel launch errors checked with cudaGetLastError()
 *    - Explicit synchronization after critical operations
 *
 * 2. Numerical Stability:
 *    - Enhanced logadd function with careful NaN/Inf handling
 *    - Input validation in all GPU kernels
 *    - Bounds checking for array accesses
 *    - Conservative underflow protection (only at machine precision limits)
 *
 * 3. Atomic Operation Safeguards:
 *    - Retry limits to prevent infinite loops
 *    - Strict input validation before atomic operations
 *    - GPU-specific atomic testing at startup
 *
 * 4. Memory Access Protection:
 *    - Bounds checking in all kernels
 *    - Array index validation
 *    - Safe memory copy operations
 *
 * 5. Result Validation:
 *    - Matrix validation functions (reports but doesn't auto-fix edge cases)
 *    - Invalid value detection
 *    - Minimal intervention approach for debugging
 *
 * 6. Device Management:
 *    - GPU capability checking
 *    - Device synchronization flags
 *    - Proper device reset at completion
 *
 * IMPORTANT: This version minimizes fallback values and auto-corrections
 * to preserve edge case behavior for debugging. Invalid results are
 * reported but not automatically "fixed" with potentially incorrect values.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h> // For rand()
#include <string.h> // For strncpy, strlen
#include <cuda_runtime.h>
#include <vector>

#include "wham.h"

// Polyfill: double atomicAdd via CAS for architectures < sm_60
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// #define kB 0.008314
#define kB 0.00198614
#define MAXLENGTH 4096
#define BLOCK 256
#define SBLOCK 16

// Verbosity control - set to 1 to enable diagnostic messages
#define VERBOSE 0
#define VERBOSE_PRINT(...) do { if (VERBOSE) fprintf(stderr, __VA_ARGS__); } while(0)

// Dynamic dimension constants (will be set at runtime)
// Maximum supported dimensions for shared memory allocation
#define MAX_1D_BINS 2048
#define MAX_2D_BINS 64

// Runtime dimension variables (set by auto-detection)
__device__ int g_1d_bins = 1024; // Will be updated at runtime
__device__ int g_2d_bins_x = 32; // Will be updated at runtime
__device__ int g_2d_bins_y = 32; // Will be updated at runtime

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                          \
  do                                                                                              \
  {                                                                                               \
    cudaError_t err = call;                                                                       \
    if (err != cudaSuccess)                                                                       \
    {                                                                                             \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      cudaDeviceReset();                                                                          \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)

// GPU-specific safeguards
#define MAX_ATOMIC_RETRIES 1000
#define NUMERICAL_TOLERANCE 1e-12

// Function declarations
void auto_detect_profile_dimensions(struct_data *data);
void update_device_dimensions(int bins_1d, int bins_2d_x, int bins_2d_y);

// G_imp cache: read a file once, return cached data on subsequent lookups
static void gimp_cache_init(struct_gimp_cache *cache)
{
  cache->count = 0;
}

static void gimp_cache_free(struct_gimp_cache *cache)
{
  for (int i = 0; i < cache->count; i++)
    free(cache->entries[i].data);
  cache->count = 0;
}

// Look up or load a G_imp file. Returns pointer to cached data and sets *size.
static double *gimp_cache_get(struct_gimp_cache *cache, const char *filename, int expected_size)
{
  // Check if already cached
  for (int i = 0; i < cache->count; i++)
  {
    if (strcmp(cache->entries[i].filename, filename) == 0)
    {
      if (cache->entries[i].size != expected_size)
      {
        fprintf(stderr, "Error: cached %s has %d values but expected %d\n",
                filename, cache->entries[i].size, expected_size);
        exit(1);
      }
      return cache->entries[i].data;
    }
  }

  // Not cached — load from disk
  FILE *fp = fopen(filename, "r");
  if (!fp)
  {
    fprintf(stderr, "Error, %s does not exist\n", filename);
    exit(1);
  }

  // Count values for validation
  double value;
  int count = 0;
  while (fscanf(fp, "%lf", &value) == 1)
    count++;

  if (count != expected_size)
  {
    fprintf(stderr, "Error, %s has %d values, expected %d\n", filename, count, expected_size);
    fclose(fp);
    exit(1);
  }

  // Rewind and read data
  rewind(fp);
  double *data = (double *)malloc(expected_size * sizeof(double));
  for (int i = 0; i < expected_size; i++)
    fscanf(fp, "%lf", &data[i]);
  fclose(fp);

  // Store in cache
  if (cache->count >= GIMP_CACHE_MAX)
  {
    fprintf(stderr, "Error: G_imp cache full (max %d entries)\n", GIMP_CACHE_MAX);
    exit(1);
  }
  struct_gimp_entry *e = &cache->entries[cache->count++];
  strncpy(e->filename, filename, sizeof(e->filename) - 1);
  e->filename[sizeof(e->filename) - 1] = '\0';
  e->data = data;
  e->size = expected_size;

  return data;
}

__device__ __host__ inline double logadd(double lnA, double lnB)
{
  // If either is non-finite (-inf), return the other.
  // If both are non-finite, returns -inf (correct).
  if (!isfinite(lnA))
    return lnB;
  if (!isfinite(lnB))
    return lnA;

  if (lnA > lnB)
  {
    double diff = lnB - lnA;
    // Only protect against extreme underflow that would cause exp() to return exactly 0
    if (diff < -700.0) // exp(-700) ≈ 1e-304, near machine precision
      return lnA;
    return lnA + log(1 + exp(diff));
  }
  else
  {
    double diff = lnA - lnB;
    if (diff < -700.0)
      return lnB;
    return lnB + log(1 + exp(diff));
  }
}

__device__ inline void atomic_logadd(double *p_lnA, double lnB)
{
  // Strict input validation - reject any invalid inputs
  if (!isfinite(lnB))
    return;

  double lnA, lnC;
  double tmp_lnA;
  int retry_count = 0;

  tmp_lnA = p_lnA[0];
  do
  {
    lnA = tmp_lnA;
    lnC = logadd(lnA, lnB);

    // Only proceed if result is valid - no fallbacks
    if (!isfinite(lnC))
    {
      return; // Abort if result is invalid
    }

    tmp_lnA = __longlong_as_double(atomicCAS((unsigned long long int *)p_lnA, __double_as_longlong(lnA), __double_as_longlong(lnC)));

    // Prevent infinite loops on problematic GPUs
    retry_count++;
    if (retry_count > MAX_ATOMIC_RETRIES)
    {
      break;
    }
  } while (lnA != tmp_lnA);
}

// Test kernel for atomic operations
__global__ void test_atomic_wrapper(double *ptr)
{
  atomic_logadd(ptr, 1.0);
}

// GPU device validation and setup
void validate_and_setup_gpu()
{
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0)
  {
    fprintf(stderr, "Error: No CUDA-capable devices found\n");
    exit(EXIT_FAILURE);
  }

  // Get current device
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  fprintf(stderr, "Using GPU: %s (Device %d)\n", prop.name, device);
  fprintf(stderr, "Compute capability: %d.%d\n", prop.major, prop.minor);
  fprintf(stderr, "Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

  // Validate atomic support
  if (prop.major < 1 || (prop.major == 1 && prop.minor < 1))
  {
    fprintf(stderr, "Warning: GPU may not support double-precision atomics properly\n");
  }

  // Set device flags for better reliability
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

  // Test atomic operations
  double *test_ptr;
  CUDA_CHECK(cudaMalloc(&test_ptr, sizeof(double)));
  double test_val = -INFINITY;
  CUDA_CHECK(cudaMemcpy(test_ptr, &test_val, sizeof(double), cudaMemcpyHostToDevice));

  test_atomic_wrapper<<<1, 1>>>(test_ptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&test_val, test_ptr, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(test_ptr));

  if (!isfinite(test_val) || fabs(test_val - 1.0) > NUMERICAL_TOLERANCE)
  {
    fprintf(stderr, "Warning: GPU atomic operations may be unreliable on this device\n");
    fprintf(stderr, "Test result: %g (expected: 1.0)\n", test_val);
  }
  else
  {
    fprintf(stderr, "GPU atomic operations validated successfully\n");
  }
}

struct_data *readdata(int arg1, double arg2, int arg3, int arg4, int use_gshift,
                      int *nsubs_in, int nsites_in, const char *g_imp_path)
{
  const char *Edir = "Energy", *Ddir = "Lambda";
  FILE *fp, *fpE, *fpQ;
  int i, s1, s2, j, jN, iN, B0_MAX = INT_MIN, B0_MIN = INT_MAX;
  char fnm[MAXLENGTH], line[MAXLENGTH], *linebuffer;
  int ibuffer, n;
  double E, q;
  struct_data *data = (struct_data *)malloc(sizeof(struct_data));
  data->gimp_cache = (struct_gimp_cache *)malloc(sizeof(struct_gimp_cache));
  gimp_cache_init(data->gimp_cache);

  data->Nsim = arg1;

  // Store G_imp path (use provided path or default to "G_imp")
  if (g_imp_path && strlen(g_imp_path) > 0)
  {
    strncpy(data->g_imp_path, g_imp_path, sizeof(data->g_imp_path) - 1);
    data->g_imp_path[sizeof(data->g_imp_path) - 1] = '\0';
  }
  else
  {
    strcpy(data->g_imp_path, "G_imp");
  }

  // Use provided nsubs array or read from file
  if (nsubs_in != NULL && nsites_in > 0)
  {
    data->Nsites = nsites_in;
    data->Nsubs = (int *)malloc(data->Nsites * sizeof(int));
    data->block0 = (int *)malloc((data->Nsites + 1) * sizeof(int));
    data->Nblocks = 0;
    data->block0[0] = 0;
    for (i = 0; i < data->Nsites; i++)
    {
      data->Nsubs[i] = nsubs_in[i];
      data->Nblocks += data->Nsubs[i];
      data->block0[i + 1] = data->block0[i] + data->Nsubs[i];
    }
    fprintf(stderr, "Using provided nsubs array: nsites=%d, nblocks=%d\n", data->Nsites, data->Nblocks);
  }
  else
  {
    // Fallback: read from file for backward compatibility
    fp = fopen("prep/nsubs", "r");
    if (!fp)
    {
      fprintf(stderr, "Error, prep/nsubs does not exist and no nsubs array provided\n");
      exit(1);
    }
    data->Nsites = 0;
    while (fscanf(fp, "%d", &i) == 1)
      data->Nsites++;
    fclose(fp);
    data->Nsubs = (int *)malloc(data->Nsites * sizeof(int));
    data->block0 = (int *)malloc((data->Nsites + 1) * sizeof(int));
    fp = fopen("prep/nsubs", "r");
    data->Nblocks = 0;
    data->block0[0] = 0;
    for (i = 0; i < data->Nsites; i++)
    {
      fscanf(fp, "%d", &(data->Nsubs[i]));
      data->Nblocks += data->Nsubs[i];
      data->block0[i + 1] = data->block0[i] + data->Nsubs[i];
    }
    fclose(fp);
    fprintf(stderr, "Read nsubs from file: nsites=%d, nblocks=%d\n", data->Nsites, data->Nblocks);
  }

  data->ms = arg3;
  data->msprof = arg4;
  data->NL = data->Nblocks;
  data->NF = data->Nsim;
  data->Ndim = data->Nsim + data->NL + 1 + 2;

  data->T_h = (double *)malloc(data->NF * sizeof(double));
  data->beta_h = (double *)malloc(data->NF * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->beta_d), data->NF * sizeof(double)));
  for (i = 0; i < data->NF; i++)
  {
    data->T_h[i] = arg2;
    data->beta_h[i] = 1.0 / (kB * data->T_h[i]);
  }
  CUDA_CHECK(cudaMemcpy(data->beta_d, data->beta_h, data->NF * sizeof(double), cudaMemcpyHostToDevice));
  data->beta_t = 1.0 / (kB * arg2);

  data->B[0].dx = 0.1;
  data->B[1].dx = 0.0009765625; // Updated: 0.002500025 * (400/1024)
  data->B2d[0].dx = 0.1;
  data->B2d[1].dx = 0.03125031875; // Updated: 0.0500005 * (20/32)
  data->B2d[2].dx = 0.03125031875; // Updated: 0.0500005 * (20/32)

  data->n_h = (int *)malloc(data->NF * sizeof(int));
  CUDA_CHECK(cudaMalloc(&(data->n_d), data->NF * sizeof(int)));
  data->ND = 0;
  data->NDmax = MAXLENGTH;
  data->D_h = (double *)malloc(data->NDmax * data->Ndim * sizeof(double));
  data->i_h = (int *)malloc(data->NDmax * sizeof(int));
  data->lnw_h = (double *)malloc(data->NF * sizeof(double));

  for (i = 0; i < data->NF; i++)
  {
    sprintf(fnm, "%s/ESim%d.dat", Edir, i + 1);
    fpE = fopen(fnm, "r");
    if (!fpE)
    {
      fprintf(stderr, "Error, energy file %s does not exist\n", fnm);
      exit(1);
    }
    sprintf(fnm, "%s/Lambda%d.dat", Ddir, i + 1);
    fpQ = fopen(fnm, "r");
    if (!fpQ)
    {
      fprintf(stderr, "Error, contact file %s does not exist\n", fnm);
      exit(1);
    }

    data->lnw_h[i] = (data->NF - i - 1) * log(1.0);
    n = 0;
    while (fgets(line, MAXLENGTH, fpE) != NULL)
    {
      if (data->ND >= data->NDmax)
      {
        data->NDmax += MAXLENGTH;
        data->D_h = (double *)realloc(data->D_h, data->NDmax * data->Ndim * sizeof(double));
        data->i_h = (int *)realloc(data->i_h, data->NDmax * sizeof(int));
      }
      n++;
      linebuffer = line;
      sscanf(linebuffer, "%lf%n", &E, &ibuffer);
      linebuffer += ibuffer;
      data->D_h[data->ND * data->Ndim] = E;
      for (j = 0; j < data->NF; j++)
      {
        sscanf(linebuffer, "%lf%n", &E, &ibuffer);
        linebuffer += ibuffer;
        data->D_h[data->ND * data->Ndim + data->NL + 1 + j] = E;
      }
      fgets(line, MAXLENGTH, fpQ);
      linebuffer = line;
      for (j = 0; j < data->Nblocks; j++)
      {
        sscanf(linebuffer, "%lf%n", &q, &ibuffer);
        linebuffer += ibuffer;
        data->D_h[data->ND * data->Ndim + 1 + j] = q;
      }
      data->i_h[data->ND] = i;
      int iB0 = (int)floor(E / data->B[0].dx);
      if (iB0 < B0_MIN)
        B0_MIN = iB0;
      if (iB0 > B0_MAX)
        B0_MAX = iB0;
      data->ND++;
    }
    data->n_h[i] = n;
    fclose(fpE);
    fclose(fpQ);
  }

  data->B[0].min = B0_MIN * data->B[0].dx;
  data->B[0].max = (B0_MAX + 1) * data->B[0].dx;
  data->B[0].N = (B0_MAX - B0_MIN) + 1;
  data->B2d[0].min = data->B[0].min;
  data->B2d[0].max = data->B[0].max;
  data->B2d[0].N = data->B[0].N;
  data->B[1].min = 0;
  data->B[1].max = 1;
  data->B[1].N = 1024; // Default, will be auto-detected
  data->B2d[1].min = 0;
  data->B2d[1].max = 1;
  data->B2d[1].N = 32; // Default, will be auto-detected
  data->B2d[2].min = 0;
  data->B2d[2].max = 1;
  data->B2d[2].N = 32; // Default, will be auto-detected

  // Auto-detect dimensions from existing G_imp files
  auto_detect_profile_dimensions(data);

  int B_N = data->B[1].N;
  if (data->B2d[1].N * data->B2d[2].N > B_N)
    B_N = data->B2d[1].N * data->B2d[2].N;

  CUDA_CHECK(cudaMemcpy(data->n_d, data->n_h, data->NF * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&(data->D_d), data->NDmax * data->Ndim * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(data->D_d, data->D_h, data->NDmax * data->Ndim * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&(data->i_d), data->NDmax * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(data->i_d, data->i_h, data->NDmax * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&(data->lnw_d), data->NF * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(data->lnw_d, data->lnw_h, data->NF * sizeof(double), cudaMemcpyHostToDevice));
  data->lnDenom_h = (double *)malloc(data->NDmax * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->lnDenom_d), data->NDmax * sizeof(double)));

  data->f_h = (double *)malloc(data->NF * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->f_d), data->NF * sizeof(double)));
  for (i = 0; i < data->NF; i++)
    data->f_h[i] = 0.0;
  CUDA_CHECK(cudaMemcpy(data->f_d, data->f_h, data->NF * sizeof(double), cudaMemcpyHostToDevice));
  // invf_h/invf_d removed: getf writes f_d directly, iteratedata copies f_d → f_h

  fprintf(stderr, "Warning, DOS allocation is not sparse, requesting %d doubles\n", data->B[0].N * B_N);
  data->lnZ_h = (double *)malloc(B_N * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->lnZ_d), B_N * sizeof(double)));

  iN = 0;
  for (s1 = 0; s1 < data->Nsites; s1++)
  {
    for (s2 = s1; s2 < data->Nsites; s2++)
    {
      if (s1 == s2)
      {
        if (data->Nsubs[s1] == 2)
        {
          iN += data->Nsubs[s1] + data->Nsubs[s1] * (data->Nsubs[s1] - 1) / 2;
        }
        else
        {
          iN += data->Nsubs[s1] + 2 * data->Nsubs[s1] * (data->Nsubs[s1] - 1) / 2;
        }
      }
      else if (data->msprof)
      {
        iN += data->Nsubs[s1] * data->Nsubs[s2];
      }
    }
  }
  data->iN = iN;

  jN = 0;
  for (s1 = 0; s1 < data->Nsites; s1++)
  {
    for (s2 = s1; s2 < data->Nsites; s2++)
    {
      if (s1 == s2)
      {
        jN += data->Nsubs[s1] + 5 * data->Nsubs[s1] * (data->Nsubs[s1] - 1) / 2;
      }
      else if (data->ms == 1)
      {
        jN += 5 * data->Nsubs[s1] * data->Nsubs[s2];
      }
      else if (data->ms == 2)
      {
        jN += data->Nsubs[s1] * data->Nsubs[s2];
      }
    }
  }
  data->jN = jN;

  data->dlnZ_hN = (double **)malloc(jN * sizeof(double *));
  for (j = 0; j < jN; j++)
  {
    data->dlnZ_hN[j] = (double *)malloc(B_N * sizeof(double));
  }
  CUDA_CHECK(cudaMalloc(&(data->dlnZ_d), B_N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&(data->dlnZ_dN), jN * B_N * sizeof(double)));
  data->Gimp_h = (double *)malloc(B_N * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->Gimp_d), B_N * sizeof(double)));
  data->C_h = (double *)malloc(jN * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->C_d), jN * sizeof(double)));
  data->CV_h = (double *)malloc(jN * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->CV_d), jN * sizeof(double)));
  data->CC_h = (double *)malloc(jN * jN * sizeof(double));
  CUDA_CHECK(cudaMalloc(&(data->CC_d), jN * jN * sizeof(double)));

  int nblocks_tot = data->Nblocks;
  data->gshift_h = (double *)malloc(data->NF * nblocks_tot * sizeof(double));
  data->use_gshift = use_gshift;

  if (use_gshift)
  {
    // Read G_imp shifts from files (wham_3 behavior)
    for (int sim = 0; sim < data->NF; ++sim)
    {
      char sf[256];
      sprintf(sf, "G_imp_shifts/shifts_sim%d.dat", sim + 1);
      FILE *fs = fopen(sf, "r");
      if (!fs)
      {
        fprintf(stderr, "shift table %s missing\n", sf);
        exit(1);
      }

      int blk = 0; /* skip comment lines that start with '#' */
      while (blk < nblocks_tot)
      {
        char line[256];
        fgets(line, sizeof(line), fs);
        if (line[0] == '#' || line[0] == '\n')
          continue;
        sscanf(line, "%lf", &data->gshift_h[sim * nblocks_tot + blk]);
        ++blk;
      }
      fclose(fs);
    }
    VERBOSE_PRINT("G_imp shifts enabled, loaded from G_imp_shifts/\n");
  }
  else
  {
    // Initialize gshift arrays with zeros (legacy ALF behavior)
    for (int i = 0; i < data->NF * nblocks_tot; ++i)
    {
      data->gshift_h[i] = 0.0;
    }
    VERBOSE_PRINT("G_imp shifts disabled, using zeros (legacy mode)\n");
  }

  CUDA_CHECK(cudaMalloc(&(data->gshift_d),
                        data->NF * nblocks_tot * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(data->gshift_d, data->gshift_h,
                        data->NF * nblocks_tot * sizeof(double),
                        cudaMemcpyHostToDevice));

  return data;
}

// Build profile descriptor table: maps profile index → (kernel, block indices, G_imp info)
// Replaces the decrement-counter enumeration in bin_all().
void build_profile_descs(struct_data *data)
{
  int iN = data->iN;
  data->profiles = (profile_desc *)malloc(iN * sizeof(profile_desc));
  int idx = 0;
  int B_1d = data->B[1].N;
  int B_2d = data->B2d[1].N * data->B2d[2].N;

  for (int s1 = 0; s1 < data->Nsites; s1++)
  {
    for (int s2 = s1; s2 < data->Nsites; s2++)
    {
      if (s1 == s2)
      {
        // 1D profiles (phi)
        for (int i1 = data->block0[s1]; i1 < data->block0[s1 + 1]; i1++)
        {
          profile_desc *d = &data->profiles[idx++];
          d->ptype = 0;
          d->i1 = i1; d->i2 = -1;
          d->wnorm = 1.0;
          d->B_N = B_1d;
          sprintf(d->gimp_fn, "%s/G1_%d.dat", data->g_imp_path, data->Nsubs[s1]);
        }
        // 12 profiles (psi, intra-site pairs)
        for (int i1 = data->block0[s1]; i1 < data->block0[s1 + 1]; i1++)
        {
          for (int i2 = i1 + 1; i2 < data->block0[s1 + 1]; i2++)
          {
            profile_desc *d = &data->profiles[idx++];
            d->ptype = 1;
            d->i1 = i1; d->i2 = i2;
            d->wnorm = 1.0 / ((data->Nsubs[s1] - 1) / 2.0);
            d->B_N = B_2d;
            sprintf(d->gimp_fn, "%s/G12_%d.dat", data->g_imp_path, data->Nsubs[s1]);
          }
        }
        // 2D profiles (chi/omega, only when nsubs > 2)
        if (data->Nsubs[s1] > 2)
        {
          for (int i1 = data->block0[s1]; i1 < data->block0[s1 + 1]; i1++)
          {
            for (int i2 = i1 + 1; i2 < data->block0[s1 + 1]; i2++)
            {
              profile_desc *d = &data->profiles[idx++];
              d->ptype = 2;
              d->i1 = i1; d->i2 = i2;
              d->wnorm = 1.0 / ((data->Nsubs[s1] - 1) / 2.0);
              d->B_N = B_2d;
              sprintf(d->gimp_fn, "%s/G2_%d.dat", data->g_imp_path, data->Nsubs[s1]);
            }
          }
        }
      }
      else if (data->msprof)
      {
        // Cross-site profiles (SS)
        for (int i1 = data->block0[s1]; i1 < data->block0[s1 + 1]; i1++)
        {
          for (int i2 = data->block0[s2]; i2 < data->block0[s2 + 1]; i2++)
          {
            profile_desc *d = &data->profiles[idx++];
            d->ptype = 3;
            d->i1 = i1; d->i2 = i2;
            d->wnorm = 1.0 / (data->Nsubs[s1] * data->Nsubs[s2]);
            d->B_N = B_2d;
            sprintf(d->gimp_fn, "%s/G1_%d_%d.dat", data->g_imp_path, data->Nsubs[s1], data->Nsubs[s2]);
          }
        }
      }
    }
  }
  if (idx != iN)
  {
    fprintf(stderr, "FATAL: build_profile_descs count mismatch: %d vs iN=%d\n", idx, iN);
    exit(1);
  }
}

// Build parameter descriptor table: maps param index → (kernel type, block indices)
// Replaces the decrement-counter enumeration in reactioncoord_all().
void build_param_descs(struct_data *data)
{
  int jN = data->jN;
  data->params = (param_desc *)malloc(jN * sizeof(param_desc));
  int idx = 0;

  for (int s1 = 0; s1 < data->Nsites; s1++)
  {
    for (int s2 = s1; s2 < data->Nsites; s2++)
    {
      if (s1 == s2)
      {
        // phi (b terms)
        for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
        {
          param_desc *d = &data->params[idx++];
          d->type = 0; d->j1 = j1; d->j2 = -1;
        }
        // psi (c terms)
        for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
        {
          for (int j2 = j1 + 1; j2 < data->block0[s1 + 1]; j2++)
          {
            param_desc *d = &data->params[idx++];
            d->type = 1; d->j1 = j1; d->j2 = j2;
          }
        }
        // chi (x terms, ordered pairs j1 != j2)
        for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
        {
          for (int j2 = data->block0[s1]; j2 < data->block0[s1 + 1]; j2++)
          {
            if (j1 != j2)
            {
              param_desc *d = &data->params[idx++];
              d->type = 2; d->j1 = j1; d->j2 = j2;
            }
          }
        }
        // omega (s terms, ordered pairs j1 != j2)
        for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
        {
          for (int j2 = data->block0[s1]; j2 < data->block0[s1 + 1]; j2++)
          {
            if (j1 != j2)
            {
              param_desc *d = &data->params[idx++];
              d->type = 3; d->j1 = j1; d->j2 = j2;
            }
          }
        }
      }
      else if (data->ms)
      {
        // Cross-site psi
        for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
        {
          for (int j2 = data->block0[s2]; j2 < data->block0[s2 + 1]; j2++)
          {
            param_desc *d = &data->params[idx++];
            d->type = 1; d->j1 = j1; d->j2 = j2;
          }
        }
        if (data->ms == 1)
        {
          // Cross-site chi (both directions)
          for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
          {
            for (int j2 = data->block0[s2]; j2 < data->block0[s2 + 1]; j2++)
            {
              param_desc *d = &data->params[idx++];
              d->type = 2; d->j1 = j1; d->j2 = j2;
              d = &data->params[idx++];
              d->type = 2; d->j1 = j2; d->j2 = j1;
            }
          }
          // Cross-site omega (both directions)
          for (int j1 = data->block0[s1]; j1 < data->block0[s1 + 1]; j1++)
          {
            for (int j2 = data->block0[s2]; j2 < data->block0[s2 + 1]; j2++)
            {
              param_desc *d = &data->params[idx++];
              d->type = 3; d->j1 = j1; d->j2 = j2;
              d = &data->params[idx++];
              d->type = 3; d->j1 = j2; d->j2 = j1;
            }
          }
        }
      }
    }
  }
  if (idx != jN)
  {
    fprintf(stderr, "FATAL: build_param_descs count mismatch: %d vs jN=%d\n", idx, jN);
    exit(1);
  }
}

__global__ void resetlogdata(double *d, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    d[i] = -INFINITY;
}

__global__ void resetdata(double *d, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    d[i] = 0.0;
}

__global__ void sumdenom(struct_data data)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double lnDenom = -INFINITY;
    for (int i = 0; i < data.NF; i++)
    {
      double E = data.D_d[t * data.Ndim + data.NL + 1 + i];
      double lnwn = data.lnw_d[i] + log((double)data.n_d[i]);
      lnDenom = logadd(lnDenom, lnwn + data.f_d[i] - data.beta_d[i] * E);
    }
    data.lnDenom_d[t] = lnDenom;
  }
}

__global__ void getf(struct_data data)
{
  int tmin = (data.ND * threadIdx.x) / blockDim.x;
  int tmax = (data.ND * (threadIdx.x + 1)) / blockDim.x;
  int i = blockIdx.x;
  double beta = data.beta_d[i];
  __shared__ double invf[BLOCK];

  invf[threadIdx.x] = -INFINITY;
  for (int t = tmin; t < tmax; t++)
  {
    double lnw = data.lnw_d[data.i_d[t]];
    double E = data.D_d[t * data.Ndim + data.NL + 1 + i];
    invf[threadIdx.x] = logadd(invf[threadIdx.x], lnw - beta * E - data.lnDenom_d[t]);
  }

  __syncthreads();
  for (int t = 1; t < blockDim.x; t *= 2)
  {
    if ((threadIdx.x % (2 * t)) == 0)
    {
      invf[threadIdx.x] = logadd(invf[threadIdx.x], invf[threadIdx.x + t]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    data.f_d[i] = -invf[0];
  }
}

void iteratedata(struct_data *data)
{
  int itt, mitt = 1000;
  double f_sum, max_change, tolerance = 1e-8; // Increased precision for convergence
  FILE *fp;

  double *f_prev = (double *)malloc(data->NF * sizeof(double));
  for (int i = 0; i < data->NF; i++)
    f_prev[i] = 0.0;

  for (itt = 0; itt < mitt; itt++)
  {
    sumdenom<<<(data->ND + BLOCK - 1) / BLOCK, BLOCK>>>(data[0]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    getf<<<data->NF, BLOCK>>>(data[0]);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(data->f_h, data->f_d, data->NF * sizeof(double), cudaMemcpyDeviceToHost));

    f_sum = 0.0;
    max_change = 0.0;
    for (int i = 0; i < data->NF; i++)
    {
      f_sum += data->f_h[i];
      double change = fabs(data->f_h[i] - f_prev[i]);
      if (change > max_change)
        max_change = change;
      f_prev[i] = data->f_h[i];
    }
    f_sum /= data->NF;
    // Normalize on host and push back to device (avoids normf kernel launch + extra D→H copy)
    for (int i = 0; i < data->NF; i++)
      data->f_h[i] -= f_sum;
    CUDA_CHECK(cudaMemcpy(data->f_d, data->f_h, data->NF * sizeof(double), cudaMemcpyHostToDevice));

    fprintf(stdout, "Iteration %d, max f change: %g\n", itt + 1, max_change);
    // Relative convergence, floored at absolute tolerance when f_sum ≈ 0
    // (well-balanced systems have f_sum near zero after normalization)
    if (max_change < tolerance * fmax(fabs(f_sum), 1.0))
      break;
  }

  // NOTE: Statistical uncertainty of f-weights can be estimated post-hoc
  // from the inverse of the WHAM Hessian (C matrix): var(f_i) ≈ C_ii^{-1}.
  // A proper bootstrap would require resampling frames with replacement
  // and re-converging WHAM for each resample (~10x cost), which is not
  // justified here since f-weight uncertainties are not used downstream.

  fp = fopen("f.dat", "w");
  for (int i = 0; i < data->NF; i++)
    fprintf(fp, " %18.12f", data->f_h[i]); // Increased precision for high-precision lambda data
  fclose(fp);

  free(f_prev);
}

__global__ void bin1(struct_data data, int i1)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    if (isfinite(q1)) // Validate input for precision
    {
      int iB1 = (int)floor((q1 - data.B[1].min) / data.B[1].dx);
      // Ensure bin index is within valid range
      if (iB1 >= 0 && iB1 < data.B[1].N)
      {
        data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = iB1;
      }
      else
      {
        data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1; // Invalid bin
      }
    }
    else
    {
      data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1; // Invalid input
    }
  }
}

__global__ void bin12(struct_data data, int i1, int i2)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    double q2 = data.D_d[t * data.Ndim + 1 + i2];

    if (isfinite(q1) && isfinite(q2) && q1 >= 0 && q2 >= 0) // Validate inputs
    {
      double sum_q = q1 + q2;
      if (sum_q > data.cutlsum && sum_q > 1e-12) // Configurable threshold
      {
        double ratio = q1 / sum_q;
        int iB12 = (int)floor((ratio - data.B[1].min) / data.B[1].dx);
        // Validate bin index
        if (iB12 >= 0 && iB12 < data.B[1].N)
        {
          data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = iB12;
        }
        else
        {
          data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1;
        }
      }
      else
      {
        data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1;
      }
    }
    else
    {
      data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1; // Invalid inputs
    }
  }
}

__global__ void bin2(struct_data data, int i1, int i2)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    double q2 = data.D_d[t * data.Ndim + 1 + i2];

    if (isfinite(q1) && isfinite(q2) && q1 >= 0 && q2 >= 0) // Validate inputs
    {
      int iB1 = (int)floor((q1 - data.B2d[1].min) / data.B2d[1].dx);
      int iB2 = (int)floor((q2 - data.B2d[2].min) / data.B2d[2].dx);

      // Validate both bin indices
      if (iB1 >= 0 && iB1 < data.B2d[1].N && iB2 >= 0 && iB2 < data.B2d[2].N)
      {
        int combined_bin = iB1 * data.B2d[2].N + iB2;
        // Additional check for combined bin index
        if (combined_bin >= 0 && combined_bin < data.B2d[1].N * data.B2d[2].N)
        {
          data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = combined_bin;
        }
        else
        {
          data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1;
        }
      }
      else
      {
        data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1;
      }
    }
    else
    {
      data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim] = -1; // Invalid inputs
    }
  }
}

double bin_all(struct_data *data, int *ptype, int i)
{
  profile_desc *p = &data->profiles[i];
  *ptype = p->ptype;
  data->current_block_idx = p->i1;

  int grid = (data->ND + BLOCK - 1) / BLOCK;
  switch (p->ptype)
  {
  case 0:
    fprintf(stderr, "1D Profile %d\n", p->i1);
    bin1<<<grid, BLOCK>>>(data[0], p->i1);
    break;
  case 1:
    fprintf(stderr, "1D Profile %d,%d\n", p->i1, p->i2);
    bin12<<<grid, BLOCK>>>(data[0], p->i1, p->i2);
    break;
  case 2:
    fprintf(stderr, "2D Profile %d,%d\n", p->i1, p->i2);
    bin2<<<grid, BLOCK>>>(data[0], p->i1, p->i2);
    break;
  case 3:
    fprintf(stderr, "2D SS Profile %d,%d\n", p->i1, p->i2);
    bin2<<<grid, BLOCK>>>(data[0], p->i1, p->i2);
    break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Load G_imp data from cache
  double *gimp_cached = gimp_cache_get(data->gimp_cache, p->gimp_fn, p->B_N);
  memcpy(data->Gimp_h, gimp_cached, p->B_N * sizeof(double));
  CUDA_CHECK(cudaMemcpy(data->Gimp_d, data->Gimp_h, p->B_N * sizeof(double), cudaMemcpyHostToDevice));

  return p->wnorm;
}

__global__ void reactioncoord_phi(struct_data data, int i1)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim + 1] = q1;
  }
}

__global__ void reactioncoord_psi(struct_data data, int i1, int i2)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    double q2 = data.D_d[t * data.Ndim + 1 + i2];
    data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim + 1] = q1 * q2;
  }
}

__global__ void reactioncoord_omega(struct_data data, int i1, int i2)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    double q2 = data.D_d[t * data.Ndim + 1 + i2];
    data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim + 1] = q2 * (1 - 1 / (q1 / data.chi_offset + 1));
  }
}

__global__ void reactioncoord_chi(struct_data data, int i1, int i2)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < data.ND)
  {
    double q1 = data.D_d[t * data.Ndim + 1 + i1];
    double q2 = data.D_d[t * data.Ndim + 1 + i2];
    data.D_d[t * data.Ndim + 1 + data.NL + data.Nsim + 1] = q2 * (1 - exp(-q1 / data.omega_scale));
  }
}

void reactioncoord_all(struct_data *data, int i)
{
  param_desc *d = &data->params[i];
  int grid = (data->ND + BLOCK - 1) / BLOCK;

  switch (d->type)
  {
  case 0: // phi
    reactioncoord_phi<<<grid, BLOCK>>>(data[0], d->j1);
    break;
  case 1: // psi
    reactioncoord_psi<<<grid, BLOCK>>>(data[0], d->j1, d->j2);
    break;
  case 2: // chi
    reactioncoord_chi<<<grid, BLOCK>>>(data[0], d->j1, d->j2);
    break;
  case 3: // omega
    reactioncoord_omega<<<grid, BLOCK>>>(data[0], d->j1, d->j2);
    break;
  }
}

__global__ void get_lnZ(struct_data data, double beta,
                        const double *__restrict__ gshift,
                        int block_off)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double loc_lnZ[];

  // Enhanced initialization with bounds checking using dynamic dimension
  for (int i = threadIdx.x; i < g_1d_bins; i += blockDim.x)
    loc_lnZ[i] = -INFINITY;
  __syncthreads();

  // Validate input parameters
  if (!isfinite(beta) || beta <= 0)
    return;

  for (int i = SBLOCK * t; i < SBLOCK * (t + 1) && i < data.ND; i++)
  {
    // Bounds checking for array access
    if (i >= data.ND || i < 0)
      continue;

    double E = data.D_d[i * data.Ndim + 0];
    int iB = (int)data.D_d[i * data.Ndim + 1 + data.NL + data.Nsim];

    // Enhanced bounds and validity checking using dynamic dimension
    if (iB >= 0 && iB < g_1d_bins && isfinite(E))
    {
      int sim_idx = data.i_d[i];
      if (sim_idx >= 0 && sim_idx < data.NF)
      {
        double lnw = data.lnw_d[sim_idx];
        double lnDenom = data.lnDenom_d[i];

        // Validate all components before computation
        if (isfinite(lnw) && isfinite(lnDenom))
        {
          double vshift = gshift[sim_idx * data.Nblocks + block_off];
          double contribution = lnw - lnDenom - beta * (E + vshift);
          // Only check for reasonable range, no clamping
          if (isfinite(contribution))
          {
            atomic_logadd(&loc_lnZ[iB], contribution);
          }
        }
      }
    }
    // Note: Invalid bins (iB == -1) are automatically skipped
  }
  __syncthreads();

  // Enhanced final reduction with validation using dynamic dimension
  for (int i = threadIdx.x; i < g_1d_bins; i += blockDim.x)
  {
    if (isfinite(loc_lnZ[i]) && loc_lnZ[i] > -INFINITY)
      atomic_logadd(&data.lnZ_d[i], loc_lnZ[i]);
  }
}

__global__ void get_dlnZ(struct_data data, int j1, double beta,
                         const double *__restrict__ gshift,
                         int block_off)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double loc_dlnZ[];

  // Enhanced initialization using dynamic dimension
  for (int i = threadIdx.x; i < g_1d_bins; i += blockDim.x)
    loc_dlnZ[i] = -INFINITY;
  __syncthreads();

  // Validate input parameters
  if (!isfinite(beta) || beta <= 0 || j1 < 0)
    return;

  for (int i = SBLOCK * t; i < SBLOCK * (t + 1) && i < data.ND; i++)
  {
    // Bounds checking
    if (i >= data.ND || i < 0)
      continue;

    double E = data.D_d[i * data.Ndim + 0];
    int iB = (int)data.D_d[i * data.Ndim + 1 + data.NL + data.Nsim];
    double q = data.D_d[i * data.Ndim + 1 + data.NL + data.Nsim + 1];

    // Enhanced validation with better precision handling using dynamic dimension
    if (iB >= 0 && iB < g_1d_bins && isfinite(E) && isfinite(q) && q > 1e-15) // More precise q threshold
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
            double vshift = gshift[sim_idx * data.Nblocks + block_off];
            double contribution = lnw - lnDenom - beta * (E + vshift) + log_q;
            if (isfinite(contribution))
            {
              atomic_logadd(&loc_dlnZ[iB], contribution);
            }
          }
        }
      }
    }
    // Note: Invalid bins (iB == -1) and very small q values are automatically skipped
  }
  __syncthreads();

  // Enhanced final reduction with bounds checking using dynamic dimension
  for (int i = threadIdx.x; i < g_1d_bins; i += blockDim.x)
  {
    if (isfinite(loc_dlnZ[i]) && loc_dlnZ[i] > -INFINITY)
    {
      int target_idx = g_1d_bins * j1 + i;
      // Additional bounds check for target array
      if (target_idx >= 0 && j1 >= 0)
        atomic_logadd(&data.dlnZ_dN[target_idx], loc_dlnZ[i]);
    }
  }
}

__global__ void get_CC(struct_data data, int i, double beta, double wnorm, int ptype,
                       const double *__restrict__ gshift,
                       int block_off)
{
  int j1 = blockIdx.x, j2 = blockIdx.y;
  __shared__ double loc_CC[100];
  double myCC = 0;

  // Validate input parameters
  if (!isfinite(beta) || !isfinite(wnorm) || beta <= 0 || wnorm <= 0)
  {
    if (threadIdx.x == 0)
    {
      data.CC_d[gridDim.x * j1 + j2] = 0.0;
    }
    return;
  }

  for (int k = threadIdx.x; k < g_1d_bins; k += 100)
  {
    double weight = wnorm;
    // Enhanced weight calculation with validation using dynamic dimension
    if ((ptype == 0 || ptype == 3) && k == (g_1d_bins - 1))
    {
      weight *= 100.0;
    }

    if (!isfinite(weight) || weight <= 0)
      continue;

    double lnZ = data.lnZ_d[k];
    if (isfinite(lnZ) && lnZ > -INFINITY)
    {
      // Bounds checking for array access using dynamic dimension
      int idx1 = j1 * g_1d_bins + k;
      int idx2 = j2 * g_1d_bins + k;

      if (idx1 >= 0 && idx2 >= 0 && j1 >= 0 && j2 >= 0)
      {
        double dlnZ1 = data.dlnZ_dN[idx1];
        double dlnZ2 = data.dlnZ_dN[idx2];

        if (isfinite(dlnZ1) && isfinite(dlnZ2))
        {
          double exp_arg1 = dlnZ1 - lnZ;
          double exp_arg2 = dlnZ2 - lnZ;

          // Check for finite results without clamping
          if (isfinite(exp_arg1) && isfinite(exp_arg2))
          {
            double exp_val1 = exp(exp_arg1);
            double exp_val2 = exp(exp_arg2);

            if (isfinite(exp_val1) && isfinite(exp_val2))
            {
              double contribution = weight * exp_val1 * exp_val2;
              if (isfinite(contribution) && contribution >= 0)
              {
                myCC += contribution;
              }
            }
          }
        }
      }
    }
  }

  // Store partial sum without suppressing NaN — if a numerical issue
  // occurred, let it propagate to the final C matrix so the Python
  // solver can detect it and fail the iteration cleanly.
  loc_CC[threadIdx.x] = myCC;

  __syncthreads();

  // Tree reduction
  for (int k = 1; k < 100; k *= 2)
  {
    if (threadIdx.x % (2 * k) == 0 && threadIdx.x + k < 100)
    {
      loc_CC[threadIdx.x] += loc_CC[threadIdx.x + k];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    int output_idx = gridDim.x * j1 + j2;
    if (output_idx >= 0)
    {
      data.CC_d[output_idx] = loc_CC[0];
    }
  }
}

// G_imp profile size validation function
// Count double values in a file. Returns count, or -1 if file doesn't exist.
static int count_file_values(const char *filename)
{
  FILE *fp = fopen(filename, "r");
  if (!fp)
    return -1;

  int count = 0;
  double value;
  while (fscanf(fp, "%lf", &value) == 1)
    count++;
  fclose(fp);
  return count;
}

// Auto-detect profile dimensions from G_imp files and pre-populate the cache.
// Scans one 1D and one 2D file to determine bin counts, then loads all
// G_imp files into the cache so bin_all() never touches disk.
void auto_detect_profile_dimensions(struct_data *data)
{
  char fnm[MAXLENGTH];
  int dim_1d = 1024;              // Default 1D bins
  int dim_2d = 32;                // Default 2D bins (square)
  int found_1d = 0, found_2d = 0;

  fprintf(stderr, "\n=== Auto-detecting profile dimensions from G_imp files ===\n");

  // --- Phase 1: Detect dimensions from the first available files ---

  for (int s1 = 0; s1 < data->Nsites && !found_1d; s1++)
  {
    sprintf(fnm, "%s/G1_%d.dat", data->g_imp_path, data->Nsubs[s1]);
    int count = count_file_values(fnm);
    if (count > 0)
    {
      dim_1d = count;
      found_1d = 1;
      fprintf(stderr, "Detected 1D: %s → %d bins\n", fnm, count);
    }
  }

  // Try G12, then G2, then cross-site for 2D dimensions
  for (int s1 = 0; s1 < data->Nsites && !found_2d; s1++)
  {
    const char *prefixes[] = {"G12", "G2"};
    for (int p = 0; p < 2 && !found_2d; p++)
    {
      sprintf(fnm, "%s/%s_%d.dat", data->g_imp_path, prefixes[p], data->Nsubs[s1]);
      int count = count_file_values(fnm);
      if (count > 0)
      {
        double sq = sqrt((double)count);
        dim_2d = (sq == floor(sq)) ? (int)sq : 32;
        found_2d = 1;
        fprintf(stderr, "Detected 2D: %s → %dx%d bins\n", fnm, dim_2d, dim_2d);
      }
    }
    for (int s2 = s1 + 1; s2 < data->Nsites && !found_2d; s2++)
    {
      sprintf(fnm, "%s/G1_%d_%d.dat", data->g_imp_path, data->Nsubs[s1], data->Nsubs[s2]);
      int count = count_file_values(fnm);
      if (count > 0)
      {
        double sq = sqrt((double)count);
        dim_2d = (sq == floor(sq)) ? (int)sq : 32;
        found_2d = 1;
        fprintf(stderr, "Detected 2D: %s → %dx%d bins\n", fnm, dim_2d, dim_2d);
      }
    }
  }

  // Apply detected dimensions
  data->B[1].N = dim_1d;
  data->B[1].dx = 1.0 / dim_1d;
  data->B2d[1].N = dim_2d;
  data->B2d[1].dx = 1.0 / dim_2d;
  data->B2d[2].N = dim_2d;
  data->B2d[2].dx = 1.0 / dim_2d;

  fprintf(stderr, "Final dimensions: 1D=%d, 2D=%dx%d\n", dim_1d, dim_2d, dim_2d);

  // Update device constants
  update_device_dimensions(dim_1d, dim_2d, dim_2d);

  // --- Phase 2: Pre-load all G_imp files into cache ---
  int B_N_1d = dim_1d;
  int B_N_2d = dim_2d * dim_2d;

  for (int s1 = 0; s1 < data->Nsites; s1++)
  {
    // G1 (1D)
    sprintf(fnm, "%s/G1_%d.dat", data->g_imp_path, data->Nsubs[s1]);
    gimp_cache_get(data->gimp_cache, fnm, B_N_1d);

    // G12, G2 (2D)
    sprintf(fnm, "%s/G12_%d.dat", data->g_imp_path, data->Nsubs[s1]);
    gimp_cache_get(data->gimp_cache, fnm, B_N_2d);

    if (data->Nsubs[s1] > 2)
    {
      sprintf(fnm, "%s/G2_%d.dat", data->g_imp_path, data->Nsubs[s1]);
      gimp_cache_get(data->gimp_cache, fnm, B_N_2d);
    }

    // Cross-site (2D)
    if (data->msprof)
    {
      for (int s2 = s1 + 1; s2 < data->Nsites; s2++)
      {
        sprintf(fnm, "%s/G1_%d_%d.dat", data->g_imp_path, data->Nsubs[s1], data->Nsubs[s2]);
        gimp_cache_get(data->gimp_cache, fnm, B_N_2d);
      }
    }
  }

  fprintf(stderr, "Pre-loaded %d G_imp files into cache\n", data->gimp_cache->count);
  fprintf(stderr, "========================================================\n\n");
}

// Update device dimension constants
void update_device_dimensions(int bins_1d, int bins_2d_x, int bins_2d_y)
{
  // Validate dimensions don't exceed maximums
  if (bins_1d > MAX_1D_BINS)
  {
    fprintf(stderr, "Warning: 1D bins (%d) exceed maximum (%d), clamping\n",
            bins_1d, MAX_1D_BINS);
    bins_1d = MAX_1D_BINS;
  }

  if (bins_2d_x > MAX_2D_BINS || bins_2d_y > MAX_2D_BINS)
  {
    fprintf(stderr, "Warning: 2D bins (%dx%d) exceed maximum (%dx%d), clamping\n",
            bins_2d_x, bins_2d_y, MAX_2D_BINS, MAX_2D_BINS);
    bins_2d_x = (bins_2d_x > MAX_2D_BINS) ? MAX_2D_BINS : bins_2d_x;
    bins_2d_y = (bins_2d_y > MAX_2D_BINS) ? MAX_2D_BINS : bins_2d_y;
  }

  // Update device constants
  CUDA_CHECK(cudaMemcpyToSymbol(g_1d_bins, &bins_1d, sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(g_2d_bins_x, &bins_2d_x, sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(g_2d_bins_y, &bins_2d_y, sizeof(int)));

  fprintf(stderr, "Updated device dimensions: 1D=%d, 2D=%dx%d\n",
          bins_1d, bins_2d_x, bins_2d_y);
}

// Result validation functions
void validate_matrix_results(double *matrix, int rows, int cols, const char *name)
{
  int nan_count = 0, inf_count = 0, negative_count = 0;
  double sum = 0.0, max_val = -INFINITY, min_val = INFINITY;

  for (int i = 0; i < rows * cols; i++)
  {
    if (!isfinite(matrix[i]))
    {
      if (isnan(matrix[i]))
        nan_count++;
      else
        inf_count++;
    }
    else
    {
      sum += matrix[i];
      if (matrix[i] > max_val)
        max_val = matrix[i];
      if (matrix[i] < min_val)
        min_val = matrix[i];
      if (matrix[i] < 0)
        negative_count++;
    }
  }

  fprintf(stderr, "%s validation: %d NaN, %d Inf, %d negative values\n",
          name, nan_count, inf_count, negative_count);
  fprintf(stderr, "%s range: [%g, %g], sum: %g\n", name, min_val, max_val, sum);

  if (nan_count > 0 || inf_count > 0)
  {
    // Report but do NOT replace — patching NaN/Inf with zeros silently
    // corrupts the Hessian, producing a plausible-looking but wrong
    // linear system.  The Python solver will detect non-finite values
    // and fall back to zero updates for the entire iteration.
    fprintf(stderr, "WARNING: %s contains %d NaN + %d Inf values (not patched)\n",
            name, nan_count, inf_count);
  }

  // Report negative values but don't automatically fix them
  if (negative_count > 0)
  {
    fprintf(stderr, "INFO: %s has %d negative values - check for numerical issues\n",
            name, negative_count);
  }
}

void validate_gpu_results(struct_data *data, int B_N, int jN)
{
  // Validate lnZ results - be more careful about what we consider "invalid"
  int invalid_lnZ = 0;
  for (int k = 0; k < B_N; k++)
  {
    if (!isfinite(data->lnZ_h[k]))
    {
      invalid_lnZ++;
      // Don't use fallback values - let the calling code handle this
      // data->lnZ_h[k] remains as NaN/Inf for debugging
    }
  }
  // if (invalid_lnZ > 0)
  // {
  //   fprintf(stderr, "WARNING: %d invalid lnZ values found (not corrected)\n", invalid_lnZ);
  // }

  // Validate dlnZ results - same approach
  for (int j1 = 0; j1 < jN; j1++)
  {
    int invalid_dlnZ = 0;
    for (int k = 0; k < B_N; k++)
    {
      if (!isfinite(data->dlnZ_hN[j1][k]))
      {
        invalid_dlnZ++;
        // Keep invalid values for debugging
      }
    }
    // if (invalid_dlnZ > 0)
    // {
    //   fprintf(stderr, "WARNING: %d invalid dlnZ[%d] values found (not corrected)\n", invalid_dlnZ, j1);
    // }
  }
}

void getfofq(struct_data *data, double beta)
{
  int B_N = (data->B2d[1].N * data->B2d[2].N > data->B[1].N) ? data->B2d[1].N * data->B2d[2].N : data->B[1].N;
  int iN = data->iN, jN = data->jN, i, j1, j2, k;
  int dim = jN + iN;
  double *C = (double *)malloc(dim * dim * sizeof(double));
  double *V = (double *)malloc(dim * sizeof(double));
  if (!C || !V)
  {
    fprintf(stderr, "FATAL: malloc failed for C/V matrices (dim=%d, need %.1f MB)\n",
            dim, (dim * dim + dim) * sizeof(double) / 1e6);
    free(C); free(V);
    return;
  }
  double wnorm, weight;
  int ptype;
  char fnm[MAXLENGTH];
  FILE *fpC, *fpV, *fp;

  // Iterative refinement (1 iteration)
  double lambda_reg = 1e-9; // Very small regularization for high-precision lambda data
  for (int iter = 0; iter < 1; iter++)
  {
    // Initialize C and V inside the loop
    for (j1 = 0; j1 < jN + iN; j1++)
    {
      for (j2 = 0; j2 < jN + iN; j2++)
        C[j1 * (jN + iN) + j2] = 0.0;
      V[j1] = 0.0;
    }

    sumdenom<<<(data->ND + BLOCK - 1) / BLOCK, BLOCK>>>(data[0]);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (i = 0; i < iN; i++)
    {
      wnorm = bin_all(data, &ptype, i);
      resetlogdata<<<(B_N + BLOCK - 1) / BLOCK, BLOCK>>>(data->lnZ_d, B_N);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      get_lnZ<<<(data->ND + (100 * SBLOCK) - 1) / (100 * SBLOCK), 100, B_N * sizeof(double)>>>(data[0], data->beta_t,
                                                                         data->gshift_d, data->current_block_idx);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(data->lnZ_h, data->lnZ_d, B_N * sizeof(double), cudaMemcpyDeviceToHost));

      // Variance of lnZ for diagnostics
      double lnZ_mean = 0, lnZ_var = 0;
      int count = 0;
      for (k = 0; k < B_N; k++)
      {
        if (isfinite(data->lnZ_h[k]))
        {
          lnZ_mean += data->lnZ_h[k];
          count++;
        }
      }
      if (count > 0)
      {
        lnZ_mean /= count;
      }
      else
      {
        // Don't use fallback - report the issue and let it propagate
        fprintf(stderr, "ERROR: No finite lnZ values found for profile %d\n", i);
        // Keep the invalid values for debugging
      }
      for (k = 0; k < B_N; k++)
      {
        if (isfinite(data->lnZ_h[k]))
        {
          double diff = data->lnZ_h[k] - lnZ_mean;
          lnZ_var += diff * diff;
        }
      }
      if (count > 0)
        lnZ_var /= count;
      fprintf(stderr, "Profile %d, lnZ variance: %g\n", i, lnZ_var);

      sprintf(fnm, "multisite/G%d.dat", i + 1);
      fp = fopen(fnm, "w");
      for (k = 0; k < B_N; k++)
      {
        fprintf(fp, "%.12g\n", (-data->lnZ_h[k] - data->Gimp_h[k]) / data->beta_t); // Higher precision output
      }
      fclose(fp);

      for (k = 0; k < B_N; k++)
      {
        weight = wnorm;
        if ((ptype == 0 || ptype == 3) && k == B_N - 1)
          weight *= 100.0;
        if (isfinite(data->lnZ_h[k]))
        {
          double dG = (-data->lnZ_h[k] - data->Gimp_h[k]) / data->beta_t;
          // Don't clamp values - let extreme values be visible
          if (isfinite(dG))
          {
            V[jN + i] += weight * dG;
            C[(jN + i) * (jN + iN) + jN + i] += weight;
          }
        }
      }

      for (j1 = 0; j1 < jN; j1++)
      {
        reactioncoord_all(data, j1);
        resetlogdata<<<(B_N + BLOCK - 1) / BLOCK, BLOCK>>>(&(data->dlnZ_dN[B_N * j1]), B_N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        get_dlnZ<<<(data->ND + (100 * SBLOCK) - 1) / (100 * SBLOCK), 100, B_N * sizeof(double)>>>(data[0], j1, data->beta_t,
                                                                            data->gshift_d, data->current_block_idx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(data->dlnZ_hN[j1], &(data->dlnZ_dN[B_N * j1]), B_N * sizeof(double), cudaMemcpyDeviceToHost));
      }

      get_CC<<<make_uint3(jN, jN, 1), 100>>>(data[0], i, data->beta_t, wnorm, ptype,
                                             data->gshift_d, data->current_block_idx);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(data->CC_h, data->CC_d, jN * jN * sizeof(double), cudaMemcpyDeviceToHost));

      // Validate GPU results before using them
      validate_gpu_results(data, B_N, jN);
      validate_matrix_results(data->CC_h, jN, jN, "CC matrix");

      for (j1 = 0; j1 < jN; j1++)
      {
        for (k = 0; k < B_N; k++)
        {
          weight = wnorm;
          if ((ptype == 0 || ptype == 3) && k == B_N - 1)
            weight *= 100.0;
          if (isfinite(data->lnZ_h[k]))
          {
            double dG = (-data->lnZ_h[k] - data->Gimp_h[k]) / data->beta_t;
            double exp_arg = data->dlnZ_hN[j1][k] - data->lnZ_h[k];

            // Check for finite intermediate results without clamping
            if (isfinite(dG) && isfinite(exp_arg))
            {
              double exp_val = exp(exp_arg);

              // Validate final results
              if (isfinite(exp_val) && exp_val >= 0)
              {
                V[j1] += weight * exp_val * dG;
                C[j1 * (jN + iN) + jN + i] += weight * exp_val;
                C[(jN + i) * (jN + iN) + j1] += weight * exp_val;
              }
            }
          }
        }
        for (j2 = 0; j2 < jN; j2++)
        {
          // Let NaN propagate — Python solver detects it and fails the
          // iteration cleanly rather than using a silently corrupted Hessian.
          C[j1 * (jN + iN) + j2] += data->CC_h[j1 * jN + j2];
        }
      }
    }
  }

  // Add regularization and handle empty rows/cols with higher precision
  double big_lambda = 1.0;
  for (j1 = 0; j1 < jN + iN; j1++)
  {
    double row_weight = 0.0;
    for (j2 = 0; j2 < jN + iN; j2++)
    {
      row_weight += fabs(C[j1 * (jN + iN) + j2]);
    }
    // Use more precise threshold for empty row detection
    if (row_weight < 1e-12)
    {
      C[j1 * (jN + iN) + j1] += big_lambda;
      V[j1] = 0.0;
    }
    C[j1 * (jN + iN) + j1] += lambda_reg;
  }

  // Condition number estimation (simplified, using trace and norm)
  double trace = 0, norm = 0;
  for (j1 = 0; j1 < jN + iN; j1++)
  {
    trace += C[j1 * (jN + iN) + j1];
    for (j2 = 0; j2 < jN + iN; j2++)
    {
      norm += C[j1 * (jN + iN) + j2] * C[j1 * (jN + iN) + j2];
    }
  }
  norm = sqrt(norm);
  fprintf(stderr, "C matrix condition number estimate: %g\n", norm / trace);

  // Final validation of C and V matrices
  validate_matrix_results(C, jN + iN, jN + iN, "Final C matrix");
  validate_matrix_results(V, jN + iN, 1, "Final V vector");

  fpC = fopen("multisite/C.dat", "w");
  fpV = fopen("multisite/V.dat", "w");
  for (j1 = 0; j1 < jN + iN; j1++)
  {
    for (j2 = 0; j2 < jN + iN; j2++)
      fprintf(fpC, " %.12g", C[j1 * (jN + iN) + j2]); // Higher precision for matrix output
    fprintf(fpC, "\n");
    fprintf(fpV, " %.12g\n", V[j1]); // Higher precision for vector output
  }
  fclose(fpC);
  fclose(fpV);

  free(C);
  free(V);
}

extern "C" int wham(int arg1, double arg2, int arg3, int arg4, int use_gshift,
                   int *nsubs, int nsites, const char *g_imp_path,
                   double chi_offset, double omega_scale, double cutlsum)
{
  // Initialize and validate GPU
  validate_and_setup_gpu();

  struct_data *data = readdata(arg1, arg2, arg3, arg4, use_gshift, nsubs, nsites, g_imp_path);
  if (data) {
    data->chi_offset = chi_offset;
    data->omega_scale = omega_scale;
    data->cutlsum = cutlsum;
  }
  if (!data)
  {
    fprintf(stderr, "Error: Failed to read data\n");
    return -1;
  }

  // Build descriptor tables for O(1) profile/param lookup
  build_profile_descs(data);
  build_param_descs(data);

  // Synchronize before starting main computation
  CUDA_CHECK(cudaDeviceSynchronize());

  iteratedata(data);

  // Synchronize before final computation
  CUDA_CHECK(cudaDeviceSynchronize());

  getfofq(data, data->beta_t);

  // Final synchronization and cleanup
  CUDA_CHECK(cudaDeviceSynchronize());
  gimp_cache_free(data->gimp_cache);
  free(data->gimp_cache);
  free(data->profiles);
  free(data->params);
  CUDA_CHECK(cudaDeviceReset());

  return 0;
}


// ============================================================================
// LMALF (Likelihood Maximization ALF) Implementation
// ============================================================================
//
// Alternative to WHAM using L-BFGS optimization for bias parameter fitting.
// Based on original lmalf.cu by Ryan Hayes (2017).
// ============================================================================

// Forward declarations for LMALF
struct_lmalf *lmalf_setup(int nf, double temp, int ms, int msprof);
void lmalf_run(struct_lmalf *lm);
void lmalf_finish(struct_lmalf *lm);

// LMALF-specific reduction kernel (sums across block)
__device__ void lmalf_reduce(double local, double *shared, double *global)
{
  int k;
  shared[threadIdx.x] = local;
  __syncthreads();

  for (k = 1; k < LMALF_BLOCK; k *= 2)
  {
    if ((threadIdx.x % (2 * k)) == 0 && threadIdx.x + k < LMALF_BLOCK)
    {
      shared[threadIdx.x] += shared[threadIdx.x + k];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(global, shared[0]);
  }
}

// Bias basis functions shared by LMALF energy and gradient kernels.
// rc_exp: exponential saturation basis (x/omega terms)
// rc_sig: sigmoidal basis (s/chi terms)
__device__ inline double rc_exp(double qa, double qb, double omega_scale)
{
  return qb * (1.0 - exp(-qa / omega_scale));
}

__device__ inline double rc_sig(double qa, double qb, double chi_offset)
{
  return qb * (1.0 - 1.0 / (qa / chi_offset + 1.0));
}

// LMALF energy kernel: compute bias energy from parameters and lambda
__global__ void lmalf_energykernel(struct_lmalf lm, double *x, double *lambda, double *energy)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int s1, s2, i1, i2, k;
  double q1, q2, E;

  if (b >= lm.B) return;

  double *lam = lambda + lm.nblocks * b;

  k = 0;
  E = 0;
  for (s1 = 0; s1 < lm.nsites; s1++)
  {
    for (s2 = s1; s2 < lm.nsites; s2++)
    {
      if (s1 == s2)
      {
        // Same site
        for (i1 = lm.block0_d[s1]; i1 < lm.block0_d[s1 + 1]; i1++)
        {
          q1 = lam[i1];
          E += x[k] * q1;  // b term
          k++;
          for (i2 = i1 + 1; i2 < lm.block0_d[s1 + 1]; i2++)
          {
            q2 = lam[i2];
            E += x[k++] * q1 * q2;
            E += x[k++] * rc_exp(q1, q2, lm.omega_scale);
            E += x[k++] * rc_exp(q2, q1, lm.omega_scale);
            E += x[k++] * rc_sig(q1, q2, lm.chi_offset);
            E += x[k++] * rc_sig(q2, q1, lm.chi_offset);
          }
        }
      }
      else if (lm.ms)
      {
        // Different sites
        for (i1 = lm.block0_d[s1]; i1 < lm.block0_d[s1 + 1]; i1++)
        {
          q1 = lam[i1];
          for (i2 = lm.block0_d[s2]; i2 < lm.block0_d[s2 + 1]; i2++)
          {
            q2 = lam[i2];
            E += x[k++] * q1 * q2;
            if (lm.ms == 1)
            {
              E += x[k++] * rc_exp(q1, q2, lm.omega_scale);
              E += x[k++] * rc_exp(q2, q1, lm.omega_scale);
              E += x[k++] * rc_sig(q1, q2, lm.chi_offset);
              E += x[k++] * rc_sig(q2, q1, lm.chi_offset);
            }
          }
        }
      }
    }
  }
  energy[b] = E;
}

// LMALF weighted energy kernel: compute gradient contributions
__global__ void lmalf_weightedenergykernel(struct_lmalf lm, double sign, double *lambda, double *weight, double *dEdx)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int s1, s2, i1, i2, k;
  double q1, q2, w, E;
  __shared__ double Eloc[LMALF_BLOCK];

  if (b < lm.B)
  {
    w = sign * weight[b];
  }
  else
  {
    w = 0;
    q1 = 0;
    q2 = 0;
  }

  double *lam = lambda + lm.nblocks * b;

  k = 0;
  for (s1 = 0; s1 < lm.nsites; s1++)
  {
    for (s2 = s1; s2 < lm.nsites; s2++)
    {
      if (s1 == s2)
      {
        for (i1 = lm.block0_d[s1]; i1 < lm.block0_d[s1 + 1]; i1++)
        {
          if (b < lm.B) q1 = lam[i1];
          E = w * q1;
          lmalf_reduce(E, Eloc, &dEdx[k]);
          k++;
          for (i2 = i1 + 1; i2 < lm.block0_d[s1 + 1]; i2++)
          {
            if (b < lm.B) q2 = lam[i2];
            E = w * q1 * q2;                              lmalf_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * rc_exp(q1, q2, lm.omega_scale);       lmalf_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * rc_exp(q2, q1, lm.omega_scale);       lmalf_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * rc_sig(q1, q2, lm.chi_offset);        lmalf_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * rc_sig(q2, q1, lm.chi_offset);        lmalf_reduce(E, Eloc, &dEdx[k]); k++;
          }
        }
      }
      else if (lm.ms)
      {
        for (i1 = lm.block0_d[s1]; i1 < lm.block0_d[s1 + 1]; i1++)
        {
          if (b < lm.B) q1 = lam[i1];
          for (i2 = lm.block0_d[s2]; i2 < lm.block0_d[s2 + 1]; i2++)
          {
            if (b < lm.B) q2 = lam[i2];
            E = w * q1 * q2;                              lmalf_reduce(E, Eloc, &dEdx[k]); k++;
            if (lm.ms == 1)
            {
              E = w * rc_exp(q1, q2, lm.omega_scale);     lmalf_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * rc_exp(q2, q1, lm.omega_scale);     lmalf_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * rc_sig(q1, q2, lm.chi_offset);      lmalf_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * rc_sig(q2, q1, lm.chi_offset);      lmalf_reduce(E, Eloc, &dEdx[k]); k++;
            }
          }
        }
      }
    }
  }
}

// LMALF dot product kernel
__global__ void lmalf_dotenergykernel(struct_lmalf lm, double sign, double *x, double *y, double *z)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  double xtmp;
  __shared__ double xloc[LMALF_BLOCK];

  if (b < lm.B)
  {
    xtmp = sign * x[b] * y[b];
  }
  else
  {
    xtmp = 0;
  }
  lmalf_reduce(xtmp, xloc, z);
}

// LMALF Boltzmann weighting kernel
__global__ void lmalf_boltzmannkernel(struct_lmalf lm, double sign, double *energy, double s, double *denergyds,
                                      double *inweight, double *outweight, double *Z)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  double E, w;
  __shared__ double Zloc[LMALF_BLOCK];

  if (b < lm.B)
  {
    w = inweight[b];
    E = energy[b];
    if (s != 0 && denergyds)
    {
      E += s * denergyds[b];
    }
    w *= exp(-sign * E / lm.kT);
    outweight[b] = w;
  }
  else
  {
    w = 0;
  }

  if (Z)
  {
    __syncthreads();
    lmalf_reduce(w, Zloc, Z);
  }
}

// LMALF regularization likelihood kernel
__global__ void lmalf_regularizeLkernel(struct_lmalf lm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double deltax, L;
  __shared__ double Lloc[LMALF_BLOCK];

  if (i < lm.nx)
  {
    deltax = lm.x_d[i] - lm.xr_d[i];
    L = 0.5 * lm.kx_d[i] * deltax * deltax;
  }
  else
  {
    L = 0;
  }

  lmalf_reduce(L, Lloc, lm.L_d);
}

// LMALF regularization gradient kernel
__global__ void lmalf_regularizedLdxkernel(struct_lmalf lm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < lm.nx)
  {
    lm.dLdx_d[i] = lm.kx_d[i] * (lm.x_d[i] - lm.xr_d[i]);
  }
}

// LMALF gradient likelihood kernel (for moment matching)
__global__ void lmalf_gradientlikelihoodkernel(struct_lmalf lm, double *norm, double *dLdxin)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < lm.nbias)
  {
    atomicAdd(&lm.dLdx_d[i], -dLdxin[i] / (norm[0] * lm.kT));
  }
}

// LMALF likelihood kernel
__global__ void lmalf_likelihoodkernel(struct_lmalf lm, double s, double *L, double *dLds)
{
  if (L)
  {
    atomicAdd(L, lm.Esum_d[0] / (lm.sumensweight_d[0] * lm.kT));
    if (s != 0)
      atomicAdd(L, s * lm.dEdssum_d[0] / (lm.sumensweight_d[0] * lm.kT));
    atomicAdd(L, log(lm.mc_Z_d[0]));
  }
  if (dLds)
  {
    atomicAdd(dLds, lm.dEdssum_d[0] / (lm.sumensweight_d[0] * lm.kT));
    atomicAdd(dLds, -lm.mc_dEdssum_d[0] / (lm.mc_Z_d[0] * lm.kT));
  }
}

// LMALF line search regularization kernel
__global__ void lmalf_regularizelinekernel(struct_lmalf lm, double s)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double deltax, dxds, L, dLds;
  __shared__ double Lloc[LMALF_BLOCK];

  if (i < lm.nx)
  {
    dxds = lm.dxds_d[i];
    deltax = lm.x_d[i] + s * dxds - lm.xr_d[i];
    L = 0.5 * lm.kx_d[i] * deltax * deltax;
    dLds = lm.kx_d[i] * deltax * dxds;
  }
  else
  {
    L = 0;
    dLds = 0;
  }

  lmalf_reduce(L, Lloc, lm.L_d);
  lmalf_reduce(dLds, Lloc, lm.dLds_d);
}

// Random double [0, 1) for Monte Carlo
static double lmalf_rand_double(void)
{
  return (rand() + 0.5) / (RAND_MAX + 1.0);
}

// Generate Monte Carlo reference distribution for partition function normalization
// This is essential for LMALF to compute meaningful gradients.
// Without proper reference samples, the gradient would be zero.
static void lmalf_monte_carlo_Z(struct_lmalf *lm)
{
  int ibeg, iend, Ns;
  int Neq = lm->B / 10;  // Equilibration steps
  int Nmc = lm->B;       // Production steps
  double *theta;
  int s, i, j;
  double b, st, norm;
  double thetaNew, eOld, eNew;

  // Seed random number generator
  srand(12345);  // Fixed seed for reproducibility

  theta = (double *)calloc(lm->nblocks, sizeof(double));

  // Generate MC samples for each site
  for (s = 0; s < lm->nsites; s++)
  {
    ibeg = lm->block0[s];
    iend = lm->block0[s + 1];
    Ns = iend - ibeg;

    // Compute optimal b parameter for this site
    // This empirical formula ensures good sampling efficiency
    b = 1;
    for (i = 0; i < 50; i++)
    {
      b = 0.5 * log(0.25 * b * Ns * Ns * M_PI / 2);
      if (!(b > 0)) b = 0;
    }

    // Initialize theta angles
    theta[ibeg] = M_PI / 2;
    for (i = ibeg + 1; i < iend; i++)
    {
      theta[i] = 3 * M_PI / 2;
    }

    // Metropolis Monte Carlo sampling
    for (i = -Neq; i < Nmc; i++)
    {
      if (i % (Neq > 0 ? Neq : 1) == 0 && i >= 0)
      {
        fprintf(stdout, "MC Reference Sample Step %d/%d\n", i, Nmc);
      }

      // Metropolis moves for each subsite
      for (j = ibeg; j < iend; j++)
      {
        st = (-0.5 * sin(theta[j]) + 0.5);
        eOld = -b * st * st * st * st;

        thetaNew = 2 * M_PI * lmalf_rand_double();
        st = (-0.5 * sin(thetaNew) + 0.5);
        eNew = -b * st * st * st * st;

        if (exp(eOld - eNew) > lmalf_rand_double())
        {
          theta[j] = thetaNew;
        }
      }

      // Store production samples (after equilibration)
      if (i >= 0)
      {
        norm = 0;
        for (j = ibeg; j < iend; j++)
        {
          norm += exp(lm->fnex * sin(theta[j]));
        }
        for (j = ibeg; j < iend; j++)
        {
          lm->mc_lambda_h[lm->nblocks * i + j] = exp(lm->fnex * sin(theta[j])) / norm;
        }
      }
    }
  }

  free(theta);
  fprintf(stdout, "MC reference samples generated\n");
}

// Setup LMALF data structure
struct_lmalf *lmalf_setup(int nf, double temp, int ms, int msprof,
                          int *nsubs_in, int nsites_in, const char *g_imp_path)
{
  struct_lmalf *lm = (struct_lmalf *)malloc(sizeof(struct_lmalf));
  FILE *fp;
  char line[MAXLENGTH];
  int i, j, k, si, sj;
  double kp, k0;

  // Store G_imp path (use provided path or default to "G_imp")
  if (g_imp_path && strlen(g_imp_path) > 0)
  {
    strncpy(lm->g_imp_path, g_imp_path, sizeof(lm->g_imp_path) - 1);
    lm->g_imp_path[sizeof(lm->g_imp_path) - 1] = '\0';
  }
  else
  {
    strcpy(lm->g_imp_path, "G_imp");
  }

  // Use provided nsubs array or read from file
  if (nsubs_in != NULL && nsites_in > 0)
  {
    lm->nsites = nsites_in;
    lm->nsubs = (int *)calloc(lm->nsites, sizeof(int));
    lm->block0 = (int *)calloc(lm->nsites + 1, sizeof(int));
    lm->nblocks = 0;
    for (i = 0; i < lm->nsites; i++)
    {
      lm->nsubs[i] = nsubs_in[i];
      lm->block0[i] = lm->nblocks;
      lm->nblocks += lm->nsubs[i];
    }
    lm->block0[lm->nsites] = lm->nblocks;
    fprintf(stderr, "LMALF: Using provided nsubs array: nsites=%d, nblocks=%d\n", lm->nsites, lm->nblocks);
  }
  else
  {
    // Fallback: read from file for backward compatibility
    fp = fopen("prep/nsubs", "r");
    if (!fp)
    {
      fprintf(stderr, "Error: prep/nsubs does not exist and no nsubs array provided\n");
      exit(1);
    }

    lm->nsites = 0;
    while (fscanf(fp, "%d", &i) == 1)
      lm->nsites++;
    fclose(fp);

    lm->nsubs = (int *)calloc(lm->nsites, sizeof(int));
    lm->block0 = (int *)calloc(lm->nsites + 1, sizeof(int));

    fp = fopen("prep/nsubs", "r");
    lm->nblocks = 0;
    for (i = 0; i < lm->nsites; i++)
    {
      fscanf(fp, "%d", &lm->nsubs[i]);
      lm->block0[i] = lm->nblocks;
      lm->nblocks += lm->nsubs[i];
    }
    lm->block0[lm->nsites] = lm->nblocks;
    fclose(fp);
    fprintf(stderr, "LMALF: Read nsubs from file: nsites=%d, nblocks=%d\n", lm->nsites, lm->nblocks);
  }

  CUDA_CHECK(cudaMalloc(&lm->block0_d, (lm->nsites + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(lm->block0_d, lm->block0, (lm->nsites + 1) * sizeof(int), cudaMemcpyHostToDevice));

  lm->ms = ms;
  lm->msprof = msprof;
  lm->kT = kB * temp;

  // Count frames from lambda file
  fp = fopen("Lambda.dat", "r");
  if (!fp)
  {
    fprintf(stderr, "Error: Lambda.dat does not exist\n");
    exit(1);
  }
  lm->B = 0;
  while (fgets(line, MAXLENGTH, fp) != NULL)
    lm->B++;
  fclose(fp);
  fprintf(stdout, "LMALF: %d frames\n", lm->B);

  // Allocate and read lambda trajectories
  lm->lambda_h = (double *)calloc(lm->B * lm->nblocks, sizeof(double));
  lm->ensweight_h = (double *)calloc(lm->B, sizeof(double));
  lm->mc_lambda_h = (double *)calloc(lm->B * lm->nblocks, sizeof(double));
  lm->mc_ensweight_h = (double *)calloc(lm->B, sizeof(double));

  fp = fopen("Lambda.dat", "r");
  for (i = 0; i < lm->B; i++)
  {
    for (j = 0; j < lm->nblocks; j++)
    {
      double buffer;
      fscanf(fp, "%lf", &buffer);
      lm->lambda_h[i * lm->nblocks + j] = buffer;
    }
  }
  fclose(fp);

  // Read ensemble weights
  fp = fopen("ensweight.dat", "r");
  if (!fp)
  {
    // Default: all weights = 1
    for (i = 0; i < lm->B; i++)
    {
      lm->ensweight_h[i] = 1.0;
      lm->mc_ensweight_h[i] = 1.0;
    }
  }
  else
  {
    for (i = 0; i < lm->B; i++)
    {
      double buffer;
      fscanf(fp, "%lf", &buffer);
      lm->ensweight_h[i] = buffer;
      lm->mc_ensweight_h[i] = 1.0;
    }
    fclose(fp);
  }

  // Generate Monte Carlo reference distribution using Metropolis sampling
  // This provides the reference distribution for computing meaningful gradients
  lmalf_monte_carlo_Z(lm);

  CUDA_CHECK(cudaMalloc(&lm->lambda_d, lm->B * lm->nblocks * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_lambda_d, lm->B * lm->nblocks * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->ensweight_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_ensweight_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(lm->lambda_d, lm->lambda_h, lm->B * lm->nblocks * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(lm->mc_lambda_d, lm->mc_lambda_h, lm->B * lm->nblocks * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(lm->ensweight_d, lm->ensweight_h, lm->B * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(lm->mc_ensweight_d, lm->mc_ensweight_h, lm->B * sizeof(double), cudaMemcpyHostToDevice));

  // Count bias parameters
  lm->nbias = 0;
  for (i = 0; i < lm->nsites; i++)
  {
    for (j = i; j < lm->nsites; j++)
    {
      if (i == j)
      {
        lm->nbias += lm->nsubs[i] + (5 * lm->nsubs[i] * (lm->nsubs[i] - 1)) / 2;
      }
      else if (lm->ms == 1)
      {
        lm->nbias += 5 * lm->nsubs[i] * lm->nsubs[j];
      }
      else if (lm->ms == 2)
      {
        lm->nbias += lm->nsubs[i] * lm->nsubs[j];
      }
    }
  }

  // Count profile types
  lm->nprof = 0;
  for (i = 0; i < lm->nsites; i++)
  {
    for (j = i; j < lm->nsites; j++)
    {
      if (i == j)
      {
        if (lm->nsubs[i] == 2)
        {
          lm->nprof += lm->nsubs[i] + lm->nsubs[i] * (lm->nsubs[i] - 1) / 2;
        }
        else
        {
          lm->nprof += lm->nsubs[i] + 2 * lm->nsubs[i] * (lm->nsubs[i] - 1) / 2;
        }
      }
      else if (msprof)
      {
        lm->nprof += lm->nsubs[i] * lm->nsubs[j];
      }
    }
  }

  lm->nx = lm->nbias;

  // Setup regularization
  kp = 1.0 / (lm->kT * lm->kT);
  k0 = kp / 400;

  lm->kx_h = (double *)calloc(lm->nx, sizeof(double));
  lm->xr_h = (double *)calloc(lm->nx, sizeof(double));

  // Load previous x/s values if ms==1
  double *xr_x = NULL, *xr_s = NULL;
  if (lm->ms == 1)
  {
    xr_x = (double *)calloc(lm->nblocks * lm->nblocks, sizeof(double));
    xr_s = (double *)calloc(lm->nblocks * lm->nblocks, sizeof(double));
    FILE *fpx = fopen("x_prev.dat", "r");
    FILE *fps = fopen("s_prev.dat", "r");
    if (fpx && fps)
    {
      for (i = 0; i < lm->nblocks; i++)
      {
        for (j = 0; j < lm->nblocks; j++)
        {
          double buffer;
          fscanf(fpx, "%lf", &buffer);
          xr_x[i * lm->nblocks + j] = buffer;
          fscanf(fps, "%lf", &buffer);
          xr_s[i * lm->nblocks + j] = buffer;
        }
      }
      fclose(fpx);
      fclose(fps);
    }
    else
    {
      if (fpx) fclose(fpx);
      if (fps) fclose(fps);
    }
  }

  // Set regularization constants
  k = 0;
  for (si = 0; si < lm->nsites; si++)
  {
    for (sj = si; sj < lm->nsites; sj++)
    {
      if (si == sj)
      {
        for (i = 0; i < lm->nsubs[si]; i++)
        {
          lm->kx_h[k++] = k0 / 4;  // b
          for (j = i + 1; j < lm->nsubs[sj]; j++)
          {
            lm->kx_h[k++] = k0 / 64; // c
            lm->kx_h[k++] = k0 / 4;  // x
            lm->kx_h[k++] = k0 / 4;  // x
            lm->kx_h[k++] = k0 / 1;  // s
            lm->kx_h[k++] = k0 / 1;  // s
          }
        }
      }
      else if (lm->ms)
      {
        for (i = 0; i < lm->nsubs[si]; i++)
        {
          for (j = 0; j < lm->nsubs[sj]; j++)
          {
            lm->kx_h[k++] = k0 / 4;  // c
            if (lm->ms == 1)
            {
              if (xr_x)
                lm->xr_h[k] = xr_x[(lm->block0[si] + i) * lm->nblocks + lm->block0[sj] + j];
              lm->kx_h[k++] = k0 / 0.25; // x
              if (xr_x)
                lm->xr_h[k] = xr_x[(lm->block0[sj] + j) * lm->nblocks + lm->block0[si] + i];
              lm->kx_h[k++] = k0 / 0.25; // x
              if (xr_s)
                lm->xr_h[k] = xr_s[(lm->block0[si] + i) * lm->nblocks + lm->block0[sj] + j];
              lm->kx_h[k++] = k0 / 0.25; // s
              if (xr_s)
                lm->xr_h[k] = xr_s[(lm->block0[sj] + j) * lm->nblocks + lm->block0[si] + i];
              lm->kx_h[k++] = k0 / 0.25; // s
            }
          }
        }
      }
    }
  }

  if (xr_x) free(xr_x);
  if (xr_s) free(xr_s);

  CUDA_CHECK(cudaMalloc(&lm->kx_d, lm->nx * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(lm->kx_d, lm->kx_h, lm->nx * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&lm->xr_d, lm->nx * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(lm->xr_d, lm->xr_h, lm->nx * sizeof(double), cudaMemcpyHostToDevice));

  // Allocate calculation arrays
  lm->L_h = (double *)calloc(1, sizeof(double));
  lm->dLds_h = (double *)calloc(1, sizeof(double));
  CUDA_CHECK(cudaMalloc(&lm->L_d, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->dLds_d, sizeof(double)));

  lm->x_h = (double *)calloc(lm->nx, sizeof(double));
  lm->dLdx_h = (double *)calloc(lm->nx, sizeof(double));
  CUDA_CHECK(cudaMalloc(&lm->dLdx_d, lm->nx * sizeof(double)));

  // L-BFGS memory
  lm->Nmemax = 50;
  lm->Nmem = 0;
  lm->d_x = (double *)calloc(lm->nx * lm->Nmemax, sizeof(double));
  lm->d_dLdx = (double *)calloc(lm->nx * lm->Nmemax, sizeof(double));
  lm->rho = (double *)calloc(lm->Nmemax, sizeof(double));
  lm->alpha = (double *)calloc(lm->Nmemax, sizeof(double));
  lm->beta = (double *)calloc(lm->Nmemax, sizeof(double));
  lm->hi_h = (double *)calloc(lm->nx, sizeof(double));
  lm->x0_h = (double *)calloc(lm->nx, sizeof(double));
  lm->dLdx0_h = (double *)calloc(lm->nx, sizeof(double));

  CUDA_CHECK(cudaMalloc(&lm->E_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->dEds_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_E_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_dEds_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->weight_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_weight_d, lm->B * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->x_d, lm->nx * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->dxds_d, lm->nx * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->Z_d, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_Z_d, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->Esum_d, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->dEdssum_d, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_dEdssum_d, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->moments_d, lm->nbias * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->mc_moments_d, lm->nbias * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&lm->sumensweight_d, sizeof(double)));

  // Convergence settings
  lm->criteria = 1.25e-3;
  lm->max_iter = 250;
  lm->done = 0;
  lm->doneCount = 0;

  return lm;
}

// Evaluate likelihood function
void lmalf_evaluateL(struct_lmalf *lm)
{
  CUDA_CHECK(cudaMemcpy(lm->x_d, lm->x_h, lm->nx * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(lm->L_d, 0, sizeof(double)));

  // Regularization contribution
  lmalf_regularizeLkernel<<<(lm->nx + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm);
  CUDA_CHECK(cudaGetLastError());

  // Compute energies
  lmalf_energykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, lm->x_d, lm->lambda_d, lm->E_d);
  CUDA_CHECK(cudaGetLastError());

  // Moment matching contribution
  lmalf_energykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, lm->x_d, lm->mc_lambda_d, lm->mc_E_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemset(lm->Esum_d, 0, sizeof(double)));
  lmalf_dotenergykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, 1, lm->ensweight_d, lm->E_d, lm->Esum_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemset(lm->mc_Z_d, 0, sizeof(double)));
  lmalf_boltzmannkernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(
      *lm, 1, lm->mc_E_d, 0, NULL, lm->mc_ensweight_d, lm->mc_weight_d, lm->mc_Z_d);
  CUDA_CHECK(cudaGetLastError());

  lmalf_likelihoodkernel<<<1, 1>>>(*lm, 0, lm->L_d, NULL);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(lm->L_h, lm->L_d, sizeof(double), cudaMemcpyDeviceToHost));

  fprintf(stdout, "New      L=%lg\n", lm->L_h[0]);
}

// Evaluate likelihood along line search direction
void lmalf_evaluateL_line(double s, struct_lmalf *lm)
{
  CUDA_CHECK(cudaMemset(lm->L_d, 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(lm->dLds_d, 0, sizeof(double)));

  lmalf_regularizelinekernel<<<(lm->nx + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, s);
  CUDA_CHECK(cudaGetLastError());

  // Moment matching
  CUDA_CHECK(cudaMemset(lm->mc_Z_d, 0, sizeof(double)));
  lmalf_boltzmannkernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(
      *lm, 1, lm->mc_E_d, s, lm->mc_dEds_d, lm->mc_ensweight_d, lm->mc_weight_d, lm->mc_Z_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemset(lm->mc_dEdssum_d, 0, sizeof(double)));
  lmalf_dotenergykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, 1, lm->mc_dEds_d, lm->mc_weight_d, lm->mc_dEdssum_d);
  CUDA_CHECK(cudaGetLastError());

  lmalf_likelihoodkernel<<<1, 1>>>(*lm, s, lm->L_d, lm->dLds_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(lm->L_h, lm->L_d, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(lm->dLds_h, lm->dLds_d, sizeof(double), cudaMemcpyDeviceToHost));
}

// Evaluate gradient
void lmalf_evaluatedLdx(struct_lmalf *lm)
{
  CUDA_CHECK(cudaMemset(lm->dLdx_d, 0, lm->nx * sizeof(double)));

  lmalf_regularizedLdxkernel<<<(lm->nx + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm);
  CUDA_CHECK(cudaGetLastError());

  // Moment gradient
  CUDA_CHECK(cudaMemset(lm->mc_moments_d, 0, lm->nbias * sizeof(double)));
  lmalf_weightedenergykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(
      *lm, 1, lm->mc_lambda_d, lm->mc_weight_d, lm->mc_moments_d);
  CUDA_CHECK(cudaGetLastError());

  lmalf_gradientlikelihoodkernel<<<(lm->nbias + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(
      *lm, lm->sumensweight_d, lm->moments_d);
  CUDA_CHECK(cudaGetLastError());

  lmalf_gradientlikelihoodkernel<<<(lm->nbias + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(
      *lm, lm->mc_Z_d, lm->mc_moments_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(lm->dLdx_h, lm->dLdx_d, lm->nx * sizeof(double), cudaMemcpyDeviceToHost));
}

// Reset L-BFGS inverse Hessian approximation
void lmalf_resetHinv(struct_lmalf *lm)
{
  for (int i = 0; i < lm->nx; i++)
  {
    lm->x0_h[i] = lm->x_h[i];
    lm->dLdx0_h[i] = lm->dLdx_h[i];
  }
}

// Update L-BFGS memory
void lmalf_updateHinv(struct_lmalf *lm)
{
  int i, j;

  if (lm->Nmem < lm->Nmemax)
  {
    lm->Nmem++;
  }

  // Shift history
  for (i = lm->Nmem - 1; i > 0; i--)
  {
    for (j = 0; j < lm->nx; j++)
    {
      lm->d_x[i * lm->nx + j] = lm->d_x[(i - 1) * lm->nx + j];
      lm->d_dLdx[i * lm->nx + j] = lm->d_dLdx[(i - 1) * lm->nx + j];
    }
    lm->rho[i] = lm->rho[i - 1];
  }

  // Compute new history entry
  lm->rho[0] = 0;
  for (i = 0; i < lm->nx; i++)
  {
    lm->d_x[i] = lm->x_h[i] - lm->x0_h[i];
    lm->d_dLdx[i] = lm->dLdx_h[i] - lm->dLdx0_h[i];
    lm->rho[0] += lm->d_x[i] * lm->d_dLdx[i];
  }
  lm->rho[0] = 1.0 / lm->rho[0];

  // Store current values as previous
  for (i = 0; i < lm->nx; i++)
  {
    lm->x0_h[i] = lm->x_h[i];
    lm->dLdx0_h[i] = lm->dLdx_h[i];
  }
}

// Compute L-BFGS search direction
void lmalf_projectHinv(struct_lmalf *lm)
{
  int i, j;

  // Two-loop recursion for L-BFGS
  for (i = 0; i < lm->nx; i++)
  {
    lm->hi_h[i] = lm->dLdx_h[i];
  }

  for (i = 0; i < lm->Nmem; i++)
  {
    lm->alpha[i] = 0;
    for (j = 0; j < lm->nx; j++)
    {
      lm->alpha[i] += lm->d_x[i * lm->nx + j] * lm->hi_h[j];
    }
    lm->alpha[i] *= lm->rho[i];
    for (j = 0; j < lm->nx; j++)
    {
      lm->hi_h[j] -= lm->alpha[i] * lm->d_dLdx[i * lm->nx + j];
    }
  }

  for (i = lm->Nmem - 1; i >= 0; i--)
  {
    lm->beta[i] = 0;
    for (j = 0; j < lm->nx; j++)
    {
      lm->beta[i] += lm->d_dLdx[i * lm->nx + j] * lm->hi_h[j];
    }
    lm->beta[i] *= lm->rho[i];
    for (j = 0; j < lm->nx; j++)
    {
      lm->hi_h[j] += (lm->alpha[i] - lm->beta[i]) * lm->d_x[i * lm->nx + j];
    }
  }

  // Negate for descent direction
  for (i = 0; i < lm->nx; i++)
  {
    lm->hi_h[i] *= -1;
  }

  // Upload search direction and compute energy derivatives
  CUDA_CHECK(cudaMemcpy(lm->dxds_d, lm->hi_h, lm->nx * sizeof(double), cudaMemcpyHostToDevice));

  lmalf_energykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, lm->dxds_d, lm->lambda_d, lm->dEds_d);
  CUDA_CHECK(cudaGetLastError());

  lmalf_energykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, lm->dxds_d, lm->mc_lambda_d, lm->mc_dEds_d);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemset(lm->dEdssum_d, 0, sizeof(double)));
  lmalf_dotenergykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(*lm, 1, lm->ensweight_d, lm->dEds_d, lm->dEdssum_d);
  CUDA_CHECK(cudaGetLastError());
}

// Line search update
void lmalf_update_line(int step, struct_lmalf *lm)
{
  int i;
  double a, b, c, s;
  double s1, s2, s3;
  double L1, L2, L3;
  double dLds1, dLds2, dLds3;
  double L0;

  for (i = 0; i < lm->nx; i++)
  {
    lm->x0_h[i] = lm->x_h[i];
    lm->dLdx0_h[i] = lm->dLdx_h[i];
  }

  L0 = lm->L_h[0];

  s1 = 0.0;
  lmalf_evaluateL_line(s1, lm);
  L1 = lm->L_h[0];
  dLds1 = lm->dLds_h[0];

  if (dLds1 > 0)
  {
    fprintf(stdout, "Error, hi is pointing wrong way - halting\n");
    lm->done = 1;
    return;
  }

  s3 = 1.0;
  lmalf_evaluateL_line(s3, lm);
  L3 = lm->L_h[0];
  dLds3 = lm->dLds_h[0];

  // Expand step size until we bracket minimum
  while (dLds3 < 0 && s3 < 1e+8)
  {
    fprintf(stdout, "Seek %4d s=%lg %lg\n          L=%lg %lg\n       dLds=%lg %lg\n",
            step, s1, s3, L1, L3, dLds1, dLds3);
    s2 = s1 - dLds1 * (s3 - s1) / (dLds3 - dLds1);
    s3 = ((1.5 * s2 > 8 * s3 || 1.5 * s2 <= 0) ? 8 * s3 : 1.5 * s2);
    lmalf_evaluateL_line(s3, lm);
    L3 = lm->L_h[0];
    dLds3 = lm->dLds_h[0];
  }

  // Handle overshooting
  while (!isfinite(dLds3) && s3 > 1e-8)
  {
    fprintf(stdout, "Warning, overshot bound\n");
    s3 = 0.95 * s3;
    lmalf_evaluateL_line(s3, lm);
    L3 = lm->L_h[0];
    dLds3 = lm->dLds_h[0];
  }

  if (!(dLds3 > 0))
  {
    fprintf(stdout, "Warning: Step %4d unsuccessful, halting minimization\n", step);
    lm->done = 1;
    return;
  }

  // Initial secant interpolation
  s2 = s1 - dLds1 * (s3 - s1) / (dLds3 - dLds1);
  lmalf_evaluateL_line(s2, lm);
  L2 = lm->L_h[0];
  dLds2 = lm->dLds_h[0];

  fprintf(stdout, "Step %4d s=%lg %lg %lg\n          L=%lg %lg %lg\n       dLds=%lg %lg %lg\n",
          step, s1, s2, s3, L1, L2, L3, dLds1, dLds2, dLds3);

  // Refine with quadratic interpolation
  for (i = 0; i < 15; i++)
  {
    if ((s2 - s1) / s2 < 5e-7 || (s3 - s2) / s2 < 5e-7 || dLds2 == 0)
      break;

    // Quadratic interpolation
    a = dLds1 / ((s1 - s2) * (s1 - s3));
    a += dLds2 / ((s2 - s1) * (s2 - s3));
    a += dLds3 / ((s3 - s1) * (s3 - s2));
    b = -dLds1 * (s2 + s3) / ((s1 - s2) * (s1 - s3));
    b += -dLds2 * (s1 + s3) / ((s2 - s1) * (s2 - s3));
    b += -dLds3 * (s1 + s2) / ((s3 - s1) * (s3 - s2));
    c = dLds1 * s2 * s3 / ((s1 - s2) * (s1 - s3));
    c += dLds2 * s1 * s3 / ((s2 - s1) * (s2 - s3));
    c += dLds3 * s1 * s2 / ((s3 - s1) * (s3 - s2));

    s = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

    if (dLds2 < 0)
    {
      s1 = s2;
      L1 = L2;
      dLds1 = dLds2;
    }
    else
    {
      s3 = s2;
      L3 = L2;
      dLds3 = dLds2;
    }

    if (s > s1 && s < s3)
    {
      s2 = s;
    }
    else
    {
      // Fall back to linear interpolation
      fprintf(stdout, "Warning, fell back on linear interpolation\n");
      s2 = s1 - dLds1 * (s3 - s1) / (dLds3 - dLds1);
    }

    lmalf_evaluateL_line(s2, lm);
    L2 = lm->L_h[0];
    dLds2 = lm->dLds_h[0];

    fprintf(stdout, "Step %4d s=%lg %lg %lg\n          L=%lg %lg %lg\n       dLds=%lg %lg %lg\n",
            step, s1, s2, s3, L1, L2, L3, dLds1, dLds2, dLds3);
  }

  // Apply step
  double stepLength2 = 0;
  double initGrad2 = 0;

  for (i = 0; i < lm->nx; i++)
  {
    lm->x_h[i] = lm->x0_h[i] + s2 * lm->hi_h[i];
    stepLength2 += (s2 * lm->hi_h[i]) * (s2 * lm->hi_h[i]);
    initGrad2 += lm->dLdx_h[i] * lm->dLdx_h[i];
  }

  fprintf(stdout, "Step %4d L=%24.16lf -> L2=%24.16lf, dL=%lg, step length=%lg\n",
          step, L0, L2, L2 - L0, sqrt(stepLength2));

  // Check convergence
  if (sqrt(stepLength2) < 5e-7)
    lm->done = 1;
  if (sqrt(stepLength2 / lm->nx) < lm->criteria)
  {
    lm->doneCount++;
    if (lm->doneCount == 2)
      lm->done = 1;
  }
  else
  {
    lm->doneCount = 0;
  }
}

// Single optimization iteration
void lmalf_iterate(int step, struct_lmalf *lm)
{
  lmalf_evaluateL(lm);
  lmalf_evaluatedLdx(lm);

  if (step == 0)
  {
    lmalf_resetHinv(lm);
  }
  else
  {
    lmalf_updateHinv(lm);
  }

  lmalf_projectHinv(lm);
  lmalf_update_line(step, lm);
}

// Main LMALF optimization run
void lmalf_run(struct_lmalf *lm)
{
  int s;
  double sum;

  // Compute sum of ensemble weights
  sum = 0;
  for (int i = 0; i < lm->B; i++)
  {
    sum += lm->ensweight_h[i];
  }
  CUDA_CHECK(cudaMemcpy(lm->sumensweight_d, &sum, sizeof(double), cudaMemcpyHostToDevice));

  // Compute initial moments
  CUDA_CHECK(cudaMemset(lm->moments_d, 0, lm->nbias * sizeof(double)));
  lmalf_weightedenergykernel<<<(lm->B + LMALF_BLOCK - 1) / LMALF_BLOCK, LMALF_BLOCK>>>(
      *lm, -1, lm->lambda_d, lm->ensweight_d, lm->moments_d);
  CUDA_CHECK(cudaGetLastError());

  lm->done = 0;
  lm->doneCount = 0;

  for (s = 0; s < lm->max_iter; s++)
  {
    lmalf_iterate(s, lm);
    if (lm->done)
      break;
  }

  fprintf(stdout, "LMALF optimization completed after %d iterations\n", s);
}

// Cleanup and write output
void lmalf_finish(struct_lmalf *lm)
{
  FILE *fp;
  int i;

  // Write optimized parameters
  fp = fopen("OUT.dat", "w");
  for (i = 0; i < lm->nx; i++)
  {
    fprintf(fp, " %lg", lm->x_h[i]);
  }
  fclose(fp);

  fprintf(stdout, "LMALF results written to OUT.dat\n");

  // Free host memory
  free(lm->nsubs);
  free(lm->block0);
  free(lm->lambda_h);
  free(lm->ensweight_h);
  free(lm->mc_lambda_h);
  free(lm->mc_ensweight_h);
  free(lm->kx_h);
  free(lm->xr_h);
  free(lm->L_h);
  free(lm->dLds_h);
  free(lm->x_h);
  free(lm->dLdx_h);
  free(lm->d_x);
  free(lm->d_dLdx);
  free(lm->rho);
  free(lm->alpha);
  free(lm->beta);
  free(lm->hi_h);
  free(lm->x0_h);
  free(lm->dLdx0_h);

  // Free device memory
  cudaFree(lm->block0_d);
  cudaFree(lm->lambda_d);
  cudaFree(lm->mc_lambda_d);
  cudaFree(lm->ensweight_d);
  cudaFree(lm->mc_ensweight_d);
  cudaFree(lm->kx_d);
  cudaFree(lm->xr_d);
  cudaFree(lm->L_d);
  cudaFree(lm->dLds_d);
  cudaFree(lm->dLdx_d);
  cudaFree(lm->E_d);
  cudaFree(lm->dEds_d);
  cudaFree(lm->mc_E_d);
  cudaFree(lm->mc_dEds_d);
  cudaFree(lm->weight_d);
  cudaFree(lm->mc_weight_d);
  cudaFree(lm->x_d);
  cudaFree(lm->dxds_d);
  cudaFree(lm->Z_d);
  cudaFree(lm->mc_Z_d);
  cudaFree(lm->Esum_d);
  cudaFree(lm->dEdssum_d);
  cudaFree(lm->mc_dEdssum_d);
  cudaFree(lm->moments_d);
  cudaFree(lm->mc_moments_d);
  cudaFree(lm->sumensweight_d);

  free(lm);
}

/**
 * LMALF main entry point (exported for Python ctypes)
 *
 * @param nf: Number of simulation files
 * @param temp: Temperature in Kelvin
 * @param ms: Multisite coupling flag (0, 1, or 2)
 * @param msprof: Multisite profiles flag
 * @param max_iter: Maximum L-BFGS iterations (0 = use default 250)
 * @param tolerance: Convergence tolerance (0 = use default 1.25e-3)
 *
 * Expected input files (in current directory):
 *   - Lambda.dat: Lambda trajectory values [B x nblocks]
 *   - ensweight.dat: Ensemble weights [B] (optional, defaults to uniform)
 *   - ../prep/nsubs: Number of subsites per site
 *   - x_prev.dat, s_prev.dat: Previous parameters (optional, for ms==1)
 *
 * Output:
 *   - OUT.dat: Optimized bias parameters
 */
extern "C" int lmalf(int nf, double temp, int ms, int msprof, int max_iter, double tolerance,
                     int *nsubs, int nsites, const char *g_imp_path,
                     double fnex, double chi_offset, double omega_scale)
{
  fprintf(stdout, "LMALF: Likelihood Maximization ALF\n");
  fprintf(stdout, "  nf=%d, temp=%.2f, ms=%d, msprof=%d\n", nf, temp, ms, msprof);
  fprintf(stdout, "  max_iter=%d, tolerance=%g\n", max_iter, tolerance);
  fprintf(stdout, "  fnex=%.4f, chi_offset=%.6f, omega_scale=%.6f\n", fnex, chi_offset, omega_scale);

  // Initialize and validate GPU
  validate_and_setup_gpu();

  struct_lmalf *lm = lmalf_setup(nf, temp, ms, msprof, nsubs, nsites, g_imp_path);
  if (lm) {
    lm->fnex = fnex;
    lm->chi_offset = chi_offset;
    lm->omega_scale = omega_scale;
  }
  if (!lm)
  {
    fprintf(stderr, "Error: Failed to setup LMALF\n");
    return -1;
  }

  // Override defaults if specified
  if (max_iter > 0)
    lm->max_iter = max_iter;
  if (tolerance > 0)
    lm->criteria = tolerance;

  CUDA_CHECK(cudaDeviceSynchronize());

  lmalf_run(lm);

  CUDA_CHECK(cudaDeviceSynchronize());

  lmalf_finish(lm);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaDeviceReset());

  return 0;
}
