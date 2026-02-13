/**
 * nonlinear.cu — L-BFGS nonlinear solver for ALF bias optimization.
 *
 * Standalone CUDA solver combining moment-matching loss with L2 regularization.
 * Optimizes bias parameters (b, c, x, s [, t, u]) via L-BFGS with bracketing
 * line search and quadratic interpolation.
 *
 * Architecture:
 *   - Energy/gradient kernels run on GPU (per-frame parallelism)
 *   - L-BFGS two-loop recursion runs on host (sequential, small nx)
 *   - Line search alternates host decisions with GPU kernel launches
 *
 * Build: nvcc -shared -Xcompiler -fPIC -arch=all-major -o libnonlinear.so nonlinear.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "nonlinear.h"

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

// ─── CUDA error checking ───────────────────────────────────────────────────

#define NL_CHECK(call)                                                             \
  do                                                                               \
  {                                                                                \
    cudaError_t err = call;                                                        \
    if (err != cudaSuccess)                                                        \
    {                                                                              \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err));                                             \
      cudaDeviceReset();                                                           \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  } while (0)

// ─── GPU setup ─────────────────────────────────────────────────────────────

static void nl_validate_gpu()
{
  int deviceCount = 0;
  NL_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0)
  {
    fprintf(stderr, "nonlinear: No CUDA devices found\n");
    exit(EXIT_FAILURE);
  }

  int device;
  NL_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  NL_CHECK(cudaGetDeviceProperties(&prop, device));
  fprintf(stdout, "nonlinear: Using GPU %d: %s (SM %d.%d, %.0f MB)\n",
          device, prop.name, prop.major, prop.minor,
          prop.totalGlobalMem / (1024.0 * 1024.0));

  if (prop.major < 6)
  {
    fprintf(stderr, "nonlinear: GPU SM %d.%d < 6.0, double atomicAdd requires SM 6.0+\n",
            prop.major, prop.minor);
    exit(EXIT_FAILURE);
  }

  NL_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
}

// ─── Bias basis functions ──────────────────────────────────────────────────

// x-term: exponential saturation 1-exp(-λ/ω)
__device__ inline double nl_rc_exp(double qa, double qb, double omega_scale)
{
  return qb * (1.0 - exp(-qa / omega_scale));
}

// s-term: sigmoid λ/(λ+χ)
__device__ inline double nl_rc_sig(double qa, double qb, double chi_offset)
{
  return qb * (1.0 - 1.0 / (qa / chi_offset + 1.0));
}

// t-term: opposite-endpoint sigmoid — activates near λ→1 (mirror of s-term)
__device__ inline double nl_rc_omega2(double qa, double qb, double chi_offset_t)
{
  return -qb * (1.0 - 1.0 / (qa / (-(1.0 + chi_offset_t)) + 1.0));
}

// u-term: s-term sigmoid with qb² (quadratic coupling in second lambda)
__device__ inline double nl_rc_omega3(double qa, double qb, double chi_offset_u)
{
  return qb * qb * (1.0 - 1.0 / (qa / chi_offset_u + 1.0));
}

// ─── Block reduction ───────────────────────────────────────────────────────

__device__ void nl_reduce(double local, double *shared, double *global)
{
  int k;
  shared[threadIdx.x] = local;
  __syncthreads();

  for (k = 1; k < NL_BLOCK; k *= 2)
  {
    if ((threadIdx.x % (2 * k)) == 0 && threadIdx.x + k < NL_BLOCK)
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

// ─── Energy kernel ─────────────────────────────────────────────────────────
// Computes E(x, λ) for each frame b: total bias energy from all parameters.

__global__ void nl_energykernel(struct_nl2024 nl, double *x, double *lambda, double *energy)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int s1, s2, i1, i2, k;
  double q1, q2, E;

  if (b >= nl.B) return;

  double *lam = lambda + nl.nblocks * b;

  k = 0;
  E = 0;
  for (s1 = 0; s1 < nl.nsites; s1++)
  {
    for (s2 = s1; s2 < nl.nsites; s2++)
    {
      if (s1 == s2)
      {
        // Intra-site terms
        for (i1 = nl.block0_d[s1]; i1 < nl.block0_d[s1 + 1]; i1++)
        {
          q1 = lam[i1];
          E += x[k] * q1;  // b term (linear bias)
          k++;
          for (i2 = i1 + 1; i2 < nl.block0_d[s1 + 1]; i2++)
          {
            q2 = lam[i2];
            E += x[k++] * q1 * q2;                            // c (quadratic)
            E += x[k++] * nl_rc_exp(q1, q2, nl.omega_scale);  // x_ij
            E += x[k++] * nl_rc_exp(q2, q1, nl.omega_scale);  // x_ji
            E += x[k++] * nl_rc_sig(q1, q2, nl.chi_offset);   // s_ij
            E += x[k++] * nl_rc_sig(q2, q1, nl.chi_offset);   // s_ji
            if (nl.ntriangle >= 7)
            {
              E += x[k++] * nl_rc_omega2(q1, q2, nl.chi_offset_t);  // t_ij
              E += x[k++] * nl_rc_omega2(q2, q1, nl.chi_offset_t);  // t_ji
            }
            if (nl.ntriangle >= 9)
            {
              E += x[k++] * nl_rc_omega3(q1, q2, nl.chi_offset_u);  // u_ij
              E += x[k++] * nl_rc_omega3(q2, q1, nl.chi_offset_u);  // u_ji
            }
          }
        }
      }
      else if (nl.ms)
      {
        // Inter-site terms
        for (i1 = nl.block0_d[s1]; i1 < nl.block0_d[s1 + 1]; i1++)
        {
          q1 = lam[i1];
          for (i2 = nl.block0_d[s2]; i2 < nl.block0_d[s2 + 1]; i2++)
          {
            q2 = lam[i2];
            E += x[k++] * q1 * q2;  // c (coupling)
            if (nl.ms == 1)
            {
              E += x[k++] * nl_rc_exp(q1, q2, nl.omega_scale);
              E += x[k++] * nl_rc_exp(q2, q1, nl.omega_scale);
              E += x[k++] * nl_rc_sig(q1, q2, nl.chi_offset);
              E += x[k++] * nl_rc_sig(q2, q1, nl.chi_offset);
              if (nl.ntriangle >= 7)
              {
                E += x[k++] * nl_rc_omega2(q1, q2, nl.chi_offset_t);
                E += x[k++] * nl_rc_omega2(q2, q1, nl.chi_offset_t);
              }
              if (nl.ntriangle >= 9)
              {
                E += x[k++] * nl_rc_omega3(q1, q2, nl.chi_offset_u);
                E += x[k++] * nl_rc_omega3(q2, q1, nl.chi_offset_u);
              }
            }
          }
        }
      }
    }
  }
  energy[b] = E;
}

// ─── Gradient kernel ───────────────────────────────────────────────────────
// Computes weighted moment ∂E/∂x[k] contributions via block reduction.

__global__ void nl_weightedenergykernel(struct_nl2024 nl, double sign, double *lambda,
                                        double *weight, double *dEdx)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int s1, s2, i1, i2, k;
  double q1, q2, w, E;
  __shared__ double Eloc[NL_BLOCK];

  if (b < nl.B)
  {
    w = sign * weight[b];
  }
  else
  {
    w = 0;
    q1 = 0;
    q2 = 0;
  }

  double *lam = lambda + nl.nblocks * b;

  k = 0;
  for (s1 = 0; s1 < nl.nsites; s1++)
  {
    for (s2 = s1; s2 < nl.nsites; s2++)
    {
      if (s1 == s2)
      {
        for (i1 = nl.block0_d[s1]; i1 < nl.block0_d[s1 + 1]; i1++)
        {
          if (b < nl.B) q1 = lam[i1];
          E = w * q1;
          nl_reduce(E, Eloc, &dEdx[k]);
          k++;
          for (i2 = i1 + 1; i2 < nl.block0_d[s1 + 1]; i2++)
          {
            if (b < nl.B) q2 = lam[i2];
            E = w * q1 * q2;                                       nl_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * nl_rc_exp(q1, q2, nl.omega_scale);             nl_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * nl_rc_exp(q2, q1, nl.omega_scale);             nl_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * nl_rc_sig(q1, q2, nl.chi_offset);              nl_reduce(E, Eloc, &dEdx[k]); k++;
            E = w * nl_rc_sig(q2, q1, nl.chi_offset);              nl_reduce(E, Eloc, &dEdx[k]); k++;
            if (nl.ntriangle >= 7)
            {
              E = w * nl_rc_omega2(q1, q2, nl.chi_offset_t);       nl_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * nl_rc_omega2(q2, q1, nl.chi_offset_t);       nl_reduce(E, Eloc, &dEdx[k]); k++;
            }
            if (nl.ntriangle >= 9)
            {
              E = w * nl_rc_omega3(q1, q2, nl.chi_offset_u);       nl_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * nl_rc_omega3(q2, q1, nl.chi_offset_u);       nl_reduce(E, Eloc, &dEdx[k]); k++;
            }
          }
        }
      }
      else if (nl.ms)
      {
        for (i1 = nl.block0_d[s1]; i1 < nl.block0_d[s1 + 1]; i1++)
        {
          if (b < nl.B) q1 = lam[i1];
          for (i2 = nl.block0_d[s2]; i2 < nl.block0_d[s2 + 1]; i2++)
          {
            if (b < nl.B) q2 = lam[i2];
            E = w * q1 * q2;                                       nl_reduce(E, Eloc, &dEdx[k]); k++;
            if (nl.ms == 1)
            {
              E = w * nl_rc_exp(q1, q2, nl.omega_scale);           nl_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * nl_rc_exp(q2, q1, nl.omega_scale);           nl_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * nl_rc_sig(q1, q2, nl.chi_offset);            nl_reduce(E, Eloc, &dEdx[k]); k++;
              E = w * nl_rc_sig(q2, q1, nl.chi_offset);            nl_reduce(E, Eloc, &dEdx[k]); k++;
              if (nl.ntriangle >= 7)
              {
                E = w * nl_rc_omega2(q1, q2, nl.chi_offset_t);     nl_reduce(E, Eloc, &dEdx[k]); k++;
                E = w * nl_rc_omega2(q2, q1, nl.chi_offset_t);     nl_reduce(E, Eloc, &dEdx[k]); k++;
              }
              if (nl.ntriangle >= 9)
              {
                E = w * nl_rc_omega3(q1, q2, nl.chi_offset_u);     nl_reduce(E, Eloc, &dEdx[k]); k++;
                E = w * nl_rc_omega3(q2, q1, nl.chi_offset_u);     nl_reduce(E, Eloc, &dEdx[k]); k++;
              }
            }
          }
        }
      }
    }
  }
}

// ─── Dot product kernel ────────────────────────────────────────────────────

__global__ void nl_dotenergykernel(struct_nl2024 nl, double sign,
                                   double *x, double *y, double *z)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  double val;
  __shared__ double loc[NL_BLOCK];

  if (b < nl.B)
    val = sign * x[b] * y[b];
  else
    val = 0;

  nl_reduce(val, loc, z);
}

// ─── Boltzmann weight kernel ───────────────────────────────────────────────
// w_out[b] = w_in[b] * exp(-sign * (E[b] + s * dE/ds[b]) / kT)

__global__ void nl_boltzmannkernel(struct_nl2024 nl, double sign, double *energy,
                                   double s, double *denergyds,
                                   double *inweight, double *outweight, double *Z)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  double E, w;
  __shared__ double Zloc[NL_BLOCK];

  if (b < nl.B)
  {
    w = inweight[b];
    E = energy[b];
    if (s != 0 && denergyds)
      E += s * denergyds[b];
    w *= exp(-sign * E / nl.kT);
    outweight[b] = w;
  }
  else
  {
    w = 0;
  }

  if (Z)
  {
    __syncthreads();
    nl_reduce(w, Zloc, Z);
  }
}

// ─── Regularization kernels ────────────────────────────────────────────────

// L_reg = Σ 0.5 * k * (x - xr)²
__global__ void nl_regularizeLkernel(struct_nl2024 nl)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double deltax, L;
  __shared__ double Lloc[NL_BLOCK];

  if (i < nl.nx)
  {
    deltax = nl.x_d[i] - nl.xr_d[i];
    L = 0.5 * nl.kx_d[i] * deltax * deltax;
  }
  else
  {
    L = 0;
  }

  nl_reduce(L, Lloc, nl.L_d);
}

// ∂L_reg/∂x = k * (x - xr)
__global__ void nl_regularizedLdxkernel(struct_nl2024 nl)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nl.nx)
  {
    nl.dLdx_d[i] = nl.kx_d[i] * (nl.x_d[i] - nl.xr_d[i]);
  }
}

// Gradient accumulation from moment matching
__global__ void nl_gradientlikelihoodkernel(struct_nl2024 nl, double *norm, double *dLdxin)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nl.nbias)
  {
    atomicAdd(&nl.dLdx_d[i], -dLdxin[i] / (norm[0] * nl.kT));
  }
}

// ─── Moment-matching likelihood kernel ─────────────────────────────────────
// L = <E>_data / (Σw * kT) + ln(Z_MC)

__global__ void nl_likelihoodkernel(struct_nl2024 nl, double s, double *L, double *dLds)
{
  if (L)
  {
    atomicAdd(L, nl.Esum_d[0] / (nl.sumensweight_d[0] * nl.kT));
    if (s != 0)
      atomicAdd(L, s * nl.dEdssum_d[0] / (nl.sumensweight_d[0] * nl.kT));
    atomicAdd(L, log(nl.mc_Z_d[0]));
  }
  if (dLds)
  {
    atomicAdd(dLds, nl.dEdssum_d[0] / (nl.sumensweight_d[0] * nl.kT));
    atomicAdd(dLds, -nl.mc_dEdssum_d[0] / (nl.mc_Z_d[0] * nl.kT));
  }
}

// ─── Line search regularization kernel ─────────────────────────────────────

__global__ void nl_regularizelinekernel(struct_nl2024 nl, double s)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double deltax, dxds, L, dLds;
  __shared__ double Lloc[NL_BLOCK];

  if (i < nl.nx)
  {
    dxds = nl.dxds_d[i];
    deltax = nl.x_d[i] + s * dxds - nl.xr_d[i];
    L = 0.5 * nl.kx_d[i] * deltax * deltax;
    dLds = nl.kx_d[i] * deltax * dxds;
  }
  else
  {
    L = 0;
    dLds = 0;
  }

  nl_reduce(L, Lloc, nl.L_d);
  nl_reduce(dLds, Lloc, nl.dLds_d);
}

// ─── Monte Carlo reference distribution ────────────────────────────────────

static double nl_rand_double(void)
{
  return (rand() + 0.5) / (RAND_MAX + 1.0);
}

static void nl_monte_carlo_Z(struct_nl2024 *nl)
{
  int ibeg, iend, Ns;
  int Neq = nl->B / 10;
  int Nmc = nl->B;
  double *theta;
  int s, i, j;
  double b, st, norm;
  double thetaNew, eOld, eNew;

  srand(12345);

  theta = (double *)calloc(nl->nblocks, sizeof(double));

  for (s = 0; s < nl->nsites; s++)
  {
    ibeg = nl->block0[s];
    iend = nl->block0[s + 1];
    Ns = iend - ibeg;

    // Empirical optimal b parameter for acceptance tuning
    b = 1.11 * exp(-0.31 * Ns) + 0.42;

    // Initialize theta
    for (i = ibeg; i < iend; i++)
      theta[i] = M_PI * nl_rand_double();

    // Equilibration
    for (j = 0; j < Neq; j++)
    {
      for (i = ibeg; i < iend; i++)
      {
        thetaNew = theta[i] + b * (nl_rand_double() - 0.5);
        if (thetaNew < 0 || thetaNew > M_PI)
          continue;

        st = sin(thetaNew);
        eNew = nl->fnex * st;
        eOld = nl->fnex * sin(theta[i]);

        if (nl_rand_double() < exp(eNew - eOld) * (st > 0 ? st / sin(theta[i]) : 0))
          theta[i] = thetaNew;
      }
    }

    // Production: generate B samples
    for (j = 0; j < Nmc; j++)
    {
      for (i = ibeg; i < iend; i++)
      {
        thetaNew = theta[i] + b * (nl_rand_double() - 0.5);
        if (thetaNew < 0 || thetaNew > M_PI)
        {
          nl->mc_lambda_h[j * nl->nblocks + i] = exp(nl->fnex * sin(theta[i]));
          continue;
        }

        st = sin(thetaNew);
        eNew = nl->fnex * st;
        eOld = nl->fnex * sin(theta[i]);

        if (nl_rand_double() < exp(eNew - eOld) * (st > 0 ? st / sin(theta[i]) : 0))
          theta[i] = thetaNew;

        nl->mc_lambda_h[j * nl->nblocks + i] = exp(nl->fnex * sin(theta[i]));
      }
    }

    // Normalize lambda per-frame so Σ_i λ_i = 1
    for (j = 0; j < Nmc; j++)
    {
      norm = 0;
      for (i = ibeg; i < iend; i++)
        norm += nl->mc_lambda_h[j * nl->nblocks + i];
      for (i = ibeg; i < iend; i++)
        nl->mc_lambda_h[j * nl->nblocks + i] /= norm;
    }
  }

  // Uniform ensemble weights for MC
  for (j = 0; j < Nmc; j++)
    nl->mc_ensweight_h[j] = 1.0;

  free(theta);
}

// ─── Evaluate L(x) ────────────────────────────────────────────────────────

static void nl_evaluateL(struct_nl2024 *nl)
{
  int grid_B = (nl->B + NL_BLOCK - 1) / NL_BLOCK;
  int grid_nx = (nl->nx + NL_BLOCK - 1) / NL_BLOCK;

  NL_CHECK(cudaMemcpy(nl->x_d, nl->x_h, nl->nx * sizeof(double), cudaMemcpyHostToDevice));
  NL_CHECK(cudaMemset(nl->L_d, 0, sizeof(double)));

  // Regularization
  nl_regularizeLkernel<<<grid_nx, NL_BLOCK>>>(*nl);
  NL_CHECK(cudaGetLastError());

  // Data energy
  nl_energykernel<<<grid_B, NL_BLOCK>>>(*nl, nl->x_d, nl->lambda_d, nl->E_d);
  NL_CHECK(cudaGetLastError());

  // MC energy
  nl_energykernel<<<grid_B, NL_BLOCK>>>(*nl, nl->x_d, nl->mc_lambda_d, nl->mc_E_d);
  NL_CHECK(cudaGetLastError());

  // <E>_data = Σ ensweight * E
  NL_CHECK(cudaMemset(nl->Esum_d, 0, sizeof(double)));
  nl_dotenergykernel<<<grid_B, NL_BLOCK>>>(*nl, 1, nl->ensweight_d, nl->E_d, nl->Esum_d);
  NL_CHECK(cudaGetLastError());

  // Z_MC = Σ exp(E_MC / kT)
  NL_CHECK(cudaMemset(nl->mc_Z_d, 0, sizeof(double)));
  nl_boltzmannkernel<<<grid_B, NL_BLOCK>>>(
      *nl, 1, nl->mc_E_d, 0, NULL, nl->mc_ensweight_d, nl->mc_weight_d, nl->mc_Z_d);
  NL_CHECK(cudaGetLastError());

  // L = <E>/(Σw * kT) + ln(Z_MC) + L_reg
  nl_likelihoodkernel<<<1, 1>>>(*nl, 0, nl->L_d, NULL);
  NL_CHECK(cudaGetLastError());

  NL_CHECK(cudaMemcpy(nl->L_h, nl->L_d, sizeof(double), cudaMemcpyDeviceToHost));

  fprintf(stdout, "New      L=%lg\n", nl->L_h[0]);
}

// ─── Evaluate L along line direction ───────────────────────────────────────

static void nl_evaluateL_line(double s, struct_nl2024 *nl)
{
  int grid_B = (nl->B + NL_BLOCK - 1) / NL_BLOCK;
  int grid_nx = (nl->nx + NL_BLOCK - 1) / NL_BLOCK;

  NL_CHECK(cudaMemset(nl->L_d, 0, sizeof(double)));
  NL_CHECK(cudaMemset(nl->dLds_d, 0, sizeof(double)));

  nl_regularizelinekernel<<<grid_nx, NL_BLOCK>>>(*nl, s);
  NL_CHECK(cudaGetLastError());

  // MC Boltzmann with line offset
  NL_CHECK(cudaMemset(nl->mc_Z_d, 0, sizeof(double)));
  nl_boltzmannkernel<<<grid_B, NL_BLOCK>>>(
      *nl, 1, nl->mc_E_d, s, nl->mc_dEds_d, nl->mc_ensweight_d, nl->mc_weight_d, nl->mc_Z_d);
  NL_CHECK(cudaGetLastError());

  NL_CHECK(cudaMemset(nl->mc_dEdssum_d, 0, sizeof(double)));
  nl_dotenergykernel<<<grid_B, NL_BLOCK>>>(
      *nl, 1, nl->mc_dEds_d, nl->mc_weight_d, nl->mc_dEdssum_d);
  NL_CHECK(cudaGetLastError());

  nl_likelihoodkernel<<<1, 1>>>(*nl, s, nl->L_d, nl->dLds_d);
  NL_CHECK(cudaGetLastError());

  NL_CHECK(cudaMemcpy(nl->L_h, nl->L_d, sizeof(double), cudaMemcpyDeviceToHost));
  NL_CHECK(cudaMemcpy(nl->dLds_h, nl->dLds_d, sizeof(double), cudaMemcpyDeviceToHost));
}

// ─── Evaluate gradient ∂L/∂x ──────────────────────────────────────────────

static void nl_evaluatedLdx(struct_nl2024 *nl)
{
  int grid_B = (nl->B + NL_BLOCK - 1) / NL_BLOCK;
  int grid_nx = (nl->nx + NL_BLOCK - 1) / NL_BLOCK;
  int grid_nb = (nl->nbias + NL_BLOCK - 1) / NL_BLOCK;

  NL_CHECK(cudaMemset(nl->dLdx_d, 0, nl->nx * sizeof(double)));

  // ∂L_reg/∂x
  nl_regularizedLdxkernel<<<grid_nx, NL_BLOCK>>>(*nl);
  NL_CHECK(cudaGetLastError());

  // MC gradient moments
  NL_CHECK(cudaMemset(nl->mc_moments_d, 0, nl->nbias * sizeof(double)));
  nl_weightedenergykernel<<<grid_B, NL_BLOCK>>>(
      *nl, 1, nl->mc_lambda_d, nl->mc_weight_d, nl->mc_moments_d);
  NL_CHECK(cudaGetLastError());

  // Data gradient: -<∂E/∂x>_data / (Σw * kT)
  nl_gradientlikelihoodkernel<<<grid_nb, NL_BLOCK>>>(
      *nl, nl->sumensweight_d, nl->moments_d);
  NL_CHECK(cudaGetLastError());

  // MC gradient: +<∂E/∂x>_MC / (Z_MC * kT)
  nl_gradientlikelihoodkernel<<<grid_nb, NL_BLOCK>>>(
      *nl, nl->mc_Z_d, nl->mc_moments_d);
  NL_CHECK(cudaGetLastError());

  NL_CHECK(cudaMemcpy(nl->dLdx_h, nl->dLdx_d, nl->nx * sizeof(double), cudaMemcpyDeviceToHost));
}

// ─── L-BFGS memory management ─────────────────────────────────────────────

static void nl_resetHinv(struct_nl2024 *nl)
{
  for (int i = 0; i < nl->nx; i++)
  {
    nl->x0_h[i] = nl->x_h[i];
    nl->dLdx0_h[i] = nl->dLdx_h[i];
  }
}

static void nl_updateHinv(struct_nl2024 *nl)
{
  int i, j;

  if (nl->Nmem < nl->Nmemax)
    nl->Nmem++;

  // Shift history (newest at index 0)
  for (i = nl->Nmem - 1; i > 0; i--)
  {
    for (j = 0; j < nl->nx; j++)
    {
      nl->d_x[i * nl->nx + j] = nl->d_x[(i - 1) * nl->nx + j];
      nl->d_dLdx[i * nl->nx + j] = nl->d_dLdx[(i - 1) * nl->nx + j];
    }
    nl->rho[i] = nl->rho[i - 1];
  }

  // New entry: s = Δx, y = ΔdL/dx, ρ = 1/(s·y)
  nl->rho[0] = 0;
  for (i = 0; i < nl->nx; i++)
  {
    nl->d_x[i] = nl->x_h[i] - nl->x0_h[i];
    nl->d_dLdx[i] = nl->dLdx_h[i] - nl->dLdx0_h[i];
    nl->rho[0] += nl->d_x[i] * nl->d_dLdx[i];
  }
  nl->rho[0] = 1.0 / nl->rho[0];

  // Store current as previous
  for (i = 0; i < nl->nx; i++)
  {
    nl->x0_h[i] = nl->x_h[i];
    nl->dLdx0_h[i] = nl->dLdx_h[i];
  }
}

// ─── L-BFGS two-loop recursion → search direction ─────────────────────────

static void nl_projectHinv(struct_nl2024 *nl)
{
  int i, j;
  int grid_B = (nl->B + NL_BLOCK - 1) / NL_BLOCK;

  // h = ∇L
  for (i = 0; i < nl->nx; i++)
    nl->hi_h[i] = nl->dLdx_h[i];

  // Forward loop: α[i] = ρ[i] (s[i]·h); h -= α[i] y[i]
  for (i = 0; i < nl->Nmem; i++)
  {
    nl->alpha[i] = 0;
    for (j = 0; j < nl->nx; j++)
      nl->alpha[i] += nl->d_x[i * nl->nx + j] * nl->hi_h[j];
    nl->alpha[i] *= nl->rho[i];
    for (j = 0; j < nl->nx; j++)
      nl->hi_h[j] -= nl->alpha[i] * nl->d_dLdx[i * nl->nx + j];
  }

  // Backward loop: β[i] = ρ[i] (y[i]·h); h += (α[i] - β[i]) s[i]
  for (i = nl->Nmem - 1; i >= 0; i--)
  {
    nl->beta[i] = 0;
    for (j = 0; j < nl->nx; j++)
      nl->beta[i] += nl->d_dLdx[i * nl->nx + j] * nl->hi_h[j];
    nl->beta[i] *= nl->rho[i];
    for (j = 0; j < nl->nx; j++)
      nl->hi_h[j] += (nl->alpha[i] - nl->beta[i]) * nl->d_x[i * nl->nx + j];
  }

  // Negate for descent
  for (i = 0; i < nl->nx; i++)
    nl->hi_h[i] *= -1;

  // Upload search direction and compute directional derivatives
  NL_CHECK(cudaMemcpy(nl->dxds_d, nl->hi_h, nl->nx * sizeof(double), cudaMemcpyHostToDevice));

  nl_energykernel<<<grid_B, NL_BLOCK>>>(*nl, nl->dxds_d, nl->lambda_d, nl->dEds_d);
  NL_CHECK(cudaGetLastError());

  nl_energykernel<<<grid_B, NL_BLOCK>>>(*nl, nl->dxds_d, nl->mc_lambda_d, nl->mc_dEds_d);
  NL_CHECK(cudaGetLastError());

  NL_CHECK(cudaMemset(nl->dEdssum_d, 0, sizeof(double)));
  nl_dotenergykernel<<<grid_B, NL_BLOCK>>>(*nl, 1, nl->ensweight_d, nl->dEds_d, nl->dEdssum_d);
  NL_CHECK(cudaGetLastError());
}

// ─── Line search: bracket + quadratic interpolation ────────────────────────

static void nl_update_line(int step, struct_nl2024 *nl)
{
  int i;
  double a, b, c, s;
  double s1, s2, s3;
  double L1, L2, L3;
  double dLds1, dLds2, dLds3;
  double L0;

  for (i = 0; i < nl->nx; i++)
  {
    nl->x0_h[i] = nl->x_h[i];
    nl->dLdx0_h[i] = nl->dLdx_h[i];
  }

  L0 = nl->L_h[0];

  // Evaluate at s=0
  s1 = 0.0;
  nl_evaluateL_line(s1, nl);
  L1 = nl->L_h[0];
  dLds1 = nl->dLds_h[0];

  if (dLds1 > 0)
  {
    fprintf(stdout, "Error, search direction pointing uphill - halting\n");
    nl->done = 1;
    return;
  }

  // Evaluate at s=1, expand until we bracket
  s3 = 1.0;
  nl_evaluateL_line(s3, nl);
  L3 = nl->L_h[0];
  dLds3 = nl->dLds_h[0];

  while (dLds3 < 0 && s3 < 1e+8)
  {
    fprintf(stdout, "Seek %4d s=%lg %lg\n          L=%lg %lg\n       dLds=%lg %lg\n",
            step, s1, s3, L1, L3, dLds1, dLds3);
    s2 = s1 - dLds1 * (s3 - s1) / (dLds3 - dLds1);
    s3 = ((1.5 * s2 > 8 * s3 || 1.5 * s2 <= 0) ? 8 * s3 : 1.5 * s2);
    nl_evaluateL_line(s3, nl);
    L3 = nl->L_h[0];
    dLds3 = nl->dLds_h[0];
  }

  // Handle overshooting
  while (!isfinite(dLds3) && s3 > 1e-8)
  {
    fprintf(stdout, "Warning, overshot bound\n");
    s3 = 0.95 * s3;
    nl_evaluateL_line(s3, nl);
    L3 = nl->L_h[0];
    dLds3 = nl->dLds_h[0];
  }

  if (!(dLds3 > 0))
  {
    fprintf(stdout, "Warning: Step %4d unsuccessful, halting minimization\n", step);
    nl->done = 1;
    return;
  }

  // Secant interpolation for initial s2
  s2 = s1 - dLds1 * (s3 - s1) / (dLds3 - dLds1);
  nl_evaluateL_line(s2, nl);
  L2 = nl->L_h[0];
  dLds2 = nl->dLds_h[0];

  fprintf(stdout, "Step %4d s=%lg %lg %lg\n          L=%lg %lg %lg\n       dLds=%lg %lg %lg\n",
          step, s1, s2, s3, L1, L2, L3, dLds1, dLds2, dLds3);

  // Quadratic refinement (up to 15 iterations)
  for (i = 0; i < 15; i++)
  {
    if ((s2 - s1) / s2 < 5e-7 || (s3 - s2) / s2 < 5e-7 || dLds2 == 0)
      break;

    // Quadratic interpolation through (s1,dLds1), (s2,dLds2), (s3,dLds3)
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
      fprintf(stdout, "Warning, fell back on linear interpolation\n");
      s2 = s1 - dLds1 * (s3 - s1) / (dLds3 - dLds1);
    }

    nl_evaluateL_line(s2, nl);
    L2 = nl->L_h[0];
    dLds2 = nl->dLds_h[0];

    fprintf(stdout, "Step %4d s=%lg %lg %lg\n          L=%lg %lg %lg\n       dLds=%lg %lg %lg\n",
            step, s1, s2, s3, L1, L2, L3, dLds1, dLds2, dLds3);
  }

  // Apply optimal step
  double stepLength2 = 0;

  for (i = 0; i < nl->nx; i++)
  {
    nl->x_h[i] = nl->x0_h[i] + s2 * nl->hi_h[i];
    stepLength2 += (s2 * nl->hi_h[i]) * (s2 * nl->hi_h[i]);
  }

  fprintf(stdout, "Step %4d L=%24.16lf -> L2=%24.16lf, dL=%lg, step=%lg\n",
          step, L0, L2, L2 - L0, sqrt(stepLength2));

  // Convergence check
  if (sqrt(stepLength2) < 5e-7)
    nl->done = 1;
  if (sqrt(stepLength2 / nl->nx) < nl->criteria)
  {
    nl->doneCount++;
    if (nl->doneCount == 2)
      nl->done = 1;
  }
  else
  {
    nl->doneCount = 0;
  }
}

// ─── Single optimization iteration ────────────────────────────────────────

static void nl_iterate(int step, struct_nl2024 *nl)
{
  nl_evaluateL(nl);
  nl_evaluatedLdx(nl);

  if (step == 0)
    nl_resetHinv(nl);
  else
    nl_updateHinv(nl);

  nl_projectHinv(nl);
  nl_update_line(step, nl);
}

// ─── Main optimization loop ───────────────────────────────────────────────

static void nl_run(struct_nl2024 *nl)
{
  int s;
  double sum;

  // Compute sum of ensemble weights
  sum = 0;
  for (int i = 0; i < nl->B; i++)
    sum += nl->ensweight_h[i];
  NL_CHECK(cudaMemcpy(nl->sumensweight_d, &sum, sizeof(double), cudaMemcpyHostToDevice));

  // Compute initial data moments: <∂E/∂x>_data
  int grid_B = (nl->B + NL_BLOCK - 1) / NL_BLOCK;
  NL_CHECK(cudaMemset(nl->moments_d, 0, nl->nbias * sizeof(double)));
  nl_weightedenergykernel<<<grid_B, NL_BLOCK>>>(
      *nl, -1, nl->lambda_d, nl->ensweight_d, nl->moments_d);
  NL_CHECK(cudaGetLastError());

  nl->done = 0;
  nl->doneCount = 0;

  for (s = 0; s < nl->max_iter; s++)
  {
    nl_iterate(s, nl);
    if (nl->done)
      break;
  }

  fprintf(stdout, "nonlinear optimization completed after %d iterations\n", s);
}

// ─── Write output and cleanup ──────────────────────────────────────────────

static void nl_finish(struct_nl2024 *nl)
{
  FILE *fp;

  fp = fopen("OUT.dat", "w");
  for (int i = 0; i < nl->nx; i++)
    fprintf(fp, " %lg", nl->x_h[i]);
  fclose(fp);

  fprintf(stdout, "nonlinear results written to OUT.dat\n");

  // Free host memory
  free(nl->nsubs);
  free(nl->block0);
  free(nl->lambda_h);
  free(nl->ensweight_h);
  free(nl->mc_lambda_h);
  free(nl->mc_ensweight_h);
  free(nl->kx_h);
  free(nl->xr_h);
  free(nl->L_h);
  free(nl->dLds_h);
  free(nl->x_h);
  free(nl->dLdx_h);
  free(nl->d_x);
  free(nl->d_dLdx);
  free(nl->rho);
  free(nl->alpha);
  free(nl->beta);
  free(nl->hi_h);
  free(nl->x0_h);
  free(nl->dLdx0_h);

  // Free device memory
  cudaFree(nl->block0_d);
  cudaFree(nl->lambda_d);
  cudaFree(nl->mc_lambda_d);
  cudaFree(nl->ensweight_d);
  cudaFree(nl->mc_ensweight_d);
  cudaFree(nl->kx_d);
  cudaFree(nl->xr_d);
  cudaFree(nl->L_d);
  cudaFree(nl->dLds_d);
  cudaFree(nl->dLdx_d);
  cudaFree(nl->E_d);
  cudaFree(nl->dEds_d);
  cudaFree(nl->mc_E_d);
  cudaFree(nl->mc_dEds_d);
  cudaFree(nl->weight_d);
  cudaFree(nl->mc_weight_d);
  cudaFree(nl->x_d);
  cudaFree(nl->dxds_d);
  cudaFree(nl->Z_d);
  cudaFree(nl->mc_Z_d);
  cudaFree(nl->Esum_d);
  cudaFree(nl->dEdssum_d);
  cudaFree(nl->mc_dEdssum_d);
  cudaFree(nl->moments_d);
  cudaFree(nl->mc_moments_d);
  cudaFree(nl->sumensweight_d);

  free(nl);
}

// ─── Setup: allocate and initialize solver ─────────────────────────────────

static struct_nl2024 *nl_setup(
    int nf, double temp, int ms, int msprof,
    int *nsubs_in, int nsites, double fnex, int ntriangle,
    double *lambda_flat, double *ensweight_flat, int n_frames,
    double *x_prev_flat, double *s_prev_flat, int nblocks_sq)
{
  struct_nl2024 *nl;
  int i, j, k, si, sj;
  double kp, k0;
  double kBoltz = 1.987204e-3;  // kcal/(mol·K)

  nl = (struct_nl2024 *)calloc(1, sizeof(struct_nl2024));
  if (!nl) return NULL;

  // System topology
  nl->nsites = nsites;
  nl->nsubs = (int *)calloc(nsites, sizeof(int));
  nl->block0 = (int *)calloc(nsites + 1, sizeof(int));

  nl->nblocks = 0;
  nl->block0[0] = 0;
  for (i = 0; i < nsites; i++)
  {
    nl->nsubs[i] = nsubs_in[i];
    nl->nblocks += nsubs_in[i];
    nl->block0[i + 1] = nl->nblocks;
  }

  NL_CHECK(cudaMalloc(&nl->block0_d, (nsites + 1) * sizeof(int)));
  NL_CHECK(cudaMemcpy(nl->block0_d, nl->block0, (nsites + 1) * sizeof(int), cudaMemcpyHostToDevice));

  // Temperature
  nl->kT = kBoltz * temp;
  nl->B = n_frames;
  nl->ms = ms;
  nl->msprof = msprof;
  nl->fnex = fnex;
  nl->ntriangle = ntriangle;

  // Copy lambda trajectories
  nl->lambda_h = (double *)calloc(nl->B * nl->nblocks, sizeof(double));
  nl->ensweight_h = (double *)calloc(nl->B, sizeof(double));
  memcpy(nl->lambda_h, lambda_flat, nl->B * nl->nblocks * sizeof(double));
  memcpy(nl->ensweight_h, ensweight_flat, nl->B * sizeof(double));

  // Generate MC reference
  nl->mc_lambda_h = (double *)calloc(nl->B * nl->nblocks, sizeof(double));
  nl->mc_ensweight_h = (double *)calloc(nl->B, sizeof(double));
  nl_monte_carlo_Z(nl);

  // GPU copies
  NL_CHECK(cudaMalloc(&nl->lambda_d, nl->B * nl->nblocks * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_lambda_d, nl->B * nl->nblocks * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->ensweight_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_ensweight_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMemcpy(nl->lambda_d, nl->lambda_h, nl->B * nl->nblocks * sizeof(double), cudaMemcpyHostToDevice));
  NL_CHECK(cudaMemcpy(nl->mc_lambda_d, nl->mc_lambda_h, nl->B * nl->nblocks * sizeof(double), cudaMemcpyHostToDevice));
  NL_CHECK(cudaMemcpy(nl->ensweight_d, nl->ensweight_h, nl->B * sizeof(double), cudaMemcpyHostToDevice));
  NL_CHECK(cudaMemcpy(nl->mc_ensweight_d, nl->mc_ensweight_h, nl->B * sizeof(double), cudaMemcpyHostToDevice));

  // Count bias parameters
  nl->nbias = 0;
  for (i = 0; i < nsites; i++)
  {
    for (j = i; j < nsites; j++)
    {
      if (i == j)
        nl->nbias += nl->nsubs[i] + (ntriangle * nl->nsubs[i] * (nl->nsubs[i] - 1)) / 2;
      else if (ms == 1)
        nl->nbias += ntriangle * nl->nsubs[i] * nl->nsubs[j];
      else if (ms == 2)
        nl->nbias += nl->nsubs[i] * nl->nsubs[j];
    }
  }
  nl->nx = nl->nbias;

  // Regularization setup: kx (strength) and xr (reference)
  kp = 1.0 / (nl->kT * nl->kT);
  k0 = kp / 400;

  nl->kx_h = (double *)calloc(nl->nx, sizeof(double));
  nl->xr_h = (double *)calloc(nl->nx, sizeof(double));

  // Load previous x/s as regularization reference for inter-site terms
  double *xr_x = NULL, *xr_s = NULL;
  if (ms == 1 && x_prev_flat && s_prev_flat && nblocks_sq > 0)
  {
    xr_x = (double *)calloc(nl->nblocks * nl->nblocks, sizeof(double));
    xr_s = (double *)calloc(nl->nblocks * nl->nblocks, sizeof(double));
    int copy_size = nl->nblocks * nl->nblocks;
    if (nblocks_sq < copy_size) copy_size = nblocks_sq;
    memcpy(xr_x, x_prev_flat, copy_size * sizeof(double));
    memcpy(xr_s, s_prev_flat, copy_size * sizeof(double));
  }

  // Set per-parameter regularization constants
  k = 0;
  for (si = 0; si < nsites; si++)
  {
    for (sj = si; sj < nsites; sj++)
    {
      if (si == sj)
      {
        for (i = 0; i < nl->nsubs[si]; i++)
        {
          nl->kx_h[k++] = k0 / 4;    // b
          for (j = i + 1; j < nl->nsubs[sj]; j++)
          {
            nl->kx_h[k++] = k0 / 64;  // c
            nl->kx_h[k++] = k0 / 4;   // x
            nl->kx_h[k++] = k0 / 4;   // x
            nl->kx_h[k++] = k0 / 1;   // s
            nl->kx_h[k++] = k0 / 1;   // s
            if (ntriangle >= 7)
            {
              nl->kx_h[k++] = k0 / 1; // t
              nl->kx_h[k++] = k0 / 1; // t
            }
            if (ntriangle >= 9)
            {
              nl->kx_h[k++] = k0 / 1; // u
              nl->kx_h[k++] = k0 / 1; // u
            }
          }
        }
      }
      else if (ms)
      {
        for (i = 0; i < nl->nsubs[si]; i++)
        {
          for (j = 0; j < nl->nsubs[sj]; j++)
          {
            nl->kx_h[k++] = k0 / 4;       // c
            if (ms == 1)
            {
              if (xr_x)
                nl->xr_h[k] = xr_x[(nl->block0[si] + i) * nl->nblocks + nl->block0[sj] + j];
              nl->kx_h[k++] = k0 / 0.25;   // x
              if (xr_x)
                nl->xr_h[k] = xr_x[(nl->block0[sj] + j) * nl->nblocks + nl->block0[si] + i];
              nl->kx_h[k++] = k0 / 0.25;   // x
              if (xr_s)
                nl->xr_h[k] = xr_s[(nl->block0[si] + i) * nl->nblocks + nl->block0[sj] + j];
              nl->kx_h[k++] = k0 / 0.25;   // s
              if (xr_s)
                nl->xr_h[k] = xr_s[(nl->block0[sj] + j) * nl->nblocks + nl->block0[si] + i];
              nl->kx_h[k++] = k0 / 0.25;   // s
              if (ntriangle >= 7)
              {
                nl->kx_h[k++] = k0 / 0.25; // t
                nl->kx_h[k++] = k0 / 0.25; // t
              }
              if (ntriangle >= 9)
              {
                nl->kx_h[k++] = k0 / 0.25; // u
                nl->kx_h[k++] = k0 / 0.25; // u
              }
            }
          }
        }
      }
    }
  }

  if (xr_x) free(xr_x);
  if (xr_s) free(xr_s);

  NL_CHECK(cudaMalloc(&nl->kx_d, nl->nx * sizeof(double)));
  NL_CHECK(cudaMemcpy(nl->kx_d, nl->kx_h, nl->nx * sizeof(double), cudaMemcpyHostToDevice));
  NL_CHECK(cudaMalloc(&nl->xr_d, nl->nx * sizeof(double)));
  NL_CHECK(cudaMemcpy(nl->xr_d, nl->xr_h, nl->nx * sizeof(double), cudaMemcpyHostToDevice));

  // Allocate calculation arrays
  nl->L_h = (double *)calloc(1, sizeof(double));
  nl->dLds_h = (double *)calloc(1, sizeof(double));
  NL_CHECK(cudaMalloc(&nl->L_d, sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->dLds_d, sizeof(double)));

  nl->x_h = (double *)calloc(nl->nx, sizeof(double));
  nl->dLdx_h = (double *)calloc(nl->nx, sizeof(double));
  NL_CHECK(cudaMalloc(&nl->dLdx_d, nl->nx * sizeof(double)));

  // L-BFGS memory
  nl->Nmemax = 50;
  nl->Nmem = 0;
  nl->d_x = (double *)calloc(nl->nx * nl->Nmemax, sizeof(double));
  nl->d_dLdx = (double *)calloc(nl->nx * nl->Nmemax, sizeof(double));
  nl->rho = (double *)calloc(nl->Nmemax, sizeof(double));
  nl->alpha = (double *)calloc(nl->Nmemax, sizeof(double));
  nl->beta = (double *)calloc(nl->Nmemax, sizeof(double));
  nl->hi_h = (double *)calloc(nl->nx, sizeof(double));
  nl->x0_h = (double *)calloc(nl->nx, sizeof(double));
  nl->dLdx0_h = (double *)calloc(nl->nx, sizeof(double));

  // GPU intermediates
  NL_CHECK(cudaMalloc(&nl->E_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->dEds_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_E_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_dEds_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->weight_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_weight_d, nl->B * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->x_d, nl->nx * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->dxds_d, nl->nx * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->Z_d, sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_Z_d, sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->Esum_d, sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->dEdssum_d, sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_dEdssum_d, sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->moments_d, nl->nbias * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->mc_moments_d, nl->nbias * sizeof(double)));
  NL_CHECK(cudaMalloc(&nl->sumensweight_d, sizeof(double)));

  // Convergence defaults
  nl->criteria = 1.25e-3;
  nl->max_iter = 250;
  nl->done = 0;
  nl->doneCount = 0;

  return nl;
}

// ─── Public entry point ────────────────────────────────────────────────────

extern "C" int nonlinear_from_memory(
    int nf, double temp, int ms, int msprof, int max_iter, double tolerance,
    int *nsubs, int nsites,
    double fnex, double chi_offset, double omega_scale,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double *lambda_flat, double *ensweight_flat, int n_frames,
    double *x_prev_flat, double *s_prev_flat, int nblocks_sq)
{
  fprintf(stdout, "nonlinear: L-BFGS bias parameter optimizer\n");
  fprintf(stdout, "  nf=%d, temp=%.2f, ms=%d, msprof=%d\n", nf, temp, ms, msprof);
  fprintf(stdout, "  max_iter=%d, tolerance=%g\n", max_iter, tolerance);
  fprintf(stdout, "  fnex=%.4f, chi_offset=%.6f, omega_scale=%.6f\n", fnex, chi_offset, omega_scale);
  fprintf(stdout, "  chi_offset_t=%.6f, chi_offset_u=%.6f, ntriangle=%d\n",
          chi_offset_t, chi_offset_u, ntriangle);
  fprintf(stdout, "  n_frames=%d, nblocks_sq=%d\n", n_frames, nblocks_sq);

  nl_validate_gpu();

  struct_nl2024 *nl = nl_setup(
      nf, temp, ms, msprof,
      nsubs, nsites, fnex, ntriangle,
      lambda_flat, ensweight_flat, n_frames,
      x_prev_flat, s_prev_flat, nblocks_sq);

  if (!nl)
  {
    fprintf(stderr, "nonlinear: Failed to setup solver\n");
    return -1;
  }

  // Set bias constants
  nl->chi_offset = chi_offset;
  nl->omega_scale = omega_scale;
  nl->chi_offset_t = chi_offset_t;
  nl->chi_offset_u = chi_offset_u;

  if (max_iter > 0)
    nl->max_iter = max_iter;
  if (tolerance > 0)
    nl->criteria = tolerance;

  NL_CHECK(cudaDeviceSynchronize());

  nl_run(nl);

  NL_CHECK(cudaDeviceSynchronize());

  nl_finish(nl);

  NL_CHECK(cudaDeviceSynchronize());
  NL_CHECK(cudaDeviceReset());

  return 0;
}
