/**
 * nonlinear — L-BFGS nonlinear solver for ALF bias optimization.
 *
 * Standalone CUDA solver that combines moment-matching and regularization
 * loss terms. Replaces the LMALF solver with a cleaner interface and
 * separate compilation unit.
 *
 * Entry point: nonlinear_from_memory() accepts host pointers,
 * runs L-BFGS optimization on GPU, writes OUT.dat with optimized coefficients.
 */

#ifndef NONLINEAR2024_H
#define NONLINEAR2024_H

// Block size for CUDA kernels (must be power of 2 for reduction)
#define NL_BLOCK 512

typedef struct struct_nl2024
{
  // System topology
  int nblocks;
  int nsites;
  int *nsubs;     // [nsites] — subsites per site
  int *block0;    // [nsites+1] — cumulative block offsets (host)
  int *block0_d;  // device copy

  // Trajectory data
  int B;          // Number of frames
  double kT;      // Boltzmann constant * temperature

  // Multisite flags
  int ms;         // 0=none, 1=full c/x/s, 2=c-only
  int msprof;     // Multisite profiles flag

  // Parameter counts
  int nbias;      // Number of bias parameters
  int nx;         // Total parameters = nbias

  // Lambda trajectories (simulation data)
  double *lambda_h;       // [B × nblocks] host
  double *lambda_d;       // [B × nblocks] device
  double *ensweight_h;    // [B] host
  double *ensweight_d;    // [B] device

  // Monte Carlo reference (unbiased canonical ensemble)
  double *mc_lambda_h;    // [B × nblocks] host
  double *mc_lambda_d;    // [B × nblocks] device
  double *mc_ensweight_h; // [B] host
  double *mc_ensweight_d; // [B] device

  // Regularization
  double *kx_h;           // [nx] host — regularization strengths
  double *kx_d;           // [nx] device
  double *xr_h;           // [nx] host — reference values
  double *xr_d;           // [nx] device

  // Energy intermediates (GPU)
  double *E_d;            // [B] bias energy per frame
  double *dEds_d;         // [B] directional derivative of E
  double *mc_E_d;         // [B] MC reference energies
  double *mc_dEds_d;      // [B] MC directional derivatives
  double *weight_d;       // [B] Boltzmann weights
  double *mc_weight_d;    // [B] MC Boltzmann weights

  // Accumulation scalars (GPU)
  double *Z_d;            // Partition function
  double *mc_Z_d;         // MC partition function
  double *Esum_d;         // Σ ensweight * E
  double *dEdssum_d;      // Σ ensweight * dE/ds
  double *mc_dEdssum_d;   // Σ mc_weight * dE/ds
  double *moments_d;      // [nbias] data moments
  double *mc_moments_d;   // [nbias] MC moments
  double *sumensweight_d; // Σ ensweight

  // L-BFGS optimization state
  double *L_h;            // [1] host — loss value
  double *L_d;            // [1] device
  double *dLds_h;         // [1] host — directional derivative
  double *dLds_d;         // [1] device
  double *x_h;            // [nx] host — current parameters
  double *x_d;            // [nx] device
  double *dLdx_h;         // [nx] host — gradient
  double *dLdx_d;         // [nx] device
  double *dxds_d;         // [nx] device — search direction
  double *x0_h;           // [nx] previous x (for L-BFGS update)
  double *dLdx0_h;        // [nx] previous gradient
  double *hi_h;           // [nx] search direction (host)

  // L-BFGS memory (limited-memory BFGS, 50-vector history)
  int Nmem;               // Current history entries
  int Nmemax;             // Maximum history (50)
  double *d_x;            // [Nmemax × nx] delta parameter history
  double *d_dLdx;         // [Nmemax × nx] delta gradient history
  double *rho;            // [Nmemax] curvature scaling factors
  double *alpha;          // [Nmemax] forward loop coefficients
  double *beta;           // [Nmemax] backward loop coefficients

  // Convergence control
  double criteria;        // Convergence tolerance (default 1.25e-3)
  int doneCount;          // Consecutive convergence count
  int done;               // Optimization complete flag
  int max_iter;           // Maximum iterations (default 250)

  // Bias constants (derived from FNEX)
  double fnex;            // FNEX softmax parameter
  double chi_offset;      // s-term sigmoid offset (4*exp(-FNEX))
  double omega_scale;     // x-term reciprocal decay (1/FNEX)
  double chi_offset_t;    // t-term sigmoid offset
  double chi_offset_u;    // u-term Hill sigmoid offset
  int ntriangle;          // Pair params per unique pair: 5, 7, or 9

  // Paths
  char g_imp_path[256];   // Path to G_imp directory (unused, reserved)
} struct_nl2024;

// Entry point: accepts pre-packed host pointers, runs L-BFGS, writes OUT.dat.
#ifdef __cplusplus
extern "C" {
#endif

int nonlinear_from_memory(
    int nf, double temp, int ms, int msprof, int max_iter, double tolerance,
    int *nsubs, int nsites,
    double fnex, double chi_offset, double omega_scale,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double *lambda_flat, double *ensweight_flat, int n_frames,
    double *x_prev_flat, double *s_prev_flat, int nblocks_sq);

#ifdef __cplusplus
}
#endif

#endif /* NONLINEAR2024_H */
