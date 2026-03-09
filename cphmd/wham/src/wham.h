/**
 * WHAM (Weighted Histogram Analysis Method) Header
 *
 * Data structures for GPU-accelerated WHAM analysis with G-shift support.
 * Bundled with cphmd from wham_3 variant.
 */

typedef struct struct_hist
{
  double max;
  double min;
  int N;
  double dx;
} struct_hist;

// G_imp file cache: avoids re-reading the same file on every bin_all() call.
// All blocks within a site share the same G_imp file (per-dimension naming),
// so caching eliminates O(profiles_per_site) redundant file reads.
#define GIMP_CACHE_MAX 32
typedef struct struct_gimp_entry
{
  char filename[256];
  double *data;
  int size;
} struct_gimp_entry;

typedef struct struct_gimp_cache
{
  struct_gimp_entry entries[GIMP_CACHE_MAX];
  int count;
} struct_gimp_cache;

// Profile descriptor for bin_all(): precomputed lookup replaces decrement-counter loop
typedef struct profile_desc
{
  int ptype;          // 0=1D(phi), 1=12(psi), 2=2D(chi/omega), 3=SS(cross-site)
  int i1, i2;         // block indices
  double wnorm;       // normalization weight for C/V matrix
  int B_N;            // number of histogram bins (1D or 2D)
  char gimp_fn[256];  // precomputed G_imp filename
} profile_desc;

// Reaction coordinate descriptor for reactioncoord_all(): direct kernel dispatch
typedef struct param_desc
{
  int type;    // 0=phi, 1=psi, 2=chi, 3=omega, 4=omega2(t), 5=omega3(u)
  int j1, j2;  // block indices
} param_desc;

typedef struct struct_data
{
  int NL;
  int NF;
  int ms;     // whether to calculate multisite interaction parameters
  int msprof; // whether to calculate intersite profiles
  double *T_h;
  double *beta_h;
  double *beta_d;
  double beta_t;
  struct_hist B[2];
  struct_hist B2d[3];
  int ND;
  int NDmax;
  int Nsim;
  int Ndim;
  double *D_h; // data
  double *D_d;
  int *i_h; // simulation index
  int *i_d;
  double *lnw_h; // simulation weight
  double *lnw_d;
  double *lnDenom_h;
  double *lnDenom_d;
  int *n_h;
  int *n_d;
  double *f_h;
  double *f_d;
  // invf_h/invf_d removed: getf writes f_d directly
  double *lnZ_h;
  double *lnZ_d;
  double **dlnZ_hN;
  double *dlnZ_dN;
  double *dlnZ_d;
  double *Gimp_h;
  double *Gimp_d;
  double *CC_h;
  double *CC_d;
  double *gshift_h; /* host  [NF × Nblocks] */
  double *gshift_d; /* device[NF × Nblocks] */
  double *C_h;
  double *C_d;
  double *CV_h;
  double *CV_d;
  int Nblocks;
  int Nsites;
  int *Nsubs;
  int *block0;
  int iN;
  int jN;
  int current_block_idx; // Current block index for gshift lookups
  int use_gshift;        // 0 = disabled (legacy), 1 = enabled (apply G_imp shifts)
  char g_imp_path[256];  // Path to G_imp directory
  double chi_offset;     // s-term sigmoid offset (derived from FNEX: 4*exp(-FNEX))
  double omega_scale;      // x-term reciprocal decay (derived from FNEX: 1/FNEX)
  double chi_offset_t;   // t-term sigmoid offset (independent of FNEX)
  double chi_offset_u;   // u-term Hill sigmoid offset (independent of FNEX)
  int ntriangle;         // pair params per unique pair: 5(bcxs), 7(+t), 9(+tu)
  double cutlsum;        // G12 conditional threshold (λ_i + λ_j > cutlsum)
  double endpoint_weight; // Phase-dependent endpoint bin weight (default 100.0)
  double endpoint_decay;  // Exponential ramp decay rate (default 2.0)
  struct_gimp_cache *gimp_cache;  // Cached G_imp file data (host-only)
  profile_desc *profiles;         // Precomputed profile descriptors [iN] (host-only)
  param_desc *params;             // Precomputed param descriptors [jN] (host-only)
} struct_data;

/**
 * LMALF (Likelihood Maximization ALF) Data Structure
 *
 * Structure for L-BFGS optimization of bias parameters using maximum likelihood.
 * Alternative to WHAM's iterative histogram reweighting approach.
 *
 * Key differences from WHAM:
 * - Direct optimization of bias parameters via quasi-Newton (L-BFGS)
 * - Profile-based fitting instead of histogram reweighting
 * - Built-in L2 regularization to prevent overfitting
 */
typedef struct struct_lmalf
{
  // System configuration
  int nblocks;       // Total number of blocks
  int nsites;        // Number of titratable sites
  int *nsubs;        // Subsites per site [nsites]
  int *block0;       // Block index offsets [nsites+1]
  int *block0_d;     // Device copy

  // Frames and temperature
  int B;             // Number of trajectory frames
  double kT;         // Boltzmann constant * temperature

  // Multisite parameters
  int ms;            // Multisite coupling flag (0, 1, or 2)
  int msprof;        // Multisite profiles flag

  // Counts
  int nbias;         // Number of bias parameters
  int nprof;         // Number of profile types
  int nx;            // Total parameters to optimize (nbias)

  // Lambda trajectories
  double *lambda_h;     // Host: lambda values [B * nblocks]
  double *lambda_d;     // Device: lambda values
  double *ensweight_h;  // Host: ensemble weights [B]
  double *ensweight_d;  // Device: ensemble weights

  // Monte Carlo reference (for partition function)
  double *mc_lambda_h;     // Host: MC reference lambda [B * nblocks]
  double *mc_lambda_d;     // Device: MC reference lambda
  double *mc_ensweight_h;  // Host: MC ensemble weights [B]
  double *mc_ensweight_d;  // Device: MC ensemble weights

  // Regularization
  double *kx_h;       // Host: regularization constants [nx]
  double *kx_d;       // Device: regularization constants
  double *xr_h;       // Host: regularization reference values [nx]
  double *xr_d;       // Device: regularization reference values
  double *kprofile_h; // Host: profile regularization [LMALF_NBINS * nprof]
  double *kprofile_d; // Device: profile regularization

  // Calculation intermediates
  double *weight_d;      // Boltzmann weights [B]
  double *mc_weight_d;   // MC reference weights [B]
  double *E_d;           // Bias energies [B]
  double *dEds_d;        // Energy derivatives [B]
  double *mc_E_d;        // MC bias energies [B]
  double *mc_dEds_d;     // MC energy derivatives [B]
  double *Z_d;           // Partition function
  double *mc_Z_d;        // MC partition function
  // NOTE: Zprofile/dLdZprofile fields removed — allocated but never referenced by any kernel.
  double *dLdE_d;        // Gradient w.r.t. energies [B]
  double *Gimp_d;        // Importance sampling reference [nprof * NBINS]
  // NOTE: G_d field removed — allocated but never referenced by any kernel.
  double *Esum_d;        // Sum of weighted energies
  double *dEdssum_d;     // Sum of energy derivatives
  double *mc_dEdssum_d;  // MC sum of energy derivatives
  double *moments_d;     // Bias parameter moments [nbias]
  double *mc_moments_d;  // MC bias parameter moments
  double *sumensweight_d;// Sum of ensemble weights

  // Optimization variables
  double *L_h;        // Host: log-likelihood
  double *L_d;        // Device: log-likelihood
  double *dLds_h;     // Host: line search derivative
  double *dLds_d;     // Device: line search derivative
  double *x_h;        // Host: bias parameters [nx]
  double *x_d;        // Device: bias parameters
  double *dLdx_h;     // Host: gradient [nx]
  double *dLdx_d;     // Device: gradient
  double *dxds_d;     // Device: search direction [nx]
  double *x0_h;       // Previous x for L-BFGS [nx]
  double *dLdx0_h;    // Previous gradient for L-BFGS [nx]
  double *hi_h;       // Search direction [nx]

  // L-BFGS memory (limited-memory BFGS)
  int Nmem;           // Current memory entries
  int Nmemax;         // Maximum memory entries (typically 50)
  double *d_x;        // Delta x history [Nmemax * nx]
  double *d_dLdx;     // Delta gradient history [Nmemax * nx]
  double *rho;        // Scaling factors [Nmemax]
  double *alpha;      // L-BFGS alpha [Nmemax]
  double *beta;       // L-BFGS beta [Nmemax]

  // Convergence
  double criteria;    // Convergence tolerance (default 1.25e-3)
  int doneCount;      // Consecutive convergence count
  int done;           // Optimization complete flag
  int max_iter;       // Maximum iterations

  // Paths
  char g_imp_path[256];  // Path to G_imp directory

  // Bias constants (derived from FNEX)
  double fnex;           // FNEX softmax constraint parameter
  double chi_offset;     // s-term sigmoid offset (4*exp(-FNEX))
  double omega_scale;      // x-term reciprocal decay (1/FNEX)
  double chi_offset_t;   // t-term sigmoid offset (independent of FNEX)
  double chi_offset_u;   // u-term Hill sigmoid offset (independent of FNEX)
  int ntriangle;         // pair params per unique pair: 5(bcxs), 7(+t), 9(+tu)
} struct_lmalf;

// In-memory entry points: accept pre-packed data from host pointers
// instead of reading files, enabling zero-file-I/O WHAM/LMALF calls.

#ifdef __cplusplus
extern "C" {
#endif

int wham_from_memory(
    int gpu_id,
    int nf, double temp, int nts0, int nts1, int use_gshift,
    int *nsubs, int nsites, const char *g_imp_path,
    double chi_offset, double omega_scale, double cutlsum,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double endpoint_weight, double endpoint_decay,
    double *D_flat, int *sim_indices, int *frame_counts,
    int total_frames, double *gshift_flat);

// Distributed WHAM: Phase A — f-value convergence only (rank 0)
int wham_iterate_from_memory(
    int gpu_id,
    int nf, double temp, int nts0, int nts1, int use_gshift,
    int *nsubs, int nsites, const char *g_imp_path,
    double chi_offset, double omega_scale, double cutlsum,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double endpoint_weight, double endpoint_decay,
    double *D_flat, int *sim_indices, int *frame_counts,
    int total_frames, double *gshift_flat,
    double *f_out, int *nf_out);

// Distributed WHAM: Phase B — profile computation for a subset (all ranks)
int wham_profiles_from_memory(
    int gpu_id,
    int nf, double temp, int nts0, int nts1, int use_gshift,
    int *nsubs, int nsites, const char *g_imp_path,
    double chi_offset, double omega_scale, double cutlsum,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double endpoint_weight, double endpoint_decay,
    double *D_flat, int *sim_indices, int *frame_counts,
    int total_frames, double *gshift_flat,
    double *f_in, int f_size,
    int profile_start, int profile_end,
    double *C_out, double *V_out, int *dim_out);

// Distributed WHAM: Phase B with slim D (no cross-energies, pre-computed lnDenom)
int wham_profiles_slim_from_memory(
    int gpu_id,
    int nf, double temp, int nts0, int nts1, int use_gshift,
    int *nsubs, int nsites, const char *g_imp_path,
    double chi_offset, double omega_scale, double cutlsum,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double endpoint_weight, double endpoint_decay,
    double *D_flat, int ndim,
    int *sim_indices, int *frame_counts,
    int total_frames, double *gshift_flat,
    double *f_in, int f_size,
    double *lnDenom_in,
    int profile_start, int profile_end,
    double *C_out, double *V_out, int *dim_out);

int lmalf_from_memory(
    int gpu_id,
    int nf, double temp, int ms, int msprof, int max_iter, double tolerance,
    int *nsubs, int nsites, const char *g_imp_path,
    double fnex, double chi_offset, double omega_scale,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double *lambda_flat, double *ensweight_flat, int n_frames,
    double *x_prev_flat, double *s_prev_flat, int nblocks_sq);

int wham_compute_weights_from_memory(
    int gpu_id,
    int nf, double temp, int nts0, int nts1, int use_gshift,
    int *nsubs, int nsites, const char *g_imp_path,
    double chi_offset, double omega_scale, double cutlsum,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double endpoint_weight, double endpoint_decay,
    double *D_flat, int *sim_indices, int *frame_counts,
    int total_frames, double *gshift_flat,
    double *weights_out,
    double *f_out,
    int *nf_out);

int nonlinear_from_memory(
    int gpu_id,
    int nf, double temp, int ms, int msprof, int max_iter, double tolerance,
    int *nsubs, int nsites,
    double fnex, double chi_offset, double omega_scale,
    double chi_offset_t, double chi_offset_u, int ntriangle,
    double *lambda_flat, double *ensweight_flat, int n_frames,
    double *x_prev_flat, double *s_prev_flat, int nblocks_sq);

#ifdef __cplusplus
}
#endif

// LMALF bin size constants (matching original lmalf.cu)
#define LMALF_NBINS 256
#define LMALF_NBINS2 16
#define LMALF_BLOCK 512
