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
  double *invf_h;
  double *invf_d;
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
  int current_block_idx; // Current block index being processed
  int use_gshift;        // 0 = disabled (legacy), 1 = enabled (apply G_imp shifts)
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
  double *Zprofile_d;    // Profile partition functions [nprof * NBINS]
  double *mc_Zprofile_d; // MC profile partition functions
  double *dLdZprofile_d; // Gradient w.r.t. profiles
  double *mc_dLdZprofile_d;
  double *dLdE_d;        // Gradient w.r.t. energies [B]
  double *Gimp_d;        // Importance sampling reference [nprof * NBINS]
  double *G_d;           // Computed free energy profiles
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
} struct_lmalf;

// LMALF bin size constants (matching original lmalf.cu)
#define LMALF_NBINS 256
#define LMALF_NBINS2 16
#define LMALF_BLOCK 512
