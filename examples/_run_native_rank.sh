#!/bin/bash
set -eo pipefail

local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-}}}}"
visible_devices="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}}"
visible_devices="${visible_devices// /,}"

if [[ -n "$local_rank" && -n "$visible_devices" ]]; then
    IFS=',' read -r -a devices <<< "$visible_devices"
    if (( ${#devices[@]} > 1 )); then
        device_idx=$((local_rank % ${#devices[@]}))
        export CPHMD_PARENT_CUDA_VISIBLE_DEVICES="$visible_devices"
        export CUDA_VISIBLE_DEVICES="${devices[$device_idx]}"
    fi
fi

if [[ "${CPHMD_DEBUG_GPU_BINDING:-0}" == "1" ]]; then
    echo "rank_gpu_binding local_rank=${local_rank:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
fi

exec "$@"
