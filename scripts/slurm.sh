#!/bin/bash

#SBATCH --job-name=slurm-demo
#SBATCH --time=00:10:00
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=64
#SBATCH --partition=standard
#SBATCH --output=demo-%j.out
#SBATCH --signal=SIGUSR1@60

ENV_NAME="resnet-slurm-demo"
PROJ_DIR="$HOME/proj/pytorch-lightning-slurm"
WORK_HOME=/temp/$USER
WORK_DIR="${WORK_HOME}/$(basename "${PROJ_DIR}")"

function create_work_dir() {
  if [ ! -d "${WORK_HOME}" ]; then
    echo "${WORK_HOME} doesn't exist, creating..."
    mkdir "${WORK_HOME}"
    chmod 700 "${WORK_HOME}"
  else
    echo "${WORK_HOME} already exists, skipping..."
  fi
}

function mamba_install() {
  pushd "${WORK_HOME}" || exit
  local MAMBAPREFIX
  MAMBAPREFIX="${WORK_HOME}/mambaforge"

  if [ -d "${MAMBAPREFIX}" ]; then
    echo "Mambaforge already installed, skipping..."
    return
  fi

  # Download and install mambaforge
  local INSTALLER
  INSTALLER="Mambaforge-$(uname)-$(uname -m).sh"
  wget https://github.com/conda-forge/miniforge/releases/latest/download/"${INSTALLER}"
  bash "${INSTALLER}" -b -p "${MAMBAPREFIX}" -s
  rm "${WORK_HOME}/${INSTALLER}"

  popd || exit
}

echo Running on "$(hostname)"...

create_work_dir

# copy files over
srun rsync -avP --delete "${PROJ_DIR}" "${WORK_HOME}"

# conda setup
mamba_install
eval "$(conda shell.bash hook)"
cd "${WORK_DIR}" || exit
mamba env remove -n "${ENV_NAME}"
mamba env create -n "${ENV_NAME}" -f environment.yaml
conda activate "${ENV_NAME}"

# training
srun python main.py fit \
  --config config/fit.yaml \
  --data.init_args.num_workers "$(nproc)" \
  --data.init_args.batch_size 512 \
  --data.init_args.data_dir "${WORK_HOME}"/datasets

# copying results back
srun rsync -avuP "${WORK_DIR}"/ "${PROJ_DIR}"
