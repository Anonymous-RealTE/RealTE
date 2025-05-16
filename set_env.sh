export CONDA_HOME=/root/miniconda3_ov/
export CUDA_HOME=/root/cuda_12.3.0_cudnn_8.9.7
export PATH=$CONDA_HOME/external_pkgs/gcc-9.5/bin:$CONDA_HOME/bin:$CONDA_HOME/condabin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/root/.zinit/polaris/sbin:/root/.zinit/polaris/bin:$CONDA_HOME/bin:$CONDA_HOME/condabin:/root/.BCloud/bin:/home/opt/cuda_tools:/opt/_internal/cpython-3.7.0/bin:/opt/conda/envs/py36/bin:/root/paddlejob/hadoop-client/hadoop/bin:/usr/local/openmpi-3.1.0/bin:/home/cmake-3.16.0-Linux-x86_64/bin:/root/paddlejob/jdk-1.8.0/bin:/root/paddlejob/flume-1.8.0/bin:/usr/local/gcc-8.2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin
export LD_LIBRARY_PATH=$CONDA_HOME/lib/python3.9/site-packages/nvidia/cudnn/lib:$CONDA_HOME/external_pkgs/gcc-9.5/lib:$CONDA_HOME/external_pkgs/gcc-9.5/lib64:$CONDA_HOME/lib:/root/cuda_12.3.0_cudnn_8.9.7/lib64:/usr/local/lib:/usr/local/x86_64-pc-linux-gnu/lib:/home/opt/nvidia_lib:/usr/local/cuda/lib64:/usr/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu/
export CUTLASS_PATH=$CONDA_HOME/external_resources/cutlass/
export NCCL_ALGO=Ring
export NCCL_NVLS_ENABLE=0