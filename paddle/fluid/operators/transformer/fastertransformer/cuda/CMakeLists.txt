set(cuda_kernel_files
  cuda_kernels.cu
  open_attention.cu
)
nv_library(fastertransformer SRCS ${cuda_kernel_files})
