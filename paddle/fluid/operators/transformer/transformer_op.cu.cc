#include "paddle/fluid/operators/transformer/transformer_op.h"
#include <iostream>
#include <cuda_runtime.h>
#include <string>
namespace paddle {
namespace operators {
#ifdef PADDLE_WITH_CUDA
template <typename T>
struct TransformerOpFunctor<platform::CUDADeviceContext, T>
{
  static bool Compute(const framework::ExecutionContext& ctx,
      EncoderInitParam<T>* param,
      BertEncoderTransformer<BertEncoderTransformerTraits<TransformerTraits<T>::OpType, cuda::OpenMultiHeadAttention > > *encoder_transformer) {
    const auto stream = ctx.cuda_device_context().stream();
    param->stream = stream;
    try {
      check_cuda_error(platform::dynload::cublasSetStream(param->cublas_handle, stream));
      encoder_transformer->initialize(*param);
//      int forward_start_us = GetCurrentUs();
      encoder_transformer->forward();
//      int forward_end_us = GetCurrentUs();
//      std::cout << "forward time: " << forward_end_us - forward_start_us << std::endl;
      return true;
    }
    catch(std::runtime_error& error)
    {
      return false;
    }
    catch(...)
    {
      return false;
    }
  }
};
template struct TransformerOpFunctor<::paddle::platform::CUDADeviceContext, float>;
} // namespace operators 
} // namespace paddle
#endif

