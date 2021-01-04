#include "paddle/fluid/operators/transformer/transformer_op.h"
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <type_traits>
//#include "cublas_v2.h"
#include <memory>
#include "paddle/fluid/platform/dynload/cublas.h"
namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
__thread cublasHandle_t cublas_handle = nullptr;
///
template <typename DeviceContext, typename T>
class TransformerOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    // int64_t start_us = GetCurrentUs();
    using EncoderTraits =
      BertEncoderTransformerTraits<TransformerTraits<T>::OpType, cuda::OpenMultiHeadAttention>;
    const auto& from_tensor = ctx.Input<framework::LoDTensor>("FromTensor");
    // TODO check dims size
    int batch_size = from_tensor->dims()[0];
    int from_seq_len = from_tensor->dims()[1];
    const auto& to_tensor = ctx.Input<framework::LoDTensor>("ToTensor");
    int to_seq_len = to_tensor->dims()[1];
    int head_num = ctx.Attr<int>("head_num");
    int size_per_head = ctx.Attr<int>("size_per_head");
    const std::string hidden_act = ctx.Attr<std::string>("hidden_act");
    PADDLE_ENFORCE_EQ(from_seq_len, to_seq_len,
        "Unsupported argument, from_seq_len should be equal to to_seq_len");
    PADDLE_ENFORCE(from_seq_len >= 4 && from_seq_len <= 1024,
        "Unsupported argument, from_seq_len should be in the range of [4,1024]");
    PADDLE_ENFORCE(size_per_head == 32 || size_per_head == 64 || size_per_head == 96,
        "Unsupported argument, size_per_head should be 32 or 64 or 96");
    PADDLE_ENFORCE(head_num * size_per_head <= 1024,
        "Unsupported arguments, head_num * size_per_head should be less than 1024");
    PADDLE_ENFORCE(batch_size * from_seq_len * 3 <= 65536,
        "Unsupported arguments, batch_size * from_seq_len * 3 should be less than 65536");
    PADDLE_ENFORCE(hidden_act == "gelu" || hidden_act == "relu",
        "Unsupported hidden_act, hidden_act should be gelu or relu");
    // int64_t cublasCreate_start_us = GetCurrentUs();
    if (cublas_handle == nullptr) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasCreate(&cublas_handle));
    }
    // int64_t cublas_create_us = GetCurrentUs();
    // int64_t cublas_create_time = cublas_create_us - cublasCreate_start_us;
    // auto del_cublas_handle = [](cublasHandle_t handle) { platform::dynload::cublasDestroy(handle); };
    // std::unique_ptr<std::remove_pointer<cublasHandle_t>::type, decltype(del_cublas_handle)>
    //   cublas_handle_guard(cublas_handle, del_cublas_handle);
    fastertransformer::Allocator<AllocatorType::PADDLE> allocator(ctx);
    std::unique_ptr<BertEncoderTransformer<EncoderTraits>> encoder_transformer;
    try {
        encoder_transformer.reset(new BertEncoderTransformer<EncoderTraits>(allocator,
                batch_size, from_seq_len, to_seq_len, head_num, size_per_head, hidden_act));
    }
    catch(std::runtime_error& error)
    {
        // PADDLE_ENFORCE(ctx, false, errors::Internal(error.what()));
    }
    // PADDLE_ENFORCE(ctx.num_inputs() == 19, errors::InvalidArgument("Less input arguments"));
    
    EncoderInitParam<T> param; //init param here
    param.cublas_handle = cublas_handle;
    param.from_tensor = ctx.Input<framework::LoDTensor>("FromTensor")->data<T>();
    param.to_tensor = ctx.Input<framework::LoDTensor>("ToTensor")->data<T>();
    param.attr_mask = ctx.Input<framework::LoDTensor>("AttrMask")->data<T>();
    param.attr_kernel_Q = ctx.Input<Tensor>("AttrKernelQ")->data<T>();
    param.attr_kernel_K = ctx.Input<Tensor>("AttrKernelK")->data<T>();
    param.attr_kernel_V = ctx.Input<Tensor>("AttrKernelV")->data<T>();
    param.attr_bias_Q = ctx.Input<Tensor>("AttrBiasQ")->data<T>();
    param.attr_bias_K = ctx.Input<Tensor>("AttrBiasK")->data<T>();
    param.attr_bias_V = ctx.Input<Tensor>("AttrBiasV")->data<T>();
    param.attr_output_kernel = ctx.Input<Tensor>("AttrOutputKernel")->data<T>();
    param.attr_output_bias = ctx.Input<Tensor>("AttrOutputBias")->data<T>();
    param.attr_output_layernorm_beta = ctx.Input<Tensor>("AttrOutputLayernormBeta")->data<T>();
    param.attr_output_layernorm_gamma = ctx.Input<Tensor>("AttrOutputLayernormGamma")->data<T>();
    param.inter_kernel = ctx.Input<Tensor>("InterKernel")->data<T>();
    param.inter_bias = ctx.Input<Tensor>("InterBias")->data<T>();
    param.output_kernel = ctx.Input<Tensor>("OutputKernel")->data<T>();
    param.output_bias = ctx.Input<Tensor>("OutputBias")->data<T>();
    param.output_layernorm_beta = ctx.Input<Tensor>("OutputLayernormBeta")->data<T>();
    param.output_layernorm_gamma = ctx.Input<Tensor>("OutputLayernormGamma")->data<T>();
    auto output = ctx.Output<framework::LoDTensor>("Output");
    // TODO
    // output->Resize
    // output->set_lod
    param.transformer_out = output->mutable_data<T>(ctx.GetPlace());
    // PADDLE_ENFORCE(
    //         TransformerOpFunctor<DeviceContext, T>::Compute(
    //             ctx,
    //             &param,
    //             encoder_transformer.get()), "");
    auto ptr = encoder_transformer.get();
    auto rc = TransformerOpFunctor<DeviceContext, T>::Compute(ctx, &param, ptr);
    PADDLE_ENFORCE(rc, "TransformerOpFunctor return failed!!");
    // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDestroy(cublas_handle));
    // int64_t end_us = GetCurrentUs();
    // int64_t total_time = end_us - start_us;
    // std::cout << "Compute function time: " << total_time << std::endl;
  }
};
} // namespace operators
} // namespace paddle
namespace ops = paddle::operators;
REGISTER_OPERATOR(transformer, ops::TransformerOp, ops::TransformerOpMaker);
REGISTER_OP_CUDA_KERNEL(transformer, ops::TransformerOpKernel<paddle::platform::CUDADeviceContext, float>);
