
#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/transformer/fastertransformer/common.h"
#include "paddle/fluid/operators/transformer/fastertransformer/faster_transformer.h"
#include "paddle/fluid/platform/dynload/cublas.h"

using namespace fastertransformer;
namespace paddle {
namespace operators {
///
inline int64_t GetCurrentUs() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
///
template <typename T>
struct TransformerTraits;
///
template <>
struct TransformerTraits<float>
{
  static const OperationType OpType = OperationType::FP32;
};
///
template <typename Device, typename T>
struct TransformerOpFunctor
{
  using EncoderTraits =
    BertEncoderTransformerTraits<TransformerTraits<T>::OpType, cuda::OpenMultiHeadAttention>;
  static bool Compute(const paddle::framework::ExecutionContext& ctx,
    EncoderInitParam<T>* param,
    BertEncoderTransformer<EncoderTraits>* encoder_transformer);
};
///
class TransformerOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("FromTensor", "");
        AddInput("ToTensor", "");
        AddInput("AttrMask", "");
        AddInput("AttrKernelQ", "");
        AddInput("AttrKernelK", "");
        AddInput("AttrKernelV", "");
        AddInput("AttrBiasQ", "");
        AddInput("AttrBiasK", "");
        AddInput("AttrBiasV", "");
        AddInput("AttrOutputKernel", "");
        AddInput("AttrOutputBias", "");
        AddInput("AttrOutputLayernormBeta", "");
        AddInput("AttrOutputLayernormGamma", "");
        AddInput("InterKernel", "");
        AddInput("InterBias", "");
        AddInput("OutputKernel", "");
        AddInput("OutputBias", "");
        AddInput("OutputLayernormBeta", "");
        AddInput("OutputLayernormGamma", "");
        AddOutput("Output", "");
        AddAttr<int>("head_num", "")
            .SetDefault(12)
            .EqualGreaterThan(1);
        AddAttr<int>("size_per_head", "")
            .SetDefault(64)
            .EqualGreaterThan(32);
        AddAttr<std::string>("hidden_act", "")
          .SetDefault("gelu");
        AddComment("");
    }
};
///
class TransformerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
      PADDLE_ENFORCE(ctx->HasInput("FromTensor"),
              "FromTensor(Input) of TransformerOp should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("ToTensor"),
              "ToTensor(Input) of TransformerOp should not be null.");
      PADDLE_ENFORCE(ctx->HasInput("AttrMask"),
              "AttrMask(Input) of TransformerOp should not be null.");
      // TODO others?
      PADDLE_ENFORCE(ctx->HasOutput("Output"),
              "Output(Output) of TransformerOp should not be null.");
      auto in_dims = ctx->GetInputDim("FromTensor");
      ctx->SetOutputDim("Output", in_dims);
  }
protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::LoDTensor>("FromTensor")->type(),
        platform::CUDAPlace());
  }
};
} // namespace operators
} // namespace paddle
