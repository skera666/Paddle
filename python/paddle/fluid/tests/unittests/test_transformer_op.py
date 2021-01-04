from __future__ import print_function
import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_, in_dygraph_mode
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.initializer import NumpyArrayInitializer
from transformer_encoder import encoder_layer
batch_size = 1
from_seq_len = 4
def encoder_new(from_tensor,
                attr_mask,
                head_num,
                size_per_head,
                param_attrs):
  hidden_dim = head_num * size_per_head
  enc_input = fluid.layers.data(name="enc_input", shape=[batch_size, from_seq_len, hidden_dim], dtype="float32",append_batch_size=False)
  input_mask = fluid.layers.data(name="input_mask", shape=[batch_size, from_seq_len, from_seq_len], dtype="float32",append_batch_size=False)
  enc_output = layers.transformer(
      enc_input,
      enc_input,
      input_mask,
      head_num=head_num,
      size_per_head=size_per_head,
      param_attrs_dict=param_attrs)
  place = fluid.CUDAPlace(0)
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())
  result = exe.run(fluid.default_main_program(),
      feed={"enc_input": from_tensor, "input_mask": attr_mask},
      fetch_list=[enc_output],
      return_numpy=True)
  print('enc_output: ', result[0])
  return np.array(result[0])
def encoder(from_tensor,
            attr_mask,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_attrs,
            name=''):
  hidden_dim = d_model
  enc_input = fluid.layers.data(name="enc_input", shape=[batch_size, from_seq_len, hidden_dim], dtype="float32",append_batch_size=False)
  input_mask = fluid.layers.data(name="input_mask", shape=[batch_size, from_seq_len, from_seq_len], dtype="float32",append_batch_size=False)
  input_mask = fluid.layers.scale(
      x=input_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
  n_head_self_attn_mask = fluid.layers.stack(
      x=[input_mask] * n_head, axis=1)
 
  enc_output = encoder_layer(
      enc_input,
      n_head_self_attn_mask,
      n_head,
      d_key,
      d_value,
      d_model,
      d_inner_hid,
      prepostprocess_dropout,
      attention_dropout,
      relu_dropout,
      hidden_act,
      preprocess_cmd,
      postprocess_cmd,
      param_attrs,
      param_initializer=None,
      name="")
  
  place = fluid.CUDAPlace(0)
  exe = fluid.Executor(place)
  
  exe.run(fluid.default_startup_program())
  result = exe.run(fluid.default_main_program(),
      feed={"enc_input": from_tensor, "input_mask": attr_mask},
      fetch_list=[enc_output],
      return_numpy=True)
  print('enc_output: ', result[0])
  return np.array(result[0])
class TestTransformerOp(OpTest):
  def setUp(self):
    self.op_type = "transformer"
    batch_size = 1
    from_seq_len = 4
    to_seq_len = from_seq_len
    head_num = 1
    size_per_head = 32
    hidden_dim = head_num * size_per_head
    hidden_act = "relu"
    from_tensor = np.random.random((batch_size, from_seq_len, hidden_dim)).astype("float32")
    to_tensor = from_tensor
    attr_mask = np.random.random((batch_size, from_seq_len, from_seq_len)).astype("float32")
    attr_kernel_q = np.random.random((hidden_dim, hidden_dim)).astype("float32")
    attr_kernel_k = np.random.random((hidden_dim, hidden_dim)).astype("float32")
    attr_kernel_v = np.random.random((hidden_dim, hidden_dim)).astype("float32")
    attr_bias_q = np.random.random((hidden_dim)).astype("float32")
    attr_bias_k = np.random.random((hidden_dim)).astype("float32")
    attr_bias_v = np.random.random((hidden_dim)).astype("float32")
    attr_output_kernel = np.random.random((hidden_dim, hidden_dim)).astype("float32")
    attr_output_bias = np.random.random((hidden_dim)).astype("float32")
    attr_output_layernorm_beta = np.random.random((hidden_dim)).astype("float32") 
    attr_output_layernorm_gamma = np.random.random((hidden_dim)).astype("float32")
    inter_kernel = np.random.random((hidden_dim, hidden_dim * 4)).astype("float32")
    inter_bias = np.random.random((hidden_dim * 4)).astype("float32")
    output_kernel = np.random.random((hidden_dim * 4, hidden_dim)).astype("float32")
    output_bias = np.random.random((hidden_dim)).astype("float32")
    output_layernorm_beta = np.random.random((hidden_dim)).astype("float32")
    output_layernorm_gamma = np.random.random((hidden_dim)).astype("float32")
    self.inputs = {
      'FromTensor': from_tensor,
      'ToTensor': to_tensor,
      'AttrMask': attr_mask,
      'AttrKernelQ': attr_kernel_q,
      'AttrKernelK': attr_kernel_k,
      'AttrKernelV': attr_kernel_v,
      'AttrBiasQ': attr_bias_q,
      'AttrBiasK': attr_bias_k,
      'AttrBiasV': attr_bias_v,
      'AttrOutputKernel': attr_output_kernel,
      'AttrOutputBias': attr_output_bias,
      'AttrOutputLayernormBeta': attr_output_layernorm_beta,
      'AttrOutputLayernormGamma': attr_output_layernorm_gamma,
      'InterKernel': inter_kernel,
      'InterBias': inter_bias,
      'OutputKernel': output_kernel,
      'OutputBias': output_bias,
      'OutputLayernormBeta': output_layernorm_beta,
      'OutputLayernormGamma': output_layernorm_gamma
    }
    self.attrs = {
      'head_num': head_num,
      'size_per_head': size_per_head,
      'hidden_act': hidden_act
    }
 
    # param_attrs
    param_attrs = {}
    name = "multi_head_att"
    param_attrs["multi_head_att_query_fc.w_0"] = fluid.ParamAttr(
        name=name + '_query_fc.w_0',
        initializer=NumpyArrayInitializer(attr_kernel_q))
    param_attrs["multi_head_att_query_fc.b_0"] = fluid.ParamAttr(
        name=name + '_query_fc.b_0',
        initializer=NumpyArrayInitializer(attr_bias_q))
    name = ""
    #param_attrs["multi_head_att_query_fc.w_0"] = NumpyArrayInitializer(attr_kernel_q)
    #param_attrs["multi_head_att_query_fc.b_0"] = NumpyArrayInitializer(attr_bias_q)
    param_attrs["multi_head_att_key_fc.w_0"] = NumpyArrayInitializer(attr_kernel_k)
    param_attrs["multi_head_att_key_fc.b_0"] = NumpyArrayInitializer(attr_bias_k)
    param_attrs["multi_head_att_value_fc.w_0"] = NumpyArrayInitializer(attr_kernel_v)
    param_attrs["multi_head_att_value_fc.b_0"] = NumpyArrayInitializer(attr_bias_v)
    param_attrs["multi_head_att_output_fc.w_0"] = NumpyArrayInitializer(attr_output_kernel)
    param_attrs["multi_head_att_output_fc.b_0"] = NumpyArrayInitializer(attr_output_bias)
    
    #post_process_layer
    param_attrs["post_att_layer_norm_scale"] = NumpyArrayInitializer(attr_output_layernorm_gamma)
    param_attrs["post_att_layer_norm_bias"] = NumpyArrayInitializer(attr_output_layernorm_beta)
    
    param_attrs["post_ffn_layer_norm_scale"] = NumpyArrayInitializer(output_layernorm_gamma)
    param_attrs["post_ffn_layer_norm_bias"] = NumpyArrayInitializer(output_layernorm_beta)
    
    # positionwise_feed_forward
    param_attrs["ffn_fc_0.b_0"] = NumpyArrayInitializer(inter_bias)
    param_attrs["ffn_fc_0.w_0"] = NumpyArrayInitializer(inter_kernel)
    param_attrs["ffn_fc_1.b_0"] = NumpyArrayInitializer(output_bias)
    param_attrs["ffn_fc_1.w_0"] = NumpyArrayInitializer(output_kernel)
    # param_attrs for layers.transformer(fastertransformer)
    param_attrs_ftf = {
      'AttrKernelQ' : NumpyArrayInitializer(attr_kernel_q),
      'AttrKernelK' : NumpyArrayInitializer(attr_kernel_k),
      'AttrKernelV' : NumpyArrayInitializer(attr_kernel_v),
      'AttrBiasQ' : NumpyArrayInitializer(attr_bias_q),
      'AttrBiasK' : NumpyArrayInitializer(attr_bias_k),
      'AttrBiasV' : NumpyArrayInitializer(attr_bias_v),
      'AttrOutputKernel' : NumpyArrayInitializer(attr_output_kernel),
      'AttrOutputBias' : NumpyArrayInitializer(attr_output_bias),
      'AttrOutputLayernormBeta' : NumpyArrayInitializer(attr_output_layernorm_beta),
      'AttrOutputLayernormGamma' : NumpyArrayInitializer(attr_output_layernorm_gamma),
      'InterKernel' : NumpyArrayInitializer(inter_kernel),
      'InterBias' : NumpyArrayInitializer(inter_bias),
      'OutputKernel' : NumpyArrayInitializer(output_kernel),
      'OutputBias' : NumpyArrayInitializer(output_bias),
      'OutputLayernormBeta' : NumpyArrayInitializer(output_layernorm_beta),
      'OutputLayernormGamma' : NumpyArrayInitializer(output_layernorm_gamma)
    }
    print('from_tensor:', from_tensor)
    out = encoder(from_tensor,
             attr_mask,
             n_head=head_num,
             d_key=size_per_head,
             d_value=size_per_head,
             d_model=hidden_dim,
             d_inner_hid=hidden_dim*4,
             prepostprocess_dropout=0.,
             attention_dropout=0.,
             relu_dropout=0.,
             hidden_act=hidden_act,
             preprocess_cmd="",
             postprocess_cmd="dan",
             param_attrs=param_attrs,
             name='')
    self.outputs = {'Output': out}
    #out_new = encoder_new(from_tensor,
    #                      attr_mask,
    #                      head_num,
    #                      size_per_head,
    #                      param_attrs=param_attrs_ftf)
    #self.outputs = {'Output': out_new}
  def test_check_output(self):
    places = []
    if core.is_compiled_with_cuda() and core.op_support_gpu("transformer"):
      places.append(core.CUDAPlace(0))
    for place in places:
      self.check_output_with_place(place, atol=1e-5) 
if __name__ == "__main__":
  unittest.main()
