"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from calculon import *
from .layers import *

class Llm:
  """
  This implements the transformer with tensor, pipeline, and data parallelism.
  Using it follows this pattern:
  1. Initialize the model with certain model parameters
  2. Compile it with certain optimizations and parallelization strategies
  3. Run on particular hardware system
  """

  class Application:
    """Specifies the application configuration."""
    def __init__(self, cfg):
      self.cfg = cfg
      self.hidden = cfg['hidden']
      self.feedforward = cfg['feedforward']
      self.seq_size = cfg['seq_size']
      self.attn_heads = cfg['attn_heads']
      self.kv_groups = cfg.get('kv_groups', self.attn_heads)
      self.attn_size = cfg['attn_size']
      self.num_blocks = cfg['num_blocks']
      self.vocab_size = cfg.get('vocab_size', 51200)
      # tie token embedding weights with LM head
      self.tie_embeddings = cfg.get('tie_embeddings', True)
      # When PP > 1, statistics reported are per-rank. To make the effect of
      # embedding visible, we assume this process is the first PP stage by
      # default. You can override this via "pp_stage_role" with values:
      #   "first | middle | last"
      self.pp_stage_role = cfg.get('pp_stage_role', 'first')

    def num_parameters(self):
      # https://cs.stanford.edu/~matei/papers/2021/sc_megatron_lm.pdf
      # Equation 2
      p = 2 * self.hidden * self.feedforward                                          # MLP weights
      p += self.hidden * (2 * self.attn_heads + 2 * self.kv_groups) * self.attn_size  # Attn weights
      p += self.hidden + self.feedforward                                             # biases MLP
      p += (2 * self.kv_groups + self.attn_heads) * self.attn_size + self.hidden      # biases Attn
      p += 2 * 2 * self.hidden                                                        # layer norm
      p *= self.num_blocks                                                            # per each block
      if self.tie_embeddings:
        p += (self.vocab_size + self.seq_size) * self.hidden                          # token+position embeddings
      else:
        p += (2 * self.vocab_size + self.seq_size) * self.hidden                      # token+position embeddings and LM head
      return p

  class Execution:
    """Specifies the execution configuration."""

    @staticmethod
    def fields():
      return (
        'num_procs', 'tensor_par', 'pipeline_par', 'data_par', 'tensor_par_net',
        'pipeline_par_net', 'data_par_net', 'batch_size', 'microbatch_size',
        'datatype', 'fused_activation', 'qkv_packing', 'attention_type',
        'activation_recompute', 'pipeline_interleaving', 'optimizer_sharding',
        'tensor_par_comm_type', 'tensor_par_overlap', 'seq_par_ag_redo',
        'data_par_overlap', 'weight_offload', 'activations_offload',
        'optimizer_offload', 'grad_reduce_in_fp32', 'training')

    @staticmethod
    def from_json(cfg):
      cfg.setdefault('qkv_packing', True)
      cfg.setdefault('grad_reduce_in_fp32', False)
      assert set(cfg.keys()) == set(Llm.Execution.fields())
      values = [cfg[field] for field in Llm.Execution.fields()]
      return Llm.Execution(*values)

    def __init__(self, num_procs, tensor_par, pipeline_par, data_par,
                 tensor_par_net, pipeline_par_net, data_par_net,
                 batch_size, microbatch_size, datatype, fused_activation,
                 qkv_packing, attention_type, activation_recompute,
                 pipeline_interleaving, optimizer_sharding,
                 tensor_par_comm_type, tensor_par_overlap,
                 seq_par_ag_redo, data_par_overlap, weight_offload,
                 activations_offload, optimizer_offload,
                 grad_reduce_in_fp32, training):
      self.training = training
      self.num_procs = num_procs
      assert self.num_procs > 0
      self.tensor_par = tensor_par
      assert self.tensor_par > 0
      self.pipeline_par = pipeline_par
      assert self.pipeline_par > 0
      self.data_par = data_par
      assert self.data_par > 0
      assert self.num_procs == self.tensor_par * self.pipeline_par * \
        self.data_par, 'tensor * pipeline * data parallelism != num_procs'
      self.tensor_par_net = tensor_par_net
      self.pipeline_par_net = pipeline_par_net
      self.data_par_net = data_par_net
      self.global_batch_size = batch_size
      assert self.global_batch_size > 0
      self.microbatch_size = microbatch_size
      assert self.microbatch_size > 0
      assert self.global_batch_size % self.data_par == 0
      self._local_batch_size = self.global_batch_size // self.data_par
      assert self._local_batch_size % self.microbatch_size == 0
      self._num_microbatches = self._local_batch_size // self.microbatch_size
      self.datatype = datatype
      self.fused_activation = fused_activation
      self.qkv_packing = bool(qkv_packing)
      self.attention_type = attention_type
      assert self.attention_type in ['multihead', 'multiquery', 'groupquery']
      self.activation_recompute = activation_recompute
      assert self.activation_recompute in ['full', 'attn_only', 'none']
      if self.activation_recompute in ['full', 'attn_only']:
        assert self.training, "We only perform recompute during training"
      self.pipeline_interleaving = pipeline_interleaving
      assert self.pipeline_interleaving > 0, \
        f'Bad pipeline interleaving of {self.pipeline_interleaving}'
      if self.pipeline_par == 1:
        assert self.pipeline_interleaving == 1, \
        f'Bad pipeline interleaving of {self.pipeline_interleaving} with PP=1'
      self.optimizer_sharding = optimizer_sharding
      if self.optimizer_sharding:
        assert self.data_par > 1, "We perform optimizer sharding with DP > 1"
      self.tensor_par_comm_type = tensor_par_comm_type
      self.in_network_reduction = False
      assert self.tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']
      self.tensor_par_overlap = tensor_par_overlap
      assert self.tensor_par_overlap in ['none', 'ring', 'pipe']
      if self.tensor_par_overlap != 'none':
        assert self.tensor_par > 1, "We perform TP comm overlap with TP > 1"
      self._sequence_par = self.tensor_par_comm_type == 'rs_ag'
      self.seq_par_ag_redo = seq_par_ag_redo
      if self.seq_par_ag_redo:
        assert self.tensor_par_comm_type == 'rs_ag', "We only redo AG comm"
        assert self._sequence_par, "We only redo AG with sequence parallelism"
        assert self.activation_recompute != 'full', \
          "We assume no extra AG with full recompute"
      self._pipeline_par_rs_ag = \
        self.tensor_par_comm_type in ['p2p_rs_ag', 'rs_ag']
      self.data_par_overlap = data_par_overlap
      if self.data_par_overlap:
        assert self.training, "We only perform DP comm overlap during training"
        assert self.data_par > 1, "We perform DP comm overlap with DP > 1"
      self.weight_offload = weight_offload
      self.activations_offload = activations_offload
      self.optimizer_offload = optimizer_offload
      self.grad_reduce_in_fp32 = bool(grad_reduce_in_fp32)
      if self.optimizer_offload:
        assert self.training, \
          "We only perform optimizer offloading during training"

    def get_json(self):
      keys = Llm.Execution.fields()
      values = [
        self.num_procs, self.tensor_par, self.pipeline_par, self.data_par, self.tensor_par_net,
        self.pipeline_par_net, self.data_par_net, self.global_batch_size, self.microbatch_size,
        self.datatype, self.fused_activation, self.qkv_packing, self.attention_type,
        self.activation_recompute, self.pipeline_interleaving, self.optimizer_sharding,
        self.tensor_par_comm_type, self.tensor_par_overlap, self.seq_par_ag_redo,
        self.data_par_overlap, self.weight_offload, self.activations_offload,
        self.optimizer_offload, self.grad_reduce_in_fp32, self.training
      ]
      assert len(keys) == len(values)
      return dict(zip(keys, values))

    def get_peers_json(self):
      peers = {}
      for di in range(self.data_par):
        for pi in range(self.pipeline_par):
          for ti in range(self.tensor_par):
            nid = (di * self.tensor_par * self.pipeline_par +
                   pi * self.tensor_par +
                   ti)
            peers[nid] = {}

            # tensor parallelism peers
            if self.tensor_par > 1:
              peers[nid]['tensor'] = []
              for ti2 in range(self.tensor_par):
                pid = (di * self.tensor_par * self.pipeline_par +
                       pi * self.tensor_par +
                       ti2)
                peers[nid]['tensor'].append(pid)

            # pipeline parallelism peer
            if self.pipeline_par > 1:
              peers[nid]['pipeline'] = None
              pi2 = (pi + 1) % self.pipeline_par
              pid = (di * self.tensor_par * self.pipeline_par +
                     pi2 * self.tensor_par +
                     ti)
              peers[nid]['pipeline'] = pid

            # data parallelism peers
            if self.data_par > 1:
              peers[nid]['data'] = []
              for di2 in range(self.data_par):
                pid = (di2 * self.tensor_par * self.pipeline_par +
                       pi * self.tensor_par +
                       ti)
                peers[nid]['data'].append(pid)
      return peers


  # This is used for errors where the user may not be fully aware of
  # limitations. Use it like this:
  #   raise self.Error(f'Foo bar {num1} is not {num2}')
  class Error(Exception):
    pass

  @staticmethod
  def _factors(x):
    for cand in range(1, x + 1):
      if x % cand == 0:
        yield cand

  @staticmethod
  def get_all_tensor_parallelisms(num_procs, hidden, attn_heads, kv_groups):
    for cand in Llm._factors(num_procs):
      if hidden % cand == 0 and attn_heads % cand == 0 and kv_groups % cand == 0:
        yield cand

  @staticmethod
  def get_all_pipeline_parallelisms(num_procs, tensor_par, num_blocks):
    assert num_procs % tensor_par == 0
    max_pp = min(num_procs // tensor_par, num_blocks)
    for cand in Llm._factors(max_pp):
      if (num_procs % (tensor_par * cand) == 0 and
          num_blocks % cand == 0):
        yield cand

  @staticmethod
  def get_data_parallelism(num_procs, tensor_par, pipeline_par):
    assert num_procs % (tensor_par * pipeline_par) == 0, \
      f'np={num_procs} tp={tensor_par} pp={pipeline_par}'
    return num_procs // (tensor_par * pipeline_par)

  @staticmethod
  def get_valid_pipeline_interleavings(num_blocks, pipeline_par):
    """
    Valid interleaving factors are divisors of the number of blocks handled
    per processor (ceil(num_blocks / pipeline_par)). For PP==1 we force 1,
    matching Execution.__init__ and compile().
    """
    if pipeline_par == 1:
      yield 1
      return
    blocks_per_proc = (num_blocks + pipeline_par - 1) // pipeline_par
    for cand in Llm._factors(blocks_per_proc):
      yield cand

  @staticmethod
  def get_valid_microbatch_sizes(
      seq_size,
      tensor_par,
      data_par,
      global_batch_size,
      pipeline_par,
      tensor_par_comm_type=None
  ):
    """
    Yield microbatch sizes that divide the local batch. If using sequence-parallel
    (tensor_par_comm_type == 'rs_ag'), additionally require that
    (microbatch * seq_size) % tensor_par == 0 so that sequence shards are even.
    """
    assert global_batch_size % data_par == 0
    local_batch_size = global_batch_size // data_par
    require_seq_div = (tensor_par > 1) and (tensor_par_comm_type == 'rs_ag')
    for cand in Llm._factors(local_batch_size):
      if require_seq_div and ((cand * seq_size) % tensor_par != 0):
        continue
      yield cand

  @staticmethod
  def can_redo_ag(tensor_par_comm_type, activation_recompute):
    return tensor_par_comm_type == 'rs_ag' and activation_recompute != 'full'

  def __init__(self, app, log):
    assert isinstance(app, self.Application)
    self.app = app
    self.log = log

    # Set during compile
    self.exe = None

    # Set during run
    self.sys = None

    # State of calling compile() and run()
    self._compiled = False
    self._executed = False

    # Holds the layers in a single block
    self._llm_block = []

    # A chunk is a set of blocks for microbatch before passing to the next
    # processor in the pipeline. Each chunk is modeled as a base
    # block that is repeated N-1 times and followed by 1 edge block.
    # Recommunication time is the same in both base and edge blocks.
    self._blocks_per_proc = None
    self._bubble_reduction_blocks = None
    self._blocks_per_chunk = None
    self._chunks_per_proc = None
    self._baseblocks_per_chunk = None
    self._edgeblocks_per_chunk = None

    # Misc compilation values
    self._bytes_per_element = None
    self._batch_seq = None
    self._batch_seq_par = None
    self._activation_size = None
    self._seq_par_activation_size = None

    # Assignments to specific networks
    self._tp_net = None
    self._pp_net = None
    self._dp_net = None

    # metrics collected after run for each microbatch
    self._block_fw_flops = None
    self._block_fw_flops_time = None
    self._block_fw_mem_accessed = None
    self._block_fw_mem_time = None
    self._block_fw_time = None
    self._block_re_flops = None
    self._block_re_flops_time = None
    self._block_re_mem_accessed = None
    self._block_re_mem_time = None
    self._block_re_time = None
    self._block_agrad_flops = None
    self._block_agrad_flops_time = None
    self._block_agrad_mem_accessed = None
    self._block_agrad_mem_time = None
    self._block_agrad_time = None
    self._block_wgrad_flops = None
    self._block_wgrad_flops_time = None
    self._block_wgrad_mem_accessed = None
    self._block_wgrad_mem_time = None
    self._block_wgrad_time = None
    self._block_optim_flops = None
    self._block_optim_flops_time = None
    self._block_optim_mem_accessed = None
    self._block_optim_mem_time = None
    self._block_optim_time = None

    self._baseblock_fw_tp_size = None
    self._edgeblock_fw_tp_size = None
    self._baseblock_agrad_tp_size = None
    self._edgeblock_agrad_tp_size = None
    self._baseblock_recomm_size = None
    self._edgeblock_recomm_size = None
    self._block_fw_pp_size = None
    self._block_bw_pp_size = None
    self._block_dp_size = None
    self._baseblock_fw_time_no_offload = None
    self._edgeblock_fw_time_no_offload = None
    self._baseblock_bw_time_no_offload = None
    self._edgeblock_bw_time_no_offload = None
    self._baseblock_fw_offload_overhead = None
    self._edgeblock_fw_offload_overhead = None
    self._baseblock_bw_offload_overhead = None
    self._edgeblock_bw_offload_overhead = None
    self._baseblock_fw_time = None
    self._edgeblock_fw_time = None
    self._baseblock_bw_time = None
    self._edgeblock_bw_time = None
    self._block_dp_time = None
    self._tp_bw_overlap_req = None
    self._dp_bw_overlap_req_chunk = None
    self._dp_bw_overlap_req_tail = None

    self._block_weight_space = None
    self._block_act_working_space = None
    self._block_act_storage_space = None
    self._block_act_checkpoint_size = None
    self._block_weight_grad_space = None
    self._block_weight_grad_space_no_sharding = None
    self._block_act_grad_space = None
    self._block_optimizer_space = None

    # Top level memory usage stats
    self._weight_space = None
    self._act_space = None
    self._act_checkpoint_size = None
    self._weight_grad_space = None
    self._act_grad_space = None
    self._optimizer_space = None

    # Top level throughput stats
    self._fw_flops = None
    self._fw_flops_time = None
    self._fw_mem_accessed = None
    self._fw_mem_time = None
    self._fw_time = None
    self._baseblock_fw_tp_time = None
    self._edgeblock_fw_tp_time = None
    self._baseblock_fw_tp_time_exposed = None
    self._edgeblock_fw_tp_time_exposed = None
    self._re_flops = None
    self._re_flops_time = None
    self._re_mem_accessed = None
    self._re_mem_time = None
    self._re_time = None
    self._baseblock_recomm_time = None
    self._edgeblock_recomm_time = None
    self._baseblock_recomm_time_exposed = None
    self._edgeblock_recomm_time_exposed = None
    self._agrad_flops = None
    self._agrad_flops_time = None
    self._agrad_mem_accessed = None
    self._agrad_mem_time = None
    self._baseblock_agrad_tp_time = None
    self._edgeblock_agrad_tp_time = None
    self._baseblock_agrad_tp_time_exposed = None
    self._edgeblock_agrad_tp_time_exposed = None
    self._agrad_time = None
    self._wgrad_flops = None
    self._wgrad_flops_time = None
    self._wgrad_mem_accessed = None
    self._wgrad_mem_time = None
    self._wgrad_time = None
    self._optim_flops = None
    self._optim_flops_time = None
    self._optim_mem_accessed = None
    self._optim_mem_time = None
    self._optim_time = None

    # Top level network stats
    self._tp_comm_time_exposed = None
    self._tp_comm_time_link = None
    self._recomm_time_exposed = None
    self._recomm_time_link = None
    self._pp_comm_time_exposed = None
    self._pp_comm_time_link = None
    self._dp_comm_time_exposed = None
    self._dp_comm_time_link = None
    self._bubble_time = None

    # Embedding and LM head stats
    self._embed_weight_elems_shard = 0
    self._lm_head_weight_elems_shard = 0
    self._pp_embed_grad_time_per_batch = 0.0  # PP AR for tied embeddings' grads, time per batch, not per micro-batch
    self._tp_embed_fw_time_per_micro = 0.0    # vocab-parallel embedding FW AR
    self._tp_embed_bw_time_per_micro = 0.0    # vocab-parallel embedding BW AR
    self._tp_lm_head_fw_time_per_micro = 0.0  # LM head input AG (or AR)
    self._tp_lm_head_bw_time_per_micro = 0.0  # LM head grad RS (or AR)
    self._padded_vocab = 0                    # Padded vocab (aligned to TP) for LM-head
    self._lm_head_fw_comp_time_per_micro = 0.0
    self._lm_head_bw_comp_time_per_micro = 0.0

  @staticmethod
  def get_stats_fields():
    return (
      'block_fw_flops',
      'block_fw_flops_time',
      'block_fw_mem_accessed',
      'block_fw_mem_time',
      'block_fw_time',
      'baseblock_fw_tp_time',
      'edgeblock_fw_tp_time',
      'baseblock_fw_tp_time_exposed',
      'edgeblock_fw_tp_time_exposed',
      'block_re_flops',
      'block_re_flops_time',
      'block_re_mem_accessed',
      'block_re_mem_time',
      'block_re_time',
      'baseblock_recomm_time',
      'edgeblock_recomm_time',
      'baseblock_recomm_time_exposed',
      'edgeblock_recomm_time_exposed',
      'block_agrad_flops',
      'block_agrad_flops_time',
      'block_agrad_mem_accessed',
      'block_agrad_mem_time',
      'block_agrad_time',
      'baseblock_agrad_tp_time',
      'edgeblock_agrad_tp_time',
      'baseblock_agrad_tp_time_exposed',
      'edgeblock_agrad_tp_time_exposed',
      'block_wgrad_flops',
      'block_wgrad_flops_time',
      'block_wgrad_mem_accessed',
      'block_wgrad_mem_time',
      'block_wgrad_time',
      'block_optim_flops',
      'block_optim_flops_time',
      'block_optim_mem_accessed',
      'block_optim_mem_time',
      'block_optim_time',

      'baseblock_fw_tp_size',
      'edgeblock_fw_tp_size',
      'baseblock_agrad_tp_size',
      'edgeblock_agrad_tp_size',
      'baseblock_recomm_size',
      'edgeblock_recomm_size',
      'block_fw_pp_size',
      'block_bw_pp_size',
      'block_dp_size',
      'tp_bw_overlap_req',
      'dp_bw_overlap_req_chunk',
      'dp_bw_overlap_req_tail',

      'block_weight_space',
      'block_act_working_space',
      'block_act_storage_space',
      'block_act_checkpoint_size',
      'block_weight_grad_space',
      'block_weight_grad_space_no_sharding',
      'block_act_grad_space',
      'block_optimizer_space',

      'weight_space_with_offload',
      'act_space_with_offload',
      'act_checkpoint_size_with_offload',
      'act_grad_space_with_offload',
      'weight_grad_space_with_offload',
      'optimizer_space_with_offload',

      'weight_space',
      'act_space',
      'act_checkpoint_size',
      'act_grad_space',
      'weight_grad_space',
      'optimizer_space',

      'fw_time',
      'bw_time',
      'optim_step_time',
      'recompute_time',
      'recomm_link_time',
      'recomm_exposed_time',
      'bubble_time',
      'tp_comm_link_time',
      'pp_comm_link_time',
      'dp_comm_link_time',
      'tp_comm_exposed_time',
      'pp_comm_exposed_time',
      'dp_comm_exposed_time',
      'fw_offload_exposed_time',
      'bw_offload_exposed_time',
      'total_time',
      'act_offload_bw_req',
      'weight_offload_bw_req',
      'optim_offload_bw_req',
      'offload_mem_bw_req',
      'proc_mem_tier1_cap_req',
      'proc_mem_tier2_cap_req',
      'useful_flops',
      'compute_efficiency',
      'system_efficiency',
      'total_efficiency',
      'sample_rate')

  def get_stats_values(self):
    assert self._executed
    return (
      self._block_fw_flops,
      self._block_fw_flops_time,
      self._block_fw_mem_accessed,
      self._block_fw_mem_time,
      self._block_fw_time,
      self._baseblock_fw_tp_time,
      self._edgeblock_fw_tp_time,
      self._baseblock_fw_tp_time_exposed,
      self._edgeblock_fw_tp_time_exposed,
      self._block_re_flops,
      self._block_re_flops_time,
      self._block_re_mem_accessed,
      self._block_re_mem_time,
      self._block_re_time,
      self._baseblock_recomm_time,
      self._edgeblock_recomm_time,
      self._baseblock_recomm_time_exposed,
      self._edgeblock_recomm_time_exposed,
      self._block_agrad_flops,
      self._block_agrad_flops_time,
      self._block_agrad_mem_accessed,
      self._block_agrad_mem_time,
      self._block_agrad_time,
      self._baseblock_agrad_tp_time,
      self._edgeblock_agrad_tp_time,
      self._baseblock_agrad_tp_time_exposed,
      self._edgeblock_agrad_tp_time_exposed,
      self._block_wgrad_flops,
      self._block_wgrad_flops_time,
      self._block_wgrad_mem_accessed,
      self._block_wgrad_mem_time,
      self._block_wgrad_time,
      self._block_optim_flops,
      self._block_optim_flops_time,
      self._block_optim_mem_accessed,
      self._block_optim_mem_time,
      self._block_optim_time,

      self._baseblock_fw_tp_size,
      self._edgeblock_fw_tp_size,
      self._baseblock_agrad_tp_size,
      self._edgeblock_agrad_tp_size,
      self._baseblock_recomm_size,
      self._edgeblock_recomm_size,
      self._block_fw_pp_size,
      self._block_bw_pp_size,
      self._block_dp_size,
      self._tp_bw_overlap_req,
      self._dp_bw_overlap_req_chunk,
      self._dp_bw_overlap_req_tail,

      self._block_weight_space,
      self._block_act_working_space,
      self._block_act_storage_space,
      self._block_act_checkpoint_size,
      self._block_weight_grad_space,
      self._block_weight_grad_space_no_sharding,
      self._block_act_grad_space,
      self._block_optimizer_space,

      self.get_weight_space_min(),
      self.get_act_space_min(),
      self.get_act_checkpoint_size_min(),
      self.get_act_grad_space_min(),
      self.get_weight_grad_space_min(),
      self.get_optimizer_space_min(),

      self.get_weight_space(),
      self.get_act_space(),
      self.get_act_checkpoint_size(),
      self.get_act_grad_space(),
      self.get_weight_grad_space(),
      self.get_optimizer_space(),

      self.get_fw_time(),
      self.get_bw_time(),
      self.get_optim_step_time(),
      self.get_recompute_time(),
      self.get_recomm_link_time(),
      self.get_recomm_exposed_time(),
      self.get_bubble_time(),
      self.get_tp_comm_link_time(),
      self.get_pp_comm_link_time(),
      self.get_dp_comm_link_time(),
      self.get_tp_comm_exposed_time(),
      self.get_pp_comm_exposed_time(),
      self.get_dp_comm_exposed_time(),
      self.get_fw_offload_overhead(),
      self.get_bw_offload_overhead(),
      self.get_total_time(),
      self.get_act_offload_bw_req(),
      self.get_weight_offload_bw_req(),
      self.get_optim_offload_bw_req(),
      self.get_offload_mem_bw_req(),
      self.get_mem_tier1_cap_req(),
      self.get_mem_tier2_cap_req(),
      self.get_useful_flops(),
      self.get_compute_efficiency(),
      self.get_system_efficiency(),
      self.get_total_efficiency(),
      self.get_sample_rate())

  def get_stats_json(self, include_layers):
    assert self._executed
    keys = Llm.get_stats_fields()
    values = self.get_stats_values()
    assert len(keys) == len(values), f'{len(keys)} {len(values)}'
    j = dict(zip(keys, values))
    if include_layers:
      j['layers'] = []
      for layer in self._llm_block:
        j['layers'].append(layer.get_stats_json())
    return j

  def _build_attn_block(self):
    recompute_flag = self.exe.activation_recompute == "full"
    recompute_attn_flag = self.exe.activation_recompute in \
      ["full", "attn_only"]
    recompute_ag_flag = recompute_attn_flag or self.exe.seq_par_ag_redo

    assert self.app.hidden % self.exe.tensor_par == 0, (
      f"We should split hidden={self.app.hidden} between"
      f" {self.exe.tensor_par} TP partitions evenly")
    assert self.app.feedforward % self.exe.tensor_par == 0, (
      f"We should split feedforward={self.app.feedforward} between"
      f" {self.exe.tensor_par} TP partitions evenly")
    assert self.app.attn_heads % self.exe.tensor_par == 0, (
      f"We should split {self.app.attn_heads} attn_heads between"
      f" {self.exe.tensor_par} TP partitions evenly")

    self._llm_block.append(Fork(
      "AttnBlock_Fork",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size, self._activation_size),
      2,
      needs_recompute=recompute_flag,
      # We account this activation when consider Residual and LayerNorm
      activation_stored=True))
    self._llm_block.append(LayerNorm(
      "AttnBlock_LayerNorm",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size, self._activation_size),
      self.app.hidden,
      needs_recompute=recompute_flag,
      # Activation is stored in Fork instead
      activation_stored=False,
      activation_reused=True))
    if self.exe.tensor_par_overlap == 'none':
      self._llm_block.append(TPComm(
        "AttnBlock_F",
        self.sys,
        self._activation_size,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        # We only compute flops/mem analyzing this layers, comm analyzed later
        # This is conservative estimate that does not consider p2p_rs_ag
        # because we don't differentiate between edge and middle blocks here
        tensor_par_comm_type=self.exe.tensor_par_comm_type,
        conjugate=False,
        in_network_reduction=self.exe.in_network_reduction,
        needs_recomm=recompute_ag_flag))
      if self.exe.qkv_packing:
        assert self.exe.attention_type != 'multiquery', \
          "Multiquery attention does not support qkv packing"
        if self.exe.attention_type in ['multihead', 'groupquery']:
          assert not (self.exe.attention_type == 'multihead' and self.app.kv_groups != self.app.attn_heads), \
            "kv_groups must be equal to attn_heads for multihead attention"
          assert self.app.attn_heads % self.app.kv_groups == 0, "Number of heads must be divisible by kv_groups"
          assert self.app.kv_groups % self.exe.tensor_par == 0, "kv_groups must be divisible by tensor_par"
          self._llm_block.append(Linear(
            "AttnBlock_QKV",
            self.sys,
            self.exe.microbatch_size,
            self.app.seq_size,
            self.app.hidden,
            (self.app.attn_heads + 2 * self.app.kv_groups) * self.app.attn_size // self.exe.tensor_par,
            needs_recompute=recompute_flag,
            # Fork removed -> QKV must claim input-activation storage
            activation_stored=(not recompute_ag_flag),
            activation_reused=False
          ))
        # View op is zero-FLOP, it shouldn't add cost, so we just ignore it.
        else:
          raise self.Error(f"Wrong attention type: {self.exe.attention_type}")
      else:
        self._llm_block.append(Fork(
          "AttnBlock_Multihead_Fork",
          self.sys,
          self._activation_size,
          3,
          needs_recompute=recompute_ag_flag,
          # With seq_par, we use activations from Comm layers to reflect that
          # they're split, otherwise we keep full size activations
          activation_stored=(not recompute_ag_flag)))
        self._llm_block.append(Linear(
          "AttnBlock_Query",
          self.sys,
          self.exe.microbatch_size,
          self.app.seq_size,
          self.app.hidden,
          self.app.attn_heads * self.app.attn_size // self.exe.tensor_par,
          needs_recompute=recompute_flag,
          # Activation is stored in Fork instead,
          activation_stored=False,
          activation_reused=True))
        if self.exe.attention_type in ['multihead', 'groupquery']:
          assert not (self.exe.attention_type == 'multihead' and self.app.kv_groups != self.app.attn_heads), \
            "kv_groups must be equal to attn_heads for multihead attention"
          assert self.app.attn_heads % self.app.kv_groups == 0, "Number of heads must be divisible by kv_groups"
          assert self.app.kv_groups % self.exe.tensor_par == 0, "kv_groups must be divisible by tensor_par"
          self._llm_block.append(Linear(
            "AttnBlock_Key",
            self.sys,
            self.exe.microbatch_size,
            self.app.seq_size,
            self.app.hidden,
            self.app.kv_groups * self.app.attn_size // self.exe.tensor_par,
            needs_recompute=recompute_flag,
            # Activation is stored in Fork instead,
            activation_stored=False,
            activation_reused=True))
          self._llm_block.append(Linear(
            "AttnBlock_Value",
            self.sys,
            self.exe.microbatch_size,
            self.app.seq_size,
            self.app.hidden,
            self.app.kv_groups * self.app.attn_size // self.exe.tensor_par,
            needs_recompute=recompute_flag,
            # Activation is stored in Fork instead,
            activation_stored=False,
            activation_reused=True))
        elif self.exe.attention_type == 'multiquery':
          # Multiquery attention uses the same K, V for all "heads" resulting in
          # smaller Wk and Wv, less matmul, faster inference
          self._llm_block.append(Linear(
            "AttnBlock_Key",
            self.sys,
            self.exe.microbatch_size,
            self.app.seq_size,
            self.app.hidden,
            self.app.attn_size,
            needs_recompute=recompute_flag,
            # Activation is stored in Fork instead,
            activation_stored=False,
            activation_reused=True))
          self._llm_block.append(Linear(
            "AttnBlock_Value",
            self.sys,
            self.exe.microbatch_size,
            self.app.seq_size,
            self.app.hidden,
            self.app.attn_size,
            needs_recompute=recompute_flag,
            # Activation is stored in Fork instead,
            activation_stored=False,
            activation_reused=True))
        else:
          raise self.Error(f"Wrong attention type: {self.exe.attention_type}")
    else:
      # LinearOverlapped already models QKV packing
      if self.exe.attention_type in ['multihead', 'groupquery']:
        assert not (self.exe.attention_type == 'multihead' and self.app.kv_groups != self.app.attn_heads), \
          "kv_groups must be equal to attn_heads for multihead attention"
        assert self.app.attn_heads % self.app.kv_groups == 0, "Number of heads must be divisible by kv_groups"
        k_packed = (self.app.attn_heads + 2 * self.app.kv_groups) * self.app.attn_size
        assert k_packed % self.exe.tensor_par == 0, "Packed QKV out-features must shard evenly across TP"
        self._llm_block.append(LinearOverlapped(
          "AttnBlock_QKV_AG",
          self.sys,
          self.exe.microbatch_size,
          self.app.seq_size,
          self.app.hidden,
          k_packed,
          self.exe.tensor_par_comm_type,
          self.exe.tensor_par,
          self.exe.tensor_par_net,
          self.exe.tensor_par,
          conjugate=False,
          tp_overlap=self.exe.tensor_par_overlap,
          needs_recompute=recompute_flag,
          needs_recomm=recompute_ag_flag))
      elif self.exe.attention_type == 'multiquery':
        self._llm_block.append(LinearOverlapped(
          "AttnBlock_Query_AG",
          self.sys,
          self.exe.microbatch_size,
          self.app.seq_size,
          self.app.hidden,
          self.app.attn_heads * self.app.attn_size,
          self.exe.tensor_par_comm_type,
          self.exe.tensor_par,
          self.exe.tensor_par_net,
          self.exe.tensor_par,
          conjugate=False,
          tp_overlap=self.exe.tensor_par_overlap,
          needs_recompute=recompute_flag,
          needs_recomm=recompute_ag_flag))
        self._llm_block.append(Fork(
          "AttnBlock_KV_Fork",
          self.sys,
          self._activation_size,
          2,
          needs_recompute=recompute_ag_flag,
          # With seq_par, we use activations from Comm layers to reflect that
          # they're split, otherwise we keep full size activations
          activation_stored=(not recompute_ag_flag)))
        self._llm_block.append(Linear(
          "AttnBlock_Key",
          self.sys,
          self.exe.microbatch_size,
          self.app.seq_size,
          self.app.hidden,
          self.app.attn_size,
          needs_recompute=recompute_flag,
          # Activation is stored in Fork instead,
          activation_stored=False,
          activation_reused=True))
        self._llm_block.append(Linear(
          "AttnBlock_Value",
          self.sys,
          self.exe.microbatch_size,
          self.app.seq_size,
          self.app.hidden,
          self.app.attn_size,
          needs_recompute=recompute_flag,
          # Activation is stored in Fork instead,
          activation_stored=False,
          activation_reused=True))
      else:
        raise self.Error(f"Wrong attention type: {self.exe.attention_type}")
    self._llm_block.append(BatchMatMulGQA(
      "AttnBlock_Multihead_Key_Query",
      self.sys,
      self.exe.microbatch_size * self.app.attn_heads // self.exe.tensor_par,
      self.app.seq_size,
      self.app.attn_size,
      self.app.seq_size,
      shared_operand='b',
      reuse_ratio=self.app.kv_groups / self.app.attn_heads,
      needs_recompute=recompute_attn_flag,
      output_stored=(not recompute_attn_flag)))
    self._llm_block.append(SoftMax(
      "AttnBlock_Multihead_SoftMax",
      self.sys,
      self.app.attn_heads // self.exe.tensor_par * \
        self.app.seq_size**2 * self.exe.microbatch_size,
      needs_recompute=recompute_attn_flag,
      output_stored=(not recompute_attn_flag)
    ))
    self._llm_block.append(DropOut(
      "AttnBlock_Multihead_DropOut",
      self.sys,
      self.app.attn_heads // self.exe.tensor_par * \
        self.app.seq_size**2 * self.exe.microbatch_size,
      needs_recompute=recompute_attn_flag,
      activation_stored=(not recompute_attn_flag)
    ))
    self._llm_block.append(BatchMatMulGQA(
      "AttnBlock_Multihead_Attn",
      self.sys,
      self.exe.microbatch_size * self.app.attn_heads // self.exe.tensor_par,
      self.app.seq_size,
      self.app.seq_size,
      self.app.attn_size,
      shared_operand='b',
      reuse_ratio=self.app.kv_groups / self.app.attn_heads,
      needs_recompute=recompute_flag))
    if self.exe.tensor_par_overlap == 'none':
      self._llm_block.append(Linear(
        "AttnBlock_MLP",
        self.sys,
        self.exe.microbatch_size,
        self.app.seq_size,
        self.app.attn_heads * self.app.attn_size // self.exe.tensor_par,
        self.app.hidden,
        needs_recompute=recompute_flag))
      self._llm_block.append(TPComm(
        "AttnBlock_G",
        self.sys,
        self._activation_size,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        # We only compute flops/mem analyzing this layers, comm analyzed later
        # This is conservative estimate that does not consider p2p_rs_ag
        # because we don't differentiate between edge and middle blocks here
        tensor_par_comm_type=self.exe.tensor_par_comm_type,
        conjugate=True,
        in_network_reduction=self.exe.in_network_reduction,
        needs_recomm=recompute_flag,
        # We don't store input to RS/AR
        activation_stored=False))
    else:
      self._llm_block.append(LinearOverlapped(
        "AttnBlock_MLP_RS",
        self.sys,
        self.exe.microbatch_size,
        self.app.seq_size,
        self.app.attn_heads * self.app.attn_size,
        self.app.hidden,
        self.exe.tensor_par_comm_type,
        self.exe.tensor_par,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        conjugate=True,
        tp_overlap=self.exe.tensor_par_overlap,
        needs_recompute=recompute_flag,
        needs_recomm=recompute_flag))
    self._llm_block.append(DropOut(
      "AttnBlock_DropOut",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag))
    self._llm_block.append(ElementWise(
      "AttnBlock_Residual",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag,
      # Activation is stored in Fork instead
      activation_stored=False,
      activation_reused=True))

  def _build_mlp_block(self):
    recompute_flag = self.exe.activation_recompute == "full"
    recompute_ag_flag = recompute_flag or self.exe.seq_par_ag_redo

    self._llm_block.append(Fork(
      "MlpBlock_Fork",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size, self._activation_size),
      2,
      needs_recompute=recompute_flag,
      # We account this activation when consider Residual and LayerNorm
      activation_stored=True))
    self._llm_block.append(LayerNorm(
      "MlpBlock_LayerNorm",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size, self._activation_size),
      self.app.hidden,
      needs_recompute=recompute_flag,
      # Activation is stored in Fork instead
      activation_stored=False,
      activation_reused=True))
    if self.exe.tensor_par_overlap == 'none':
      self._llm_block.append(TPComm(
        "MlpBlock_F",
        self.sys,
        # We only do compute/mem analyzing this layers, comm analyzed later
        # We keep extra mem buffer for comm, consider full tensor mem access
        # to be consistent with how much data comm moves/touches
        # This is conservative estimate that does not consider p2p_rs_ag
        # because we don't differentiate between edge and middle blocks here
        self._activation_size,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        tensor_par_comm_type=self.exe.tensor_par_comm_type,
        conjugate=False,
        in_network_reduction=self.exe.in_network_reduction,
        needs_recomm=recompute_ag_flag))
      self._llm_block.append(Linear(
        "MlpBlock_Mlp1",
        self.sys,
        self.exe.microbatch_size,
        self.app.seq_size,
        self.app.hidden,
        self.app.feedforward // self.exe.tensor_par,
        needs_recompute=recompute_flag,
        # With seq_par, we use activations from Comm layers to reflect that
        # they're split, otherwise we keep full size activations
        activation_stored=(not recompute_ag_flag)))
    else:
      self._llm_block.append(LinearOverlapped(
        "MlpBlock_Mlp1_AG",
        self.sys,
        self.exe.microbatch_size,
        self.app.seq_size,
        self.app.hidden,
        self.app.feedforward,
        self.exe.tensor_par_comm_type,
        self.exe.tensor_par,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        conjugate=False,
        tp_overlap=self.exe.tensor_par_overlap,
        needs_recompute=recompute_flag,
        needs_recomm=recompute_ag_flag))
    self._llm_block.append(GeLU(
      "MlpBlock_GeLU",
      self.sys,
      self.app.feedforward * self._batch_seq // self.exe.tensor_par,
      needs_recompute=recompute_flag,
      fused=self.exe.fused_activation))
    if self.exe.tensor_par_overlap == 'none':
      self._llm_block.append(Linear(
        "MlpBlock_Mlp2",
        self.sys,
        self.exe.microbatch_size,
        self.app.seq_size,
        self.app.feedforward // self.exe.tensor_par,
        self.app.hidden,
        needs_recompute=recompute_flag))
      self._llm_block.append(TPComm(
        "MlpBlock_G",
        self.sys,
        self._activation_size,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        # We only compute flops/mem analyzing this layers, comm analyzed later
        # This is conservative estimate that does not consider p2p_rs_ag
        # because we don't differentiate between edge and middle blocks here
        tensor_par_comm_type=self.exe.tensor_par_comm_type,
        conjugate=True,
        in_network_reduction=self.exe.in_network_reduction,
        needs_recomm=recompute_flag,
        # We don't store input to RS/AR
        activation_stored=False))
    else:
      self._llm_block.append(LinearOverlapped(
        "MlpBlock_Mlp2_RS",
        self.sys,
        self.exe.microbatch_size,
        self.app.seq_size,
        self.app.feedforward,
        self.app.hidden,
        self.exe.tensor_par_comm_type,
        self.exe.tensor_par,
        self.exe.tensor_par_net,
        self.exe.tensor_par,
        conjugate=True,
        tp_overlap=self.exe.tensor_par_overlap,
        needs_recompute=recompute_flag,
        needs_recomm=recompute_flag))
    self._llm_block.append(DropOut(
      "MlpBlock_DropOut",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag))
    self._llm_block.append(ElementWise(
      "MlpBlock_Residual",
      self.sys,
      pick(self.exe._sequence_par, self._seq_par_activation_size, self._activation_size),
      pick(self.exe._sequence_par, self._seq_par_activation_size, self._activation_size),
      needs_recompute=recompute_flag,
      # Activation is stored in Fork instead
      activation_stored=False,
      activation_reused=True))

  def compile(self, sys, exe):
    assert not self._compiled
    assert isinstance(exe, self.Execution)
    self.exe = exe
    assert isinstance(sys, System)
    self.sys = sys
    self._check_network_assignments()

    self.sys.set_datatype(self.exe.datatype)

    # If we have number of blocks not divisible by PP, we can allocate the
    # remainder of the blocks on the first num_block % PP Procs and block
    # "bubbles" on the last PP - (num_block % PP) Procs. To reflect that,
    # we round up blocks_per_proc. We report time for Proc0. In that case
    # its bubble time is `PP - (num_block % PP)` blocks shorter
    self._blocks_per_proc = self.app.num_blocks // self.exe.pipeline_par
    if self.app.num_blocks % self.exe.pipeline_par != 0:
      self._blocks_per_proc += 1
      self._bubble_reduction_blocks = self.exe.pipeline_par - (
        self.app.num_blocks % self.exe.pipeline_par)
    else:
      self._bubble_reduction_blocks = 0
    if self.exe.pipeline_interleaving > self._blocks_per_proc:
      raise self.Error('Pipeline interleaving must be less than or equal to '
                       'the number of blocks per processor')
    if self._blocks_per_proc % self.exe.pipeline_interleaving != 0:
      raise self.Error('Pipeline interleaving must be a factor value of the '
                       'number of blocks per processor')
    self._bytes_per_element = System.TypeSizes[self.exe.datatype]

    # Checks that enough blocks per processor exist if offloading is being
    # performed
    if (self.exe.weight_offload or self.exe.activations_offload or
        self.exe.optimizer_offload) and (self._blocks_per_proc <= 2):
      raise self.Error('Offloading requires each processor to handle at least'
                       ' 3 blocks')

    # A chunk is a set of blocks for microbatch before passing to the next
    # processor in the pipeline. Each chunk is modeled as a base
    # block that is repeated N-1 times and followed by 1 edge block.
    # Recommunication time is the same in both base and edge blocks.
    self._blocks_per_chunk = \
      self._blocks_per_proc // self.exe.pipeline_interleaving
    assert self._blocks_per_proc % self._blocks_per_chunk == 0, \
      f"PP interleaving should evenly divide {self._blocks_per_proc} blocks"
    self._chunks_per_proc = self._blocks_per_proc // self._blocks_per_chunk
    assert self._chunks_per_proc == self.exe.pipeline_interleaving, \
      "Number of chunks should be equal to pipeline_interleaving"
    self._baseblocks_per_chunk = self._blocks_per_chunk - 1
    self._edgeblocks_per_chunk = 1

    # Build model during the compilation step
    self._batch_seq = self.exe.microbatch_size * self.app.seq_size
    self._activation_size = self._batch_seq * self.app.hidden
    self._batch_seq_par = self._batch_seq // self.exe.tensor_par
    if self.exe._sequence_par or self.exe._pipeline_par_rs_ag:
      assert self._batch_seq % self.exe.tensor_par == 0, (
        f"We should split batch_seq={self._batch_seq} between"
        f" {self.exe.tensor_par} TP partitions evenly")
    self._seq_par_activation_size = self._batch_seq_par * self.app.hidden
    self._build_attn_block()
    self._build_mlp_block()
    for layer in self._llm_block:
      layer.set_bytes_per_element(self._bytes_per_element)
      if self.exe.optimizer_sharding:
        layer.shard_optimizer(self.exe.data_par)
    # Embedding / LM head derived sizes & PP embedding-grad AR time
    # Vocab-parallel token embedding:
    #   - The embedding matrix is sharded along the vocab dimension across TP.
    #   - The forward produces partial outputs that are summed via TP AR
    #     (we model the additional TP AR explicitly below), and the backward
    #     produces weight-grads that, when embeddings are tied with the output
    #     head, must be all-reduced across the PP group.
    #   - This "backward-embedding-all-reduce" is a PP AR.
    # We model PP AR here. We compute the per-rank shard size and its AR time on
    # the selected PP network and later add it once per batch iteration.
    if self.app.vocab_size % self.exe.tensor_par != 0:
      # Pad vocab so that vocab % TP == 0
      self._padded_vocab = ((self.app.vocab_size + self.exe.tensor_par - 1) //
                       self.exe.tensor_par) * self.exe.tensor_par
    else:
      self._padded_vocab = self.app.vocab_size
    self._embed_weight_elems_shard = (self._padded_vocab // self.exe.tensor_par) * self.app.hidden
    # If embeddings are untied, LM head is a separate matrix with the same shape
    self._lm_head_weight_elems_shard = 0 if self.app.tie_embeddings else self._embed_weight_elems_shard
    self._compiled = True

  def _check_network_assignments(self):
    used = [False] * self.sys.num_networks
    size = [1] * self.sys.num_networks

    assert self.exe.tensor_par_net < self.sys.num_networks
    assert self.exe.pipeline_par_net < self.sys.num_networks
    assert self.exe.data_par_net < self.sys.num_networks

    if self.exe.tensor_par > 1:
      used[self.exe.tensor_par_net] = True
      size[self.exe.tensor_par_net] *= self.exe.tensor_par
    self._tp_net = self.sys.get_network(self.exe.tensor_par_net)

    if self.exe.pipeline_par > 1:
      used[self.exe.pipeline_par_net] = True
      size[self.exe.pipeline_par_net] *= self.exe.pipeline_par
    self._pp_net = self.sys.get_network(self.exe.pipeline_par_net)

    if self.exe.data_par > 1:
      used[self.exe.data_par_net] = True
      size[self.exe.data_par_net] *= self.exe.data_par
    self._dp_net = self.sys.get_network(self.exe.data_par_net)

    for tier_used, tier_size, tier in zip(
        used, size, range(self.sys.num_networks)):
      if tier_used:
        if tier_size > self.sys.get_network(tier).size:
          raise self.Error(f'Network tier{tier} isn\'t big enough')
        if (self.sys.get_network(tier).must_be_filled and
            self.sys.get_network(tier).size % tier_size != 0):
          raise self.Error(f'Network tier{tier} isn\'t fully used')

  def _compute_block_stats(self):
    """
    This function computes the statistics for one microbatch on a single block.
    This only computes flops, flop time, and communication sizes. Since
    tensor and pipeline parallelism cause different communication operations to
    occur at the full batch level, the communication times are computed later.

    For one transformer block and one microbatch, it aggregates:
      Compute: FLOPs and their time for FW, BW (agrad + wgrad), recompute, and optimizer.
      Memory: bytes accessed and time per stage.
      TP comm: per‑stage sizes, link time, and exposed time for both base blocks and edge blocks
        (used later by the pipeline schedule).
      PP comm size for FW and BW (time is not computed here).
      Memory footprint: weights, grads, activations (working vs. stored), optimizer state.
      Overlap metric: the TP bandwidth required to fully hide TP collectives under compute.
    All of these are saved in self._block_* / self._baseblock_* / self._edgeblock_* fields for
      subsequent top‑level aggregation.
    """

    # If we’re training with full activation recompute, the checkpointed bytes per block are
    # act_size * bytes_per_element, otherwise 0. This represents the per‑block activation
    # explicitly kept when recomputing everything else during BW.
    if self.exe.training and self.exe.activation_recompute == "full":
      self._block_act_checkpoint_size = \
        self._activation_size * self._bytes_per_element
    else:
      self._block_act_checkpoint_size = 0

    # Initializes values to zero for accumulation in layer loop
    # All of these are per rank, for one microbatch, on a single
    # transformer block; they are sums over all layers in the block.
    # Forward compute & memory
    self._block_fw_flops = 0                # Total forward FLOPs
    self._block_fw_flops_time = 0           # Sum of compute‑only time for FW (FLOPs / matrix or vector throughput)
    self._block_fw_mem_accessed = 0         # Sum of FW bytes accessed
    self._block_fw_mem_time = 0             # Sum of memory‑only time for FW, i.e., mem_bytes / mem_bw
    self._block_fw_time = 0                 # Sum of processing time for FW per block, compute+memory and overlapped time

    # Forward TP comm (sizes & times)
    self._baseblock_fw_tp_size = 0          # Total TP FW comm size (AG/RS/AR) for base/edge blocks. Non‑overlapped layers contribute 0
    self._edgeblock_fw_tp_size = 0
    self._baseblock_fw_tp_time = 0          # Link time (+ potential on‑device reduction time) for FW TP comm
    self._edgeblock_fw_tp_time = 0
    self._baseblock_fw_tp_time_exposed = 0  # Exposed (non‑hidden) part of FW TP comm under the layer’s tiling/overlap model
    self._edgeblock_fw_tp_time_exposed = 0

    # Memory footprints (capacity)
    self._block_weight_space = 0            # Sum of all parameter bytes for this block
    self._block_act_working_space = 0       # Sum of working activation bytes that are not flagged as “reused elsewhere”
    self._block_act_working_peak = 0        # Peak working-set across layers (realistic for PP=1)
    self._block_act_storage_space = 0       # Sum of stored activations that persist for BW (before global recompute overrides)

    # Recompute (re‑executed FW during BW when checkpointing)
    # We use this block for self.exe.training, but initialize anyway
    self._block_re_flops = 0                # Total recompute FLOPs. On each layer with needs_recompute=True, the code adds the cumulative FW FLOPs so far—modeling that FW prefix is re‑run during BW.
    self._block_re_flops_time = 0           # Sum of compute‑only time for those recomputed FLOPs
    self._block_re_mem_accessed = 0         # Sum of memory bytes re‑accessed during recompute
    self._block_re_mem_time = 0             # Sum of memory‑only time for recompute
    self._block_re_time = 0                 # Sum of processing time for the recomputed FW segments

    # TP re-communication during weight-grad (e.g., redo AllGather)
    # It is about sequence-parallel RS/AG and activation (re)compute
    # colliding with what the weight-grad matmul actually needs.
    self._baseblock_recomm_size = 0         # Total extra TP W‑grad comm size caused by “redo AG” (sequence parallel) or similar
    self._edgeblock_recomm_size = 0
    self._baseblock_recomm_time = 0         # Link time for that re‑communication
    self._edgeblock_recomm_time = 0
    self._baseblock_recomm_time_exposed = 0 # Exposed portion of the re‑communication time
    self._edgeblock_recomm_time_exposed = 0

    # Backward – activation‑gradient (A‑grad) compute & memory
    self._block_agrad_flops = 0             # Total BW A‑grad FLOPs
    self._block_agrad_flops_time = 0        # Sum of compute‑only time for A‑grad
    self._block_agrad_mem_accessed = 0      # Sum of A‑grad bytes accessed
    self._block_agrad_mem_time = 0          # Sum of memory‑only time for A‑grad
    self._block_agrad_time = 0              # Sum of processing time for A‑grad per block (compute + memory with any TP overlap)

    # Backward TP communication (sizes & times) — A‑grad
    self._baseblock_agrad_tp_size = 0         # Total TP BW (A‑grad) comm size (RS/AG/AR) for base/edge blocks
    self._edgeblock_agrad_tp_size = 0
    self._baseblock_agrad_tp_time = 0         # Link time (+ potential on‑device reductions) for BW TP comm
    self._edgeblock_agrad_tp_time = 0
    self._baseblock_agrad_tp_time_exposed = 0 # Exposed (not hidden) BW TP comm time
    self._edgeblock_agrad_tp_time_exposed = 0

    # Backward – weight‑gradient(W‑grad) compute & memory
    self._block_wgrad_flops = 0             # Total BW weight‑grad FLOPs. Parameter‑free layers contribute 0
    self._block_wgrad_flops_time = 0        # Sum of compute‑only time for W‑grad
    self._block_wgrad_mem_accessed = 0      # Sum of W‑grad bytes accessed, or 0 for layers without parameters
    self._block_wgrad_mem_time = 0          # Sum of memory‑only time for W‑grad
    self._block_wgrad_time = 0              # Sum of processing time for W‑grad per block

    # Optimizer (Adam) compute & memory
    self._block_optim_flops = 0             # Total optimizer step FLOPs (Adam), modeled as 11 * weight_grads / DP_shard
    self._block_optim_flops_time = 0        # Sum of compute‑only time for optimizer (vector throughput path).
    self._block_optim_mem_accessed = 0      # Total bytes read/written by optimizer state (FP32 moments + optional FP32 master weights), sharded by DP if enabled
    self._block_optim_mem_time = 0          # Sum of memory‑only time for optimizer
    self._block_optim_time = 0              # Sum of processing time for optimizer per block

    # Gradient/optimizer capacity footprints
    self._block_weight_grad_space = 0       # Total weight‑grad storage, using low‑precision and DP sharding (what’s actually kept for communication/accumulation)
    self._block_weight_grad_space_no_sharding = 0  # The same weight‑grad storage but as unsharded FP32
    self._block_act_grad_space = 0          # Total activation‑gradient storage footprint across layers
    self._block_optimizer_space = 0         # Total optimizer state capacity (FP32 moments + optional FP32 master weights), already divided by the DP shard count if sharding is enabled

    # TP overlap bandwidth requirement
    self._tp_bw_overlap_req = 0             # The maximum per‑tile link bandwidth required (across FW/BW and base/edge) to fully hide TP collectives under compute. Driven primarily by LinearOverlapped’s tiling model

    # The loop accumulates per‑microbatch, per‑block stats by summing each layer’s contribution
    for layer in self._llm_block:
      # Forward pass (FW): compute, memory, overlapped time
      # Adds (a) pure FLOPs, (b) pure compute time (FLOPs / throughput), (c) pure memory
      # bytes & time, and (d) processing time (compute + memory with any modeled overlap)
      # for FW. For layers like LinearOverlapped, it also bakes in TP comm overlap and sets
      # up “exposed net” bookkeeping
      self._block_fw_flops += layer.get_fw_flops()
      self._block_fw_flops_time += layer.compute_flops_time("fw")
      self._block_fw_mem_accessed += layer.get_fw_mem_accessed()
      self._block_fw_mem_time += layer.compute_mem_time("fw")
      self._block_fw_time += layer.compute_processing_time("fw")

      # FW TP comm: sizes, link time, exposed time, required bandwidth
      self._baseblock_fw_tp_size += layer.get_comm_bytes("fw", baseblock=True)  # returns the collective payload (e.g., AG/RS/AR) in bytes; non‑TP layers return 0
      self._edgeblock_fw_tp_size += layer.get_comm_bytes("fw", baseblock=False)
      self._baseblock_fw_tp_time += layer.compute_net_time("fw", baseblock=True)  # asks the selected network model for the collective’s time
      self._edgeblock_fw_tp_time += layer.compute_net_time("fw", baseblock=False)
      self._baseblock_fw_tp_time_exposed += layer.get_exposed_net_time("fw", baseblock=True)  # the non‑hidden remainder after the layer’s overlap schedule is applied
      self._edgeblock_fw_tp_time_exposed += layer.get_exposed_net_time("fw", baseblock=False)

      self._tp_bw_overlap_req = max(self._tp_bw_overlap_req, layer.get_required_bandwidth("fw", baseblock=True))  # the per‑tile link bandwidth needed to fully hide the TP collective under compute; the loop keeps the max across all FW/BW stages and base/edge
      self._tp_bw_overlap_req = max(self._tp_bw_overlap_req, layer.get_required_bandwidth("fw", baseblock=False))

      # Training‑only: checkpoint recompute and “re‑communication”
      if self.exe.training:
        # Checkpoint recompute (FW re‑execution during BW)
        if layer.get_recompute_flag():
          # Model “re‑running FW from the last checkpoint up to here” during BW.
          # This works when only boundary layers set needs_recompute=True (e.g.,
          # a Fork that stores and later reuses)
          self._block_re_flops += self._block_fw_flops
          self._block_re_flops_time += self._block_fw_flops_time
          self._block_re_mem_accessed += self._block_fw_mem_accessed
          self._block_re_mem_time += self._block_fw_mem_time
          self._block_re_time += layer.compute_processing_time("fw")

        # “Re‑communication” during W‑grad (e.g., redo AG for RS/AG)
        if layer.get_recomm_flag():
          # Some TP modes (sequence parallel, RS/AG) need to redo an AllGather
          # during W‑grad. Layers that require it raise needs_recomm=True; this
          # block captures the extra bytes/time (base & edge).
          self._baseblock_recomm_size += layer.get_comm_bytes("wgrad", baseblock=True)
          self._edgeblock_recomm_size += layer.get_comm_bytes("wgrad", baseblock=False)
          self._baseblock_recomm_time += layer.compute_net_time("wgrad", baseblock=True)
          self._edgeblock_recomm_time += layer.compute_net_time("wgrad", baseblock=False)
          self._baseblock_recomm_time_exposed += layer.get_exposed_net_time("wgrad", baseblock=True)
          self._edgeblock_recomm_time_exposed += layer.get_exposed_net_time("wgrad", baseblock=False)

        # Backward – activation gradients (A‑grad) and its TP comm
        # Mirrors FW accumulation but for A‑grad (backprop through activations).
        self._block_agrad_flops += layer.get_agrad_flops()
        self._block_agrad_flops_time += layer.compute_flops_time("agrad")
        self._block_agrad_mem_accessed += layer.get_agrad_mem_accessed()
        self._block_agrad_mem_time += layer.compute_mem_time("agrad")
        self._block_agrad_time += layer.compute_processing_time("agrad")

        self._baseblock_agrad_tp_size += layer.get_comm_bytes("agrad", baseblock=True)
        self._edgeblock_agrad_tp_size += layer.get_comm_bytes("agrad", baseblock=False)
        self._baseblock_agrad_tp_time += layer.compute_net_time("agrad", baseblock=True)
        self._edgeblock_agrad_tp_time += layer.compute_net_time("agrad", baseblock=False)
        self._baseblock_agrad_tp_time_exposed += layer.get_exposed_net_time("agrad", baseblock=True)
        self._edgeblock_agrad_tp_time_exposed += layer.get_exposed_net_time("agrad", baseblock=False)

        self._tp_bw_overlap_req = max(self._tp_bw_overlap_req, layer.get_required_bandwidth("agrad", baseblock=True))
        self._tp_bw_overlap_req = max(self._tp_bw_overlap_req, layer.get_required_bandwidth("agrad", baseblock=False))

        # Backward – weight gradients (W‑grad) and Optimizer
        # W‑grad: Adds math/bytes/time for computing weight gradients (zero for parameter‑free layers).
        self._block_wgrad_flops += layer.get_wgrad_flops()
        self._block_wgrad_flops_time += layer.compute_flops_time("wgrad")
        self._block_wgrad_mem_accessed += layer.get_wgrad_mem_accessed()
        self._block_wgrad_mem_time += layer.compute_mem_time("wgrad")
        self._block_wgrad_time += layer.compute_processing_time("wgrad")

        # Optimizer: Adds the Adam step’s FLOPs and the optimizer state bytes (vector throughput path)
        self._block_optim_flops += layer.get_optim_step_flops()
        self._block_optim_flops_time += layer.compute_flops_time("optim")
        self._block_optim_mem_accessed += layer.get_optim_step_mem_accessed()
        self._block_optim_mem_time += layer.compute_mem_time("optim")
        self._block_optim_time += layer.compute_processing_time("optim")

      # Footprint (memory capacity, bytes) per block
      self._block_weight_space += layer.get_weight()              # parameters
      if not layer.reuses_activation():
        # Cache the activation size once and track a *peak* working-set for PP=1:
        # activations from different layers don’t live at the same time, so summing would
        # double-count; the resident footprint is the max per-layer working activation.
        act_size_ = layer.get_activation()
        self._block_act_working_space += act_size_                # working activations
        self._block_act_working_peak = max(self._block_act_working_peak, act_size_)  # peak working-set for PP=1
      self._block_act_storage_space += layer.get_activation()     # stored activations

      if self.exe.training:
        if not layer.stores_output():                             # do not store the output
          self._block_act_storage_space -= layer.get_output()
        if not layer.stores_activation():                         # do not store activations
          self._block_act_storage_space -= layer.get_activation()
        self._block_weight_grad_space += layer.get_weight_grad()  # weight gradients (sharded low-precision)
        self._block_weight_grad_space_no_sharding += layer.get_weight_grad(sharded=False) # weight gradients (unsharded FP32)
        self._block_act_grad_space += layer.get_activation_grad() # activation gradients
        self._block_optimizer_space += layer.get_optimizer()      # optimizer state

      # Debug logging for this layer (a layer in the current block)
      self.log.debug("%s %s %s", layer.name, 'Recompute flag:',
                     str(layer.get_recompute_flag()))
      self.log.debug("%s %s %s", layer.name, 'Recomm flag:',
                     str(layer.get_recomm_flag()))
      self.log.debug("%s %s %s", layer.name, 'Stores activation:',
                     str(layer.stores_activation()))
      self.log.debug("%s %s %s", layer.name, 'Reuses activation:',
                     str(layer.reuses_activation()))
      self.log.debug("%s %s %s", layer.name, 'Stores output:',
                     str(layer.stores_output()))
      self.log.debug("%s %s %s", layer.name, 'FW flops:',
                     human_format(layer.get_fw_flops(), 'flops'))
      self.log.debug("%s %s %s", layer.name, 'FW num inputs:',
                     human_format(layer.inputs_size, 'base2'))
      self.log.debug("%s %s %s", layer.name, 'FW num output:',
                     human_format(layer.output_size, 'base2'))
      self.log.debug("%s %s %s", layer.name, 'FW num weights:',
                     human_format(layer.weight_space, 'base2'))
      self.log.debug("%s %s %s", layer.name, 'FW mem:',
                     human_format(layer.get_fw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW baseblock comm tile size:',
                     human_format(layer.get_comm_tile("fw", baseblock=True),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW edgeblock comm tile size:',
                     human_format(layer.get_comm_tile("fw", baseblock=False),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW baseblock comm size:',
                     human_format(layer.get_comm_bytes("fw", baseblock=True),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW edgeblock comm size:',
                     human_format(layer.get_comm_bytes("fw", baseblock=False),
                     'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'FW net link time:',
                     layer.compute_net_time("fw"))
      self.log.debug("%s %s %.3e", layer.name, 'FW net exposed time:',
                     layer.get_exposed_net_time("fw"))
      self.log.debug("%s %s %.3e", layer.name, 'FW time:',
                     layer.compute_processing_time("fw"))
      self.log.debug("%s %s %s", layer.name, 'BW flops:',
                     human_format(
                      layer.get_agrad_flops() + layer.get_wgrad_flops(),
                      'flops'))
      self.log.debug("%s %s %s", layer.name, 'BW num Wgrads:',
                     human_format(layer.weight_grads, 'base2'))
      self.log.debug("%s %s %s", layer.name, 'BW num Agrads:',
                     human_format(layer.activation_grads, 'base2'))
      self.log.debug("%s %s %s", layer.name, 'BW num Igrads:',
                     human_format(layer.inputs_size, 'base2'))
      self.log.debug("%s %s %s", layer.name, 'BW mem:',
                     human_format(
                      layer.get_agrad_mem_accessed() +
                      layer.get_wgrad_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW baseblock comm tile size:',
                     human_format(layer.get_comm_tile("agrad", baseblock=True),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW edgeblock comm tile size:',
                     human_format(layer.get_comm_tile("agrad", baseblock=False),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW baseblock comm size:',
                     human_format(layer.get_comm_bytes("agrad", baseblock=True),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW edgeblock comm size:',
                     human_format(layer.get_comm_bytes("agrad", baseblock=False),
                     'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'BW net link time:',
                     layer.compute_net_time("agrad"))
      self.log.debug("%s %s %.3e", layer.name, 'BW net exposed time:',
                     layer.get_exposed_net_time("agrad"))
      self.log.debug("%s %s %.3e", layer.name, 'BW time:',
                     layer.compute_processing_time("agrad") +
                     layer.compute_processing_time("wgrad"))
      self.log.debug("%s %s %s", layer.name, 'Recomm baseblock comm tile size:',
                     human_format(layer.get_comm_tile("wgrad", baseblock=True),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Recomm edgeblock comm tile size:',
                     human_format(layer.get_comm_tile("wgrad", baseblock=False),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Recomm baseblock comm size:',
                     human_format(layer.get_comm_bytes("wgrad", baseblock=True),
                     'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Recomm edgeblock comm size:',
                     human_format(layer.get_comm_bytes("wgrad", baseblock=False),
                     'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'Recomm net link time:',
                     layer.compute_net_time("wgrad"))
      self.log.debug("%s %s %.3e", layer.name, 'Recomm net exposed time:',
                     layer.get_exposed_net_time("wgrad"))
      self.log.debug("%s %s %s", layer.name, 'Optim flops:',
                     human_format(layer.get_optim_step_flops(), 'flops'))
      self.log.debug("%s %s %s", layer.name, 'BW Optimizer size:',
                     human_format(layer.get_optimizer(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Optim mem:',
                     human_format(layer.get_optim_step_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'Optim time:',
                     layer.compute_processing_time("optim"))
      self.log.debug("%s %s %s", layer.name, 'Recompute:',
                     layer.get_recompute_flag())
      self.log.debug("%s %s %s", layer.name, 'Recompute mem saving:',
                     human_format(layer.stores_output() * \
                       layer.get_output(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Weight:',
                     human_format(layer.get_weight(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Act:',
                     human_format(layer.get_activation(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Weight grad:',
                     human_format(layer.get_weight_grad(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Act grad:',
                     human_format(layer.get_activation_grad(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Optim:',
                     human_format(layer.get_optimizer(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight:',
                     human_format(self._block_weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act Working space:',
                     human_format(self._block_act_working_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act Storage space:',
                     human_format(self._block_act_storage_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight grad:',
                     human_format(self._block_weight_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act grad:',
                     human_format(self._block_act_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Optim:',
                     human_format(self._block_optimizer_space, 'bytes'))

    # outside layer loop here
    if self.exe.activation_recompute == 'full':
      self._block_act_storage_space = 0

    # Sets the PP communication size
    if self.exe.pipeline_par > 1:
      if self.exe._pipeline_par_rs_ag:
        self._block_fw_pp_size = self._seq_par_activation_size * \
          self._bytes_per_element
      else:
        self._block_fw_pp_size = self._activation_size * \
          self._bytes_per_element
    else:
      self._block_fw_pp_size = 0

    # When training, BW sizes for TP and PP are same as FW
    if self.exe.training:
      self._block_bw_pp_size = self._block_fw_pp_size
    else:
      self._block_bw_pp_size = 0

    self.log.debug("%s %s", 'TP comm FW baseblock size:',
                   human_format(self._baseblock_fw_tp_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm FW edgeblock size:',
                   human_format(self._edgeblock_fw_tp_size, 'bytes'))
    self.log.debug("%s %s", 'PP comm FW size:',
                   human_format(self._block_fw_pp_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm BW baseblock size:',
                   human_format(self._baseblock_agrad_tp_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm BW edgeblock size:',
                   human_format(self._edgeblock_agrad_tp_size, 'bytes'))
    self.log.debug("%s %s", 'PP comm BW size:',
                   human_format(self._block_bw_pp_size, 'bytes'))
    self.log.debug("%s %s", 'TP recomm baseblock size:',
                   human_format(self._baseblock_recomm_size, 'bytes'))
    self.log.debug("%s %s", 'TP recomm edgeblock size:',
                   human_format(self._edgeblock_recomm_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm required bandwidth for tiled overlap:',
                   human_format(self._tp_bw_overlap_req, 'bandwidth'))

  def _micros_in_flight(self) -> float:
    """
    Resident microbatches for capacity accounting.
    Under 1F1B, at any instant a stage holds at most min(M, PP) micros.
    Do not include warm-up/flush stretch factors here (those are time effects).
    """
    pp = int(getattr(self.exe, "pipeline_par", 1))
    if pp <= 1: return 1.0
    num_micros = int(getattr(self.exe, "_num_microbatches", 1))
    return float(min(num_micros, pp))

  def _compute_batch_stats(self):
    """
    This function computes the statistics for a full batch. This uses the per
    microbatch per block statistics from the prior function (see above).
    """

    # Helper: bytes for gradient reductions (DP / PP embedding-grad)
    def _grad_comm_bytes(num_elems: int) -> int:
      if self.exe.grad_reduce_in_fp32:
        return num_elems * System.TypeSizes['float32']
      else:
        return num_elems * self._bytes_per_element

    # PP AR for tied embeddings' weight-grads (first <-> last stage only).
    # Barrier to all PP stages, charge it once per batch (after grad accumulation).
    self._pp_embed_grad_time_per_batch = 0.0
    if self.exe.training and self.exe.pipeline_par > 1 and self.app.tie_embeddings:
      embed_grad_bytes = _grad_comm_bytes(self._embed_weight_elems_shard)
      # Only the two owner stages participate in the all-reduce, so group size is 2.
      self._pp_embed_grad_time_per_batch = self._pp_net.time('all_reduce', embed_grad_bytes, 2)

    # TP comm for embedding / LM head
    # Only owner pipeline stages pay this cost:
    #  - Embedding owner: first stage always; last stage also when embeddings are tied
    #      for capacity/optimizer/PP-AR, but TP FW AR for embedding happens only on the
    #      first stage.
    #  - LM head owner: last stage (both tied and untied; compute lives there).
    self._tp_embed_fw_time_per_micro = 0.0
    self._tp_embed_bw_time_per_micro = 0.0
    self._tp_lm_head_fw_time_per_micro = 0.0
    self._tp_lm_head_bw_time_per_micro = 0.0
    if self.exe.tensor_par > 1:
      owner_first = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'first')
      owner_last = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'last')
      act_bytes = self._activation_size * self._bytes_per_element
      # Vocab-parallel embedding: FW needs TP AR of partial outputs; BW needs no TP collective.
      if owner_first:
        self._tp_embed_fw_time_per_micro = self._tp_net.time('all_reduce', act_bytes, self.exe.tensor_par)
        self._tp_embed_bw_time_per_micro = 0.0
      # LM head (on the last stage):
      if owner_last:
        # Vocab-parallel LM head with parallel cross-entropy.
        # FW: no TP collective. BW: grad-input needs TP AR.
        self._tp_lm_head_fw_time_per_micro = 0.0
        if self.exe.training:
          self._tp_lm_head_bw_time_per_micro = self._tp_net.time('all_reduce', act_bytes, self.exe.tensor_par)
          # Vocab-parallel cross-entropy forward needs two small ARs over TP
          ce_fw_bytes = self.exe.microbatch_size * self.app.seq_size * System.TypeSizes['float32']
          self._tp_lm_head_fw_time_per_micro += 2 * self._tp_net.time('all_reduce', ce_fw_bytes, self.exe.tensor_par)
        else:
          self._tp_lm_head_bw_time_per_micro = 0.0

    # Total stats for compute and memory
    mult = self._blocks_per_proc * self.exe._num_microbatches
    self._fw_flops = mult * self._block_fw_flops
    self._fw_flops_time = mult * self._block_fw_flops_time
    self._fw_mem_accessed = mult * self._block_fw_mem_accessed
    self._fw_mem_time = mult * self._block_fw_mem_time
    self._fw_time = mult * self._block_fw_time
    self._re_flops = mult * self._block_re_flops
    self._re_flops_time = mult * self._block_re_flops_time
    self._re_mem_accessed = mult * self._block_re_mem_accessed
    self._re_mem_time = mult * self._block_re_mem_time
    self._re_time = mult * self._block_re_time
    self._agrad_flops = mult * self._block_agrad_flops
    self._agrad_flops_time = mult * self._block_agrad_flops_time
    self._agrad_mem_accessed = mult * self._block_agrad_mem_accessed
    self._agrad_mem_time = mult * self._block_agrad_mem_time
    self._agrad_time = mult * self._block_agrad_time
    self._wgrad_flops = mult * self._block_wgrad_flops
    self._wgrad_flops_time = mult * self._block_wgrad_flops_time
    self._wgrad_mem_accessed = mult * self._block_wgrad_mem_accessed
    self._wgrad_mem_time = mult * self._block_wgrad_mem_time
    self._wgrad_time = mult * self._block_wgrad_time

    # The optimizer stats are multiplied only by self._blocks_per_proc because
    # the optimizer step is executed once per batch.
    self._optim_flops = self._blocks_per_proc * self._block_optim_flops
    self._optim_flops_time = self._blocks_per_proc * self._block_optim_flops_time
    self._optim_mem_accessed = self._blocks_per_proc * self._block_optim_mem_accessed
    self._optim_mem_time = self._blocks_per_proc * self._block_optim_mem_time
    self._optim_time = self._blocks_per_proc * self._block_optim_time

    # Total TP communication time (link time and exposed time)
    # It builds batch totals for TP collectives by summing base‑block and
    # edge‑block contributions across all chunks and microbatches.
    tp_fw_comm_time = self.exe._num_microbatches * self._chunks_per_proc * (
      (self._baseblocks_per_chunk * self._baseblock_fw_tp_time) +
      (self._edgeblocks_per_chunk * self._edgeblock_fw_tp_time))
    tp_fw_comm_time_exposed = \
      self.exe._num_microbatches * self._chunks_per_proc * (
        (self._baseblocks_per_chunk * self._baseblock_fw_tp_time_exposed) +
        (self._edgeblocks_per_chunk * self._edgeblock_fw_tp_time_exposed))
    tp_bw_comm_time = self.exe._num_microbatches * self._chunks_per_proc * (
      self._baseblocks_per_chunk * self._baseblock_agrad_tp_time +
      self._edgeblocks_per_chunk * self._edgeblock_agrad_tp_time)
    tp_bw_comm_time_exposed = \
      self.exe._num_microbatches * self._chunks_per_proc * (
        self._baseblocks_per_chunk * self._baseblock_agrad_tp_time_exposed +
        self._edgeblocks_per_chunk * self._edgeblock_agrad_tp_time_exposed)
    tp_recomm_time = self.exe._num_microbatches * self._chunks_per_proc * (
      (self._baseblocks_per_chunk * self._baseblock_recomm_time) +
      (self._edgeblocks_per_chunk * self._edgeblock_recomm_time))
    tp_recomm_time_exposed = \
      self.exe._num_microbatches * self._chunks_per_proc * (
        (self._baseblocks_per_chunk * self._baseblock_recomm_time_exposed) +
        (self._edgeblocks_per_chunk * self._edgeblock_recomm_time_exposed))

    def _gemm_processing_time(batch, m, n, k, read_bytes, write_bytes):
      """
      Returns processing time for a GEMM under the system model:
      - flops_time uses matrix throughput (accounts for shape/efficiency);
      - mem_time uses mem1 throughput (size-dependent efficiency);
      - combined via system processing mode (roofline/no_overlap).
      """
      flops = 2.0 * batch * m * n * k
      thr_mat = self.sys.get_matrix_throughput(batch=batch, m=m, n=n, k=k)
      flops_time = 0.0 if thr_mat <= 0.0 else (flops / thr_mat)
      bytes_total = read_bytes + write_bytes
      thr_mem = self.sys.get_mem1_throughput(bytes_total)
      mem_time = 0.0 if thr_mem <= 0.0 else (bytes_total / thr_mem)
      return self.sys.get_processing_time(flops_time, mem_time)

    self._lm_head_fw_comp_time_per_micro = 0.0
    self._lm_head_bw_comp_time_per_micro = 0.0
    assert self._padded_vocab > 0
    V_shard = self._padded_vocab // self.exe.tensor_par
    B = self.exe.microbatch_size
    S = self.app.seq_size
    H = self.app.hidden
    e = self._bytes_per_element
    # Forward GEMM: [B*S, H] x [H, V_shard] -> [B*S, V_shard]
    #   read X + W, write Y
    fw_reads = (B * S * H + H * V_shard) * e
    fw_writes = (B * S * V_shard) * e
    self._lm_head_fw_comp_time_per_micro = _gemm_processing_time(
      batch=B, m=S, n=V_shard, k=H, read_bytes=fw_reads, write_bytes=fw_writes
    )
    # Backward:
    # 1) dX GEMM: [B*S, V_shard] x [V_shard, H] -> [B*S, H]
    bx_reads = (B * S * V_shard + H * V_shard) * e
    bx_writes = (B * S * H) * e
    t_dx = _gemm_processing_time(
      batch=B, m=S, n=H, k=V_shard, read_bytes=bx_reads, write_bytes=bx_writes
    )
    # 2) dW GEMM (always executed; when tied, this contributes to the shared embedding weight):
    #    [H, B*S] x [B*S, V_shard] -> [H, V_shard]
    dw_reads = (B * S * H + B * S * V_shard) * e
    dw_writes = (H * V_shard) * e
    # Use batch=1 to reflect a single GEMM on (H, V_shard, B*S).
    t_dw = _gemm_processing_time(
      batch=1, m=H, n=V_shard, k=B*S, read_bytes=dw_reads, write_bytes=dw_writes
    )
    if self.exe.training:
      self._lm_head_bw_comp_time_per_micro = t_dx + t_dw
    else:
      self._lm_head_bw_comp_time_per_micro = 0.0

    # LM head per-micro FLOPs & bytes (for reporting only)
    lm_fw_flops_per_micro = 2.0 * B * S * H * V_shard  # [B*S,H]x[H,V]
    lm_dx_flops_per_micro = 2.0 * B * S * H * V_shard  # [B*S,V]x[V,H]
    lm_dw_flops_per_micro = 2.0 * H * V_shard * (B * S)  # [H,B*S]x[B*S,V]

    lm_fw_bytes_per_micro = fw_reads + fw_writes
    lm_dx_bytes_per_micro = bx_reads + bx_writes
    lm_dw_bytes_per_micro = dw_reads + dw_writes

    P_eff = self.exe.pipeline_par * self.exe.pipeline_interleaving
    owners = [i % self.exe.pipeline_par for i in range(P_eff)]

    # Per-boundary P2P times (FW/BW)
    t_fw = []  # len = max(P_eff-1, 0)
    t_bw = []
    if P_eff > 1:
      for i in range(P_eff - 1):
        src = owners[i]
        dst = owners[i + 1]
        if src == dst:
          # Adjacent virtual stages on same rank ⇒ no P2P for this boundary
          t_fw.append(0.0)
          t_bw.append(0.0)
          continue
        net = None
        for net_ in self.sys.networks:
          if src // net_.size == dst // net_.size:
            net = net_
            break
        assert net is not None
        t_fw.append(net.time('p2p', self._block_fw_pp_size, 2))
        t_bw.append(net.time('p2p', self._block_bw_pp_size, 2))

    # Aggregate to per-stage PP times (enter + leave)
    # For stage s:
    #   FW: enter = t_fw[s-1] (if s>0), leave = t_fw[s] (if s<P_eff-1)
    #   BW: enter = t_bw[s],   leave = t_bw[s-1] (if s>0)
    pp_fw_stage = [0.0] * P_eff
    pp_bw_stage = [0.0] * P_eff
    if P_eff > 0:
      for s in range(P_eff):
        fw_in = t_fw[s - 1] if (s > 0) and t_fw else 0.0
        fw_out = t_fw[s] if (s < P_eff - 1) and t_fw else 0.0
        bw_in = t_bw[s] if (s < P_eff - 1) and t_bw else 0.0
        bw_out = t_bw[s - 1] if (s > 0) and t_bw else 0.0
        pp_fw_stage[s] = fw_in + fw_out
        pp_bw_stage[s] = bw_in + bw_out

    # Owner-local compute extras: LM head only lives on the last real stage.
    # We add the extra compute time into the corresponding virtual stage period.
    if P_eff > 0 and self.exe.pipeline_par > 1:
      for s in range(P_eff):
        if owners[s] == (self.exe.pipeline_par - 1):
          pp_fw_stage[s] += self._lm_head_fw_comp_time_per_micro
          if self.exe.training:
            pp_bw_stage[s] += self._lm_head_bw_comp_time_per_micro

    # Choose this-rank's stage-period PP cost for block-level injection
    # consider interleaving by taking, for each owner (real rank), the max
    # across its virtual stages; since we don't know our rank here, use
    # the conservative max across owners.
    pp_fw_per_owner = [0.0] * self.exe.pipeline_par
    pp_bw_per_owner = [0.0] * self.exe.pipeline_par
    for s in range(P_eff):
      o = owners[s]
      if pp_fw_stage[s] > pp_fw_per_owner[o]:
        pp_fw_per_owner[o] = pp_fw_stage[s]
      if pp_bw_stage[s] > pp_bw_per_owner[o]:
        pp_bw_per_owner[o] = pp_bw_stage[s]

    # Batch-level KPI totals
    chunk_fw_pp_time = max(pp_fw_per_owner) if pp_fw_per_owner else 0.0
    pp_fw_comm_time = self.exe._num_microbatches * sum(t_fw)
    if self.exe.training:
      chunk_bw_pp_time = max(pp_bw_per_owner) if pp_bw_per_owner else 0.0
      pp_bw_comm_time = self.exe._num_microbatches * sum(t_bw)
    else:
      chunk_bw_pp_time = 0.0
      pp_bw_comm_time = 0.0

    self._tp_comm_time_link = tp_fw_comm_time + tp_bw_comm_time
    self._tp_comm_time_exposed = tp_fw_comm_time_exposed + tp_bw_comm_time_exposed
    # Add TP comm for embedding / LM head
    tp_extra_link = self.exe._num_microbatches * (
        self._tp_embed_fw_time_per_micro + self._tp_lm_head_fw_time_per_micro)
    if self.exe.training:
      tp_extra_link += self.exe._num_microbatches * (
          self._tp_embed_bw_time_per_micro + self._tp_lm_head_bw_time_per_micro
      )
    self._tp_comm_time_link += tp_extra_link
    self._tp_comm_time_exposed += tp_extra_link
    self._recomm_time_link = tp_recomm_time
    self._recomm_time_exposed = tp_recomm_time_exposed
    self._pp_comm_time_link = pp_fw_comm_time + pp_bw_comm_time
    # Add PP AR for tied embeddings' weight-grads (backward-embedding-all-reduce).
    # Once per batch iteration.
    self._pp_comm_time_link += self._pp_embed_grad_time_per_batch
    self._pp_comm_time_exposed = self._pp_comm_time_link  # no-overlap ⇒ exposed == link

    # Add LM-head compute to this-rank compute totals (owner last only).
    owner_last_rank = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'last')
    if owner_last_rank:
      m = self.exe._num_microbatches
      # Times
      self._fw_time += m * self._lm_head_fw_comp_time_per_micro
      if self.exe.training:
        self._agrad_time += m * t_dx
        self._wgrad_time += m * t_dw
      # FLOPs & bytes accounting for LM head
      self._fw_flops += m * lm_fw_flops_per_micro
      self._fw_mem_accessed += m * lm_fw_bytes_per_micro
      if self.exe.training:
        self._agrad_flops += m * lm_dx_flops_per_micro
        self._agrad_mem_accessed += m * lm_dx_bytes_per_micro
        self._wgrad_flops += m * lm_dw_flops_per_micro
        self._wgrad_mem_accessed += m * lm_dw_bytes_per_micro

    self.log.debug("%s %s", 'TP comm baseblock FW time:', self._baseblock_fw_tp_time)
    self.log.debug("%s %s", 'TP comm edgeblock FW time:', self._edgeblock_fw_tp_time)
    self.log.debug("%s %s", 'TP comm FW time:', tp_fw_comm_time)
    self.log.debug("%s %s", 'TP comm baseblock FW exposed time:', self._baseblock_fw_tp_time_exposed)
    self.log.debug("%s %s", 'TP comm edgeblock FW exposed time:', self._edgeblock_fw_tp_time_exposed)
    self.log.debug("%s %s", 'TP comm FW exposed time:', tp_fw_comm_time_exposed)
    self.log.debug("%s %s", 'TP comm baseblock BW time:', self._baseblock_agrad_tp_time)
    self.log.debug("%s %s", 'TP comm edgeblock BW time:', self._edgeblock_agrad_tp_time)
    self.log.debug("%s %s", 'TP comm BW time:', tp_bw_comm_time)
    self.log.debug("%s %s", 'TP comm baseblock BW exposed time:', self._baseblock_agrad_tp_time_exposed)
    self.log.debug("%s %s", 'TP comm edgeblock BW exposed time:', self._edgeblock_agrad_tp_time_exposed)
    self.log.debug("%s %s", 'TP comm BW exposed time:', tp_bw_comm_time_exposed)
    self.log.debug("%s %s", 'TP comm extra (embed/lm_head) link time:', tp_extra_link)
    self.log.debug("%s %s", 'PP comm chunk FW time:', chunk_fw_pp_time)
    self.log.debug("%s %s", 'PP comm chunk BW time:', chunk_bw_pp_time)
    self.log.debug("%s %s", 'PP comm FW time:', pp_fw_comm_time)
    self.log.debug("%s %s", 'PP comm BW time:', pp_bw_comm_time)

    # Bubble forms between i-th microbatch FW and BW passes on the 1st GPU.
    # With no interleaving between blocks, it includes
    # L/gpu * microbatch_time * (p-1) * Tcycle, where cycle includes both
    # FW and BW passes, TP and PP communication for FW and BW passes
    # With full interleaving, we only need microbatch_time * (p-1) * Tcycle time
    self._baseblock_fw_time_no_offload = self._block_fw_time + self._baseblock_fw_tp_time_exposed
    self._edgeblock_fw_time_no_offload = self._block_fw_time + self._edgeblock_fw_tp_time_exposed + chunk_fw_pp_time
    self._baseblock_fw_offload_overhead = max(
      0,
      self.get_fw_offload_time() + self._block_fw_mem_time - self._baseblock_fw_time_no_offload
    )
    self._edgeblock_fw_offload_overhead = max(
      0,
      self.get_fw_offload_time() + self._block_fw_mem_time - self._edgeblock_fw_time_no_offload
    )
    self._baseblock_fw_time = self._baseblock_fw_time_no_offload + self._baseblock_fw_offload_overhead
    self._edgeblock_fw_time = self._edgeblock_fw_time_no_offload + self._edgeblock_fw_offload_overhead

    # When we consider block BW time, we do not add optimizer step to it
    # because we have optimizer only for last microbatches, while offloading
    # works during the whole backward pass.
    # Optimizer step is overall memory bound streaming task, it is reasonable
    # to not overlap offloading with optimizer step
    self._baseblock_bw_time_no_offload = (
      self._block_re_time + self._baseblock_recomm_time_exposed +
      self._block_agrad_time + self._block_wgrad_time +
      self._baseblock_agrad_tp_time_exposed)
    self._edgeblock_bw_time_no_offload = (
      self._block_re_time + self._edgeblock_recomm_time_exposed +
      self._block_agrad_time + self._block_wgrad_time +
      self._edgeblock_agrad_tp_time_exposed + chunk_bw_pp_time)
    self._baseblock_bw_offload_overhead = max(
      0, self.get_bw_offload_time() + self._block_agrad_mem_time +
      self._block_wgrad_mem_time -
      self._baseblock_bw_time_no_offload)
    self._edgeblock_bw_offload_overhead = max(
      0, self.get_bw_offload_time() + self._block_agrad_mem_time +
      self._block_wgrad_mem_time -
      self._edgeblock_bw_time_no_offload)
    self._baseblock_bw_time = (
      self._baseblock_bw_time_no_offload + self._baseblock_bw_offload_overhead)
    self._edgeblock_bw_time = (
      self._edgeblock_bw_time_no_offload + self._edgeblock_bw_offload_overhead)
    chunk_fw_time = (
      (self._baseblocks_per_chunk * self._baseblock_fw_time) +
      (self._edgeblocks_per_chunk * self._edgeblock_fw_time))
    chunk_bw_time = (
      (self._baseblocks_per_chunk * self._baseblock_bw_time) +
      (self._edgeblocks_per_chunk * self._edgeblock_bw_time))

    # Can't overlap DP comm with mem accesses, but can overlap with offload
    baseblock_dp_overlap_time = self._baseblock_bw_time - (
      self._block_agrad_mem_time + self._block_wgrad_mem_time +
      self._block_re_mem_time)
    edgeblock_dp_overlap_time = self._edgeblock_bw_time - (
      self._block_agrad_mem_time + self._block_wgrad_mem_time +
      self._block_re_mem_time)
    block_dp_compute_time = (
      self._block_agrad_flops_time + self._block_wgrad_flops_time +
      self._block_re_flops_time)

    if not self.exe.optimizer_sharding:
      # If optimizer is not sharded, we can overlap optimizer step with
      # communication, except for memory access time
      baseblock_dp_overlap_time += (
        self._block_optim_time - self._block_optim_mem_time)
      edgeblock_dp_overlap_time += (
        self._block_optim_time - self._block_optim_mem_time)
      block_dp_compute_time += self._block_optim_flops_time

    if self._dp_net == self._tp_net:
      # Can't overlap DP with TP if in the same network
      baseblock_dp_overlap_time -= (
        self._baseblock_recomm_time + self._baseblock_agrad_tp_time)
      edgeblock_dp_overlap_time -= (
        self._edgeblock_recomm_time + self._edgeblock_agrad_tp_time)

    chunk_dp_overlap_time = (
      self._baseblocks_per_chunk * baseblock_dp_overlap_time +
      self._edgeblocks_per_chunk * edgeblock_dp_overlap_time)
    chunk_dp_compute_time = self._blocks_per_chunk * block_dp_compute_time
    chunk_time = chunk_fw_time + chunk_bw_time

    # Block bubbles appear due to uneven division of blocks by pipeline stages
    # and result in the schedule bubble shorten by the missing edge blocks on
    # the later pipeline stages (missing block case)
    if self._baseblocks_per_chunk > 0:
      # We cut last block of chunk, which is half-edge (has PP comm in the end)
      bubble_reduction_time = self._bubble_reduction_blocks * (
        self._baseblock_fw_time + self._edgeblock_fw_time +
        self._baseblock_bw_time + self._edgeblock_bw_time) / 2
    else:
      # If chunk doesn't have base blocks, we cut edge block
      bubble_reduction_time = self._bubble_reduction_blocks * (
        self._edgeblock_fw_time + self._edgeblock_bw_time)

    # With PP interleaving we assume that we move through every chunk at least
    # PP mini batches. If num_microbatches < PP, then we have extra bubbles
    # (missing microbatches case). We have the bubbles in the last microbatches
    # of every overlappable chunk (all but last chunks). Size of bubbles is
    # equal to microbatch_shortage, same number of microbatches will be missing
    # in the last chunk
    chunks_in_bubble = self.exe.pipeline_par - 1
    num_overlappable_chunks = self.exe.pipeline_interleaving - 1
    microbatch_shortage = self.exe.pipeline_par - (
      self.exe._num_microbatches % self.exe.pipeline_par)
    if self.exe._num_microbatches % self.exe.pipeline_par != 0:
      extra_interleaving_bubbles = num_overlappable_chunks * microbatch_shortage
    else:
      extra_interleaving_bubbles = 0
    self._bubble_time = chunks_in_bubble * chunk_time + (
      extra_interleaving_bubbles * chunk_time - bubble_reduction_time)

    self.log.debug("%s %s", 'Block FW time:', self._block_fw_time)
    self.log.debug("%s %s", 'Baseblock FW time:', self._baseblock_fw_time)
    self.log.debug("%s %s", 'With FW offload overhead time:', self._baseblock_fw_offload_overhead)
    self.log.debug("%s %s", 'Edgeblock FW time:', self._edgeblock_fw_time)
    self.log.debug("%s %s", 'With FW offload overhead time:', self._edgeblock_fw_offload_overhead)
    self.log.debug("%s %s", 'Baseblock REcomm exposed time:', self._baseblock_recomm_time_exposed)
    self.log.debug("%s %s", 'Edgeblock REcomm exposed time:', self._edgeblock_recomm_time_exposed)
    self.log.debug("%s %s", 'Block RE time:', self._block_re_time)
    self.log.debug("%s %s", 'Block BW Agrad time:', self._block_agrad_time)
    self.log.debug("%s %s", 'Block BW Wgrad time:', self._block_wgrad_time)
    self.log.debug("%s %s", 'Block optim time:', self._block_optim_time)
    self.log.debug("%s %s", 'Baseblock BW time:', self._baseblock_bw_time)
    self.log.debug("%s %s", 'With BW offload overhead time:', self._baseblock_bw_offload_overhead)
    self.log.debug("%s %s", 'Edgeblock BW time:', self._edgeblock_bw_time)
    self.log.debug("%s %s", 'With BW offload overhead time:', self._edgeblock_bw_offload_overhead)

    # Determines how long it takes to perform the DP per block
    # This assumes no DP communication overlap (will be adjusted later).
    if self.exe.data_par > 1 and self.exe.training:
      self._block_dp_size = _grad_comm_bytes(self._block_weight_space // self._bytes_per_element)
      if self.exe.optimizer_sharding:
        # When performing optimizer sharding, the communication time is a
        # reduce-scatter plus an all-gather.
        self._block_dp_time = (
          self._dp_net.time(
            'reduce_scatter', self._block_dp_size, self.exe.data_par) +
          self._dp_net.time(
            'all_gather', self._block_dp_size, self.exe.data_par))
      else:
        # When not performing optimizer sharding, the communication time is a
        # single all-reduce.
        self._block_dp_time = self._dp_net.time(
          'all_reduce', self._block_dp_size, self.exe.data_par)
    else:
      self._block_dp_size = 0
      self._block_dp_time = 0

    self.log.debug('DP block comm size: %s', human_format(self._block_dp_size, 'bytes'))
    self.log.debug('DP block comm time (no overlap): %.3e', self._block_dp_time)

    # DP overlap happens if DP time for a previous block(s) is lower than
    # microbatch BW pass time for next pack of consecutive blocks
    # If no interleaving, we move a single microbatch through each block
    # and need to overlap DP during a single block single microbatch time
    # In case of full interleaving, we propagate p microbatches through each
    # block and need to overlap DP comm with p-1 microbatches over a block
    # In a mixed case, we can overlap DP communication of several chunks, e.g.
    # non-interleaved blocks (L/gpu / interleaving_factor) over BW pass of
    # p-1 microbatches through the same amount of blocks if memory capacity is
    # enough, or perform offload/prefetch after each block-microbatch
    # For simplicity we count only bandwidth-optimal case
    # Note that uneven extra PP bubbles won't affect overlapping
    if self.exe.data_par > 1 and self.exe.training:
      if self.exe.data_par_overlap:
        # we can evenly overlap all the chunks except for the last one
        # in the last chunk we can overlap only all blocks except for the last
        num_overlappable_chunks = self.exe.pipeline_interleaving - 1
        last_chunk_overlap_size = self._blocks_per_chunk - 1
        # We can overlap DP with BW pass, overlapping AR for previous layer
        # with BW for current, except when optimizer sharded.
        micros_in_flight = min(self.exe._num_microbatches, self.exe.pipeline_par)
        overlap_window = micros_in_flight * chunk_dp_overlap_time
        overlap_compute = micros_in_flight * chunk_dp_compute_time
        chunk_dp_time = self._blocks_per_chunk * self._block_dp_time
        # We may have PP and DP comm colliding if DP comm takes longer than
        # a single chunk BW time. We can't collide more PP than microbatches
        if self._dp_net == self._pp_net:
          if self.exe._num_microbatches % self.exe.pipeline_par != 0:
            num_overlapped_pp = min(
              chunk_dp_time // chunk_bw_time,
              self.exe._num_microbatches % self.exe.pipeline_par)
          else:
            num_overlapped_pp = min(
              chunk_dp_time // chunk_bw_time,
              self.exe.pipeline_par)
        else:
          # if PP and DP on different networks, overlapping is fine
          num_overlapped_pp = 0
        # we add DP/PP collision time and compute slowdown due to overlap
        overlap_inflection = chunk_dp_time - (overlap_window -
          num_overlapped_pp * chunk_bw_pp_time) + overlap_compute * \
          self._dp_net.processor_usage
        if overlap_inflection > 0:
          # Tcomm is larger than compute, excess is exposed
          overlappable_chunks_exposed_time = num_overlappable_chunks * \
            overlap_inflection
        else:
          # Tcomm is smaller than compute and hidden, but it contributes to
          # compute slowdown due part of compute resources orchestrating comm
          overlappable_chunks_exposed_time = num_overlappable_chunks * \
            chunk_dp_time * self._dp_net.processor_usage
        # Compute minimal bandwidth required for DP comm overlap of all chunks
        # but the last one.
        chunk_overlap_time = overlap_window + overlap_compute * \
          self._dp_net.processor_usage
        if self._dp_net == self._pp_net:
          chunk_overlap_time -= chunk_bw_pp_time
        chunk_overlap_time *= num_overlappable_chunks
        if chunk_overlap_time > 0:
          self._dp_bw_overlap_req_chunk = self._blocks_per_chunk * \
            self._block_dp_size / chunk_overlap_time
          if self.exe.optimizer_sharding:
            self._dp_bw_overlap_req_chunk *= (
              self._dp_net._ops["reduce_scatter"].scalar +
              self._dp_net._ops["all_gather"].scalar)
          else:
            self._dp_bw_overlap_req_chunk *= self._dp_net._ops["all_reduce"].scalar
        else:
          self._dp_bw_overlap_req_chunk = 0
        # in the last chunk, we overlap DP comm over last edge block and all
        # middle blocks, so we substract the time of the first edge block
        if self._baseblocks_per_chunk > 0:
          last_chunk_window = chunk_dp_overlap_time - chunk_bw_pp_time - (
            self._baseblock_bw_time + self._edgeblock_bw_time) / 2
          if not self.exe.optimizer_sharding:
            # If optimizer is not sharded, we can overlap optimizer step with
            # communication, except for memory access time
            last_chunk_window += (
              self._block_optim_time - self._block_optim_mem_time)
        else:
          # if there is no base blocks, we only have a single edge block
          # and last chunk is completely not overlappable
          last_chunk_window = 0
        last_chunk_inflection = (
          last_chunk_overlap_size * self._block_dp_time) + (
            block_dp_compute_time * self._dp_net.processor_usage -
            last_chunk_window)
        if last_chunk_inflection > 0:
          # Tcomm is larger than compute, excess is exposed
          last_chunk_exposed_time = last_chunk_inflection
        else:
          # Tcomm is smaller than compute and hidden, but it contributes to
          # compute slowdown due part of compute resources orchestrating comm
          last_chunk_exposed_time = last_chunk_overlap_size * \
            self._block_dp_time * self._dp_net.processor_usage
        exposed_time = \
          overlappable_chunks_exposed_time + last_chunk_exposed_time
        # Compute minimal bandwidth required for DP comm overlap of last chunk
        tail_overlap_time = last_chunk_window + last_chunk_overlap_size * \
          self._block_dp_time * self._dp_net.processor_usage
        if tail_overlap_time > 0:
          self._dp_bw_overlap_req_tail = self._blocks_per_chunk * \
          self._block_dp_size / tail_overlap_time
          if self.exe.optimizer_sharding:
            self._dp_bw_overlap_req_tail *= (
              self._dp_net._ops["reduce_scatter"].scalar +
              self._dp_net._ops["all_gather"].scalar)
          else:
            self._dp_bw_overlap_req_tail *= self._dp_net._ops["all_reduce"].scalar
        else:
          self._dp_bw_overlap_req_tail = 0
        self._dp_comm_time_exposed = self._block_dp_time + exposed_time
        self._dp_comm_time_link = self._blocks_per_proc * self._block_dp_time
        self.log.debug('Blocks per chunk: %d', self._blocks_per_chunk)
        self.log.debug('Num overlappable chunks: %d', num_overlappable_chunks)
        self.log.debug('Last chunk size: %d', last_chunk_overlap_size)
        self.log.debug('Chunk exposed time: %.3e', max(0, \
          chunk_dp_time + num_overlapped_pp * chunk_bw_pp_time - \
          overlap_window))
        self.log.debug('Last chunk exposed time: %.3e', last_chunk_exposed_time)
      else:
        self._dp_comm_time_exposed = self._blocks_per_proc * self._block_dp_time
        self._dp_comm_time_link = self._dp_comm_time_exposed
        self._dp_bw_overlap_req_chunk = 0
        self._dp_bw_overlap_req_tail = 0
    else:
      self._dp_comm_time_exposed = 0
      self._dp_comm_time_link = 0
      self._dp_bw_overlap_req_chunk = 0
      self._dp_bw_overlap_req_tail = 0

    # Add per-batch DP comm for embedding and LM-head
    if self.exe.training and self.exe.data_par > 1:
      dp_extra_time = 0.0
      # Ownership:
      #  - Embedding: first stage always owns; last stage also owns only when tied.
      #  - Untied LM head: last stage owns.
      owner_first = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'first')
      owner_last = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'last')
      # Token embedding shard (TP-sharded along vocab)
      if owner_first or (self.app.tie_embeddings and owner_last):
        emb_bytes = _grad_comm_bytes(self._embed_weight_elems_shard)
        if self.exe.optimizer_sharding:
          dp_extra_time += self._dp_net.time('reduce_scatter', emb_bytes, self.exe.data_par)
          dp_extra_time += self._dp_net.time('all_gather',     emb_bytes, self.exe.data_par)
        else:
          dp_extra_time += self._dp_net.time('all_reduce',     emb_bytes, self.exe.data_par)
      # Untied LM head shard only on the last stage (or PP==1)
      if (not self.app.tie_embeddings) and owner_last:
        lm_head_bytes = _grad_comm_bytes(self._lm_head_weight_elems_shard)
        if self.exe.optimizer_sharding:
          dp_extra_time += self._dp_net.time('reduce_scatter', lm_head_bytes, self.exe.data_par)
          dp_extra_time += self._dp_net.time('all_gather',     lm_head_bytes, self.exe.data_par)
        else:
          dp_extra_time += self._dp_net.time('all_reduce',     lm_head_bytes, self.exe.data_par)
      self._dp_comm_time_link    += dp_extra_time
      self._dp_comm_time_exposed += dp_extra_time
      self.log.debug('DP extra (emb/head) time (per-batch): %.3e', dp_extra_time)

    self.log.debug('Chunk FW time: %.3e', chunk_fw_time)
    self.log.debug('Chunk BW time: %.3e', chunk_bw_time)
    self.log.debug('Chunk BW time for DP overlap: %.3e', chunk_dp_overlap_time)
    self.log.debug('DP comm time exposed: %.3e', self._dp_comm_time_exposed)
    self.log.debug('DP comm time on the link: %.3e', self._dp_comm_time_link)
    self.log.debug('DP comm required bandwidth for overlapped chunks: %s',
                   human_format(self._dp_bw_overlap_req_chunk, "bandwidth"))
    self.log.debug('DP comm required bandwidth for the last chunk: %s',
                   human_format(self._dp_bw_overlap_req_tail, "bandwidth"))

    # memory capacity stats
    self._weight_space = self._block_weight_space * self._blocks_per_proc
    # account for activation recomputation for full recompute we keep single block's activations
    # (no scaling by L/gpu)
    if self.exe.training:
      mem_microbatches = min(self.exe._num_microbatches, self.exe.pipeline_par)
      if self.exe.activation_recompute == "full":
        assert self._block_act_storage_space == 0, \
          "We expect with full act recomputation we recompute ALL activations"
        self._act_space = self._block_act_working_peak
        self._act_checkpoint_size = (
            self._blocks_per_proc * self._block_act_checkpoint_size * mem_microbatches)
      else:
        # Without full recompute, we don't need checkpoints
        self._act_checkpoint_size = 0
        if self.exe.pipeline_par == 1:
          # PP=1: sequential; keep peak working for current block + stored for all previous blocks
          self._act_space = (
              self._block_act_working_peak +
              self._block_act_storage_space * (self._blocks_per_proc - 1)
          )
        else:
          # PP>1: at most mem_microbatches micros resident on a stage
          # working set is a *peak* of one block; stored activations per micro are sum over this stage
          self._act_space = (
              self._block_act_working_peak +
              self._block_act_storage_space * (self._blocks_per_proc * mem_microbatches)
          )
      # Only need activation grads for a single block
      self._act_grad_space = self._block_act_grad_space
    else:
      # Inference: PP=1 still benefits from working-peak.
      if self.exe.pipeline_par == 1:
        self._act_space = self._block_act_working_peak + \
          self._block_act_storage_space * (self._blocks_per_proc - 1)
      else:
        self._act_space = self._block_act_working_peak
      self._act_checkpoint_size = 0
      self._act_grad_space = 0

    # Optimizer split already accounted for during block compilation
    # We should keep non-sharded weight grad for a current block for AllReduce
    # and one that we currently compute, so 2x total
    # We only need a single no sharded weight grad copy for before reduction
    if self.exe.training:
      if self._blocks_per_proc == 1:
        self._weight_grad_space = self._block_weight_grad_space_no_sharding
      else:
        self._weight_grad_space = \
          self._block_weight_grad_space_no_sharding + \
          self._block_weight_grad_space * (self._blocks_per_proc - 1)
      self._optimizer_space = \
        self._block_optimizer_space * self._blocks_per_proc
    else:
      self._weight_grad_space = 0
      self._optimizer_space = 0

    # Add embedding / LM head capacity on owner stages
    # Embedding owner: first stage always; last stage only when tied.
    owner_first = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'first')
    owner_last = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'last')
    opt_shard = self.exe.data_par if self.exe.optimizer_sharding else 1
    if owner_first or (self.app.tie_embeddings and owner_last):
      # Token embedding shard
      self._weight_space += self._embed_weight_elems_shard * self._bytes_per_element
      # Weight grads: we keep one non-sharded copy before reduction, in FP32,
      # plus (blocks_per_proc-1) sharded copies; however embeddings are a
      # single "layer", so model 1× non-sharded only, matching Linear logic.
      if self.exe.training:
        # Non-sharded FP32 grad before reduction
        self._weight_grad_space += self._embed_weight_elems_shard * System.TypeSizes['float32']
        # Optimizer state (Adam): 2 FP32 moments; master weights only if
        # training dtype is lower precision than FP32.
        self._optimizer_space += (2 * self._embed_weight_elems_shard * System.TypeSizes['float32']) / opt_shard
        if self.exe.datatype != 'float32':
          self._optimizer_space += (self._embed_weight_elems_shard * System.TypeSizes['float32']) / opt_shard
    # If embeddings are untied, add separate LM head on the last stage.
    if (not self.app.tie_embeddings) and owner_last:
      self._weight_space += self._lm_head_weight_elems_shard * self._bytes_per_element
      if self.exe.training:
        self._weight_grad_space += self._lm_head_weight_elems_shard * System.TypeSizes['float32']
        self._optimizer_space += (2 * self._lm_head_weight_elems_shard * System.TypeSizes['float32']) / opt_shard
        if self.exe.datatype != 'float32':
          self._optimizer_space += (self._lm_head_weight_elems_shard * System.TypeSizes['float32']) / opt_shard

  def _check_mem_caps(self):
    if self.get_mem_tier1_cap_req() > self.sys.mem1.capacity:
      raise self.Error(f'Mem tier1 needs '
                       f'{human_format(self.get_mem_tier1_cap_req(), "bytes")} '
                       f'but only has '
                       f'{human_format(self.sys.mem1.capacity, "bytes")}')
    if self.get_mem_tier2_cap_req() > self.sys.mem2.capacity:
      raise self.Error(f'Mem tier2 needs '
                       f'{human_format(self.get_mem_tier2_cap_req(), "bytes")} '
                       f'but only has '
                       f'{human_format(self.sys.mem2.capacity, "bytes")}')

  def _misc_sanity_checks(self):
    if self.exe.tensor_par == 1:
      assert self.get_tp_comm_exposed_time() == 0
      assert self.get_tp_comm_link_time() == 0
    if self.exe.pipeline_par == 1:
      assert self.get_pp_comm_exposed_time() == 0
      assert self.get_pp_comm_link_time() == 0
    if self.exe.data_par == 1:
      assert self.get_dp_comm_exposed_time() == 0
      assert self.get_dp_comm_link_time() == 0

    assert self._fw_flops >= self._block_fw_flops
    assert self._fw_flops_time >= self._block_fw_flops_time
    assert self._fw_mem_accessed >= self._block_fw_mem_accessed
    assert self._fw_mem_time >= self._block_fw_mem_time
    assert self._fw_time >= self._block_fw_time
    assert self._re_flops >= self._block_re_flops
    assert self._re_flops_time >= self._block_re_flops_time
    assert self._re_mem_accessed >= self._block_re_mem_accessed
    assert self._re_mem_time >= self._block_re_mem_time
    assert self._re_time >= self._block_re_time
    assert self._agrad_flops >= self._block_agrad_flops
    assert self._agrad_flops_time >= self._block_agrad_flops_time
    assert self._agrad_mem_accessed >= self._block_agrad_mem_accessed
    assert self._agrad_mem_time >= self._block_agrad_mem_time
    assert self._agrad_time >= self._block_agrad_time
    assert self._wgrad_flops >= self._block_wgrad_flops
    assert self._wgrad_flops_time >= self._block_wgrad_flops_time
    assert self._wgrad_mem_accessed >= self._block_wgrad_mem_accessed
    assert self._wgrad_mem_time >= self._block_wgrad_mem_time
    assert self._wgrad_time >= self._block_wgrad_time
    assert self._optim_flops >= self._block_optim_flops
    assert self._optim_flops_time >= self._block_optim_flops_time
    assert self._optim_mem_accessed >= self._block_optim_mem_accessed
    assert self._optim_mem_time >= self._block_optim_mem_time
    assert self._optim_time >= self._block_optim_time
    assert self._weight_space >= self._block_weight_space
    assert self._act_space >= self._block_act_working_space
    assert self._act_checkpoint_size >= self._block_act_checkpoint_size
    assert self._weight_grad_space >= self._block_weight_grad_space_no_sharding
    assert self._act_grad_space == self._block_act_grad_space
    assert self._optimizer_space >= self._block_optimizer_space

    if not self.exe.training:
      # when not training (inference), backward is not performed and DP has no
      # communication overhead
      assert self.get_bw_time() == 0
      assert self.get_optim_step_time() == 0
      assert self.get_bw_offload_time() == 0
      assert self.get_recompute_time() == 0
      assert self.get_act_checkpoint_size() == 0
      assert self.get_dp_comm_exposed_time() == 0
      assert self.get_dp_comm_link_time() == 0
    else:
      # when training, backward is performed
      assert self.get_bw_time() > 0
      assert self.get_optim_step_time() > 0
      if self.exe.activation_recompute == 'full':
        assert self.get_recompute_time() > 0
        assert self.get_act_checkpoint_size() > 0
      elif self.exe.activation_recompute == 'attn_only':
        assert self.get_recompute_time() > 0
        assert self.get_act_checkpoint_size() == 0
      else:
        if not self.exe.seq_par_ag_redo:
          assert self.get_recompute_time() == 0
        assert self.get_act_checkpoint_size() == 0


  def run(self, sys):
    assert self._compiled, "You must first call self.compile()"
    assert not self._executed
    assert isinstance(sys, System)
    self._compute_block_stats()
    self._compute_batch_stats()
    self._check_mem_caps()
    self._misc_sanity_checks()
    self._executed = True

  def _get_fw_offload_size(self):
    if self.exe.weight_offload:
      weight_offload_size = self._block_weight_space
    else:
      weight_offload_size = 0
    if self.exe.activations_offload:
      if self.exe.activation_recompute != 'full':
        act_offload_size = self._block_act_storage_space
      else:
        act_offload_size = self._block_act_checkpoint_size
    else:
      act_offload_size = 0
    return max(weight_offload_size, act_offload_size)

  def _get_bw_offload_size(self):
    bw_offload_size = 0
    if self.exe.training:
      if self.exe.weight_offload:
        bw_offload_size += self._block_weight_space
      if self.exe.activations_offload:
        if self.exe.activation_recompute != 'full':
          bw_offload_size += self._block_act_storage_space
        else:
          bw_offload_size += self._block_act_checkpoint_size
      if self.exe.optimizer_offload:
        bw_offload_size += self._block_optimizer_space
    return bw_offload_size

  def get_fw_time(self):
    return self._fw_time

  def get_fw_offload_time(self):
    return self.sys.compute_offload_time(self._get_fw_offload_size())

  def get_fw_offload_overhead(self):
    full_overhead = self.exe._num_microbatches * self._chunks_per_proc * (
      (self._baseblocks_per_chunk * self._baseblock_fw_offload_overhead) +
      (self._edgeblocks_per_chunk * self._edgeblock_fw_offload_overhead))
    return full_overhead

  def get_bw_time(self):
    return self._agrad_time + self._wgrad_time

  def get_optim_step_time(self):
    return self._optim_time

  def get_bw_offload_time(self):
    if self.exe.training:
      return self.sys.compute_offload_time(self._get_bw_offload_size())
    else:
      return 0

  def get_bw_offload_overhead(self):
    if self.exe.training:
      full_overhead = self.exe._num_microbatches * self._chunks_per_proc * (
        (self._baseblocks_per_chunk * self._baseblock_bw_offload_overhead) +
        (self._edgeblocks_per_chunk * self._edgeblock_bw_offload_overhead))
      return full_overhead
    else:
      return 0

  def get_recompute_time(self):
    return self._re_time

  def get_recomm_exposed_time(self):
    if self.exe.training:
      return self._recomm_time_exposed
    else:
      return 0

  def get_recomm_link_time(self):
    if self.exe.training:
      return self._recomm_time_link
    else:
      return 0

  def get_bubble_time(self):
    return self._bubble_time

  def get_tp_comm_exposed_time(self):
    return self._tp_comm_time_exposed

  def get_pp_comm_exposed_time(self):
    return self._pp_comm_time_exposed

  def get_dp_comm_exposed_time(self):
    if self.exe.training:
      return self._dp_comm_time_exposed
    else:
      return 0

  def get_tp_comm_link_time(self):
    return self._tp_comm_time_link

  def get_pp_comm_link_time(self):
    return self._pp_comm_time_link

  def get_dp_comm_link_time(self):
    if self.exe.training:
      return self._dp_comm_time_link
    else:
      return 0

  def get_dp_comm_net_time(self):
    if self.exe.training:
      return self._blocks_per_proc * self._block_dp_time
    else:
      return 0

  def get_total_time(self):
    time = self.get_fw_time()
    time += self.get_bw_time()
    time += self.get_optim_step_time()
    time += self.get_fw_offload_overhead()
    time += self.get_bw_offload_overhead()
    time += self.get_recompute_time()
    time += self.get_recomm_exposed_time()
    time += self.get_bubble_time()
    time += self.get_tp_comm_exposed_time()
    time += self.get_pp_comm_exposed_time()
    time += self.get_dp_comm_exposed_time()
    return time

  def get_useful_flops(self):
    total_flops = sum(
      [block.get_fw_flops() for block in self._llm_block])
    if self.exe.training:
      total_flops += sum(
        [block.get_agrad_flops() + block.get_wgrad_flops() + \
          block.get_optim_step_flops() for block in self._llm_block])
    # Add LM-head FLOPs (per microbatch), but spread them evenly across
    # blocks so the caller's multiplication by blocks_per_proc yields
    # the correct per-rank total.
    owner_last = (self.exe.pipeline_par == 1) or (self.app.pp_stage_role == 'last')
    if owner_last:
      B = self.exe.microbatch_size
      S = self.app.seq_size
      H = self.app.hidden
      V_shard = self._padded_vocab // self.exe.tensor_par
      lm_fw = 2.0 * B * S * H * V_shard      # [B*S,H] x [H,V_shard]
      if self.exe.training:
        lm_dx = 2.0 * B * S * H * V_shard    # [B*S,V] x [V,H]
        lm_dw = 2.0 * H * V_shard * (B * S)  # [H,B*S] x [B*S,V]
        lm_total = lm_fw + lm_dx + lm_dw
      else:
        lm_total = lm_fw
      denom = max(1, self._blocks_per_proc)
      total_flops += lm_total / denom
    return total_flops

  def get_compute_efficiency(self):
    total_flops = self.get_useful_flops()
    compute_time = self.get_fw_time() + self.get_bw_time() + \
      self.get_optim_step_time()
    perfect_time = self._blocks_per_proc * self.exe._num_microbatches * \
      total_flops / self.sys.matrix.flops(self.exe.datatype)
    return perfect_time / max(compute_time, 1e-12)

  def get_system_efficiency(self):
    compute_time = self.get_fw_time() + self.get_bw_time() + \
      self.get_optim_step_time()
    return compute_time / max(self.get_total_time(), 1e-12)

  def get_total_efficiency(self):
    total_flops = self.get_useful_flops()
    perfect_time = self._blocks_per_proc * self.exe._num_microbatches * \
      total_flops / self.sys.matrix.flops(self.exe.datatype)
    return perfect_time / max(self.get_total_time(), 1e-12)

  def get_weight_space_min(self):
    """
    Minimum tier-1 capacity needed for weights on this rank.
    Double-buffer only when weight offload is enabled (for overlap).
    Otherwise, one copy is sufficient.
    """
    if self.exe.weight_offload:
      return 2 * self._block_weight_space
    else:
      return self._block_weight_space

  def get_weight_space(self):
    return self._weight_space

  def get_act_space_min(self):
    """
    Minimum tier-1 activation capacity required on this rank (bytes).
    - PP=1: peak working set for the current block + stored activations for the
      remaining (blocks_per_proc-1) blocks in the stage.
    - PP>1: at most min(num_microbatches, PP) micros are resident; stored
      activations scale with blocks_per_proc * mem_microbatches.
    - With full activation recompute, only the per-block peak working set is kept.
    """
    if self.exe.activation_recompute != 'full':
      if self.exe.pipeline_par == 1:
        return self._block_act_working_peak + \
          self._block_act_storage_space * (self._blocks_per_proc - 1)
      else:
        mem_microbatches = min(self.exe._num_microbatches, self.exe.pipeline_par)
        return self._block_act_working_peak + \
          self._block_act_storage_space * (self._blocks_per_proc * mem_microbatches)
    else:
      return self._block_act_working_peak

  def get_act_space(self):
    return self._act_space

  def get_act_checkpoint_size_min(self):
    if self.exe.training:
      if self.exe.activation_recompute != 'full':
        return 0
      else:
        return self._block_act_checkpoint_size * 2
    else:
      return 0

  def get_act_checkpoint_size(self):
    if self.exe.training:
      if self.exe.activation_recompute != 'full':
        return 0
      else:
        return self._act_checkpoint_size
    else:
      return 0

  def get_weight_grad_space_min(self):
    if self.exe.training:
      # We keep one set of non-sharded weight grads after compute before
      # reduction, and one sharded set for offloading
      return self._block_weight_grad_space_no_sharding + \
        self._block_weight_grad_space
    else:
      return 0

  def get_weight_grad_space(self):
    if self.exe.training:
      return self._weight_grad_space
    else:
      return 0

  def get_act_grad_space_min(self):
    return self.get_act_grad_space()

  def get_act_grad_space(self):
    if self.exe.training:
      return self._act_grad_space
    else:
      return 0

  def get_optimizer_space_min(self):
    if self.exe.training:
      return self._block_optimizer_space * 2
    else:
      return 0

  def get_optimizer_space(self):
    if self.exe.training:
      return self._optimizer_space
    else:
      return 0

  def _get_mem_cap_reqs(self):
    """
    This method models the peak memory footprint of a transformer stage
    under realistic parallelism (TP/PP/DP) and overlap behavior.

    Rather than simply summing all tensor spaces, it approximates the
    true temporal residency of major memory components — weights,
    optimizer states, forward activations, and gradients — and distinguishes:
      • Forward-dominated peaks (when activations dominate memory)
      • Backward-dominated peaks (when gradients dominate memory)
      • Their partial overlap (when both coexist briefly in PP=1 or 1F1B)

    For offload cases, it separates device-resident (tier1) and offloaded
    (tier2) memory to reflect multi-tier memory hierarchies realistically.
    """
    if self.exe.weight_offload or self.exe.activations_offload or self.exe.optimizer_offload:
      tier1 = 0
      tier2 = 0
      if self.exe.weight_offload:
        tier1 += self.get_weight_space_min()
        tier2 += self.get_weight_space()
      else:
        tier1 += self.get_weight_space()

      if self.exe.activations_offload:
        if self.exe.activation_recompute != 'full':
          tier1 += self.get_act_space_min()
          tier2 += self.get_act_space()
        else:
          tier1 += self.get_act_space_min()
          tier1 += self.get_act_checkpoint_size_min()
          tier2 += self.get_act_checkpoint_size()
      else:
        tier1 += self.get_act_space()
        tier1 += self.get_act_checkpoint_size()

      if self.exe.optimizer_offload:
        tier1 += self.get_weight_grad_space_min()
        tier1 += self.get_optimizer_space_min()
        tier2 += self._block_weight_grad_space * self._blocks_per_proc
        tier2 += self.get_optimizer_space()
      else:
        tier1 += self.get_weight_grad_space() + self.get_optimizer_space()

      tier1 += self.get_act_grad_space()
      return tier1, tier2

    # No offload: base components
    W = self.get_weight_space()
    OPT = self.get_optimizer_space() if self.exe.training else 0
    ACT = self.get_act_space() + self.get_act_checkpoint_size()
    AG = self.get_act_grad_space() if self.exe.training else 0
    fw_peak = W + OPT + ACT

    bytes_param = System.TypeSizes[self.exe.datatype]
    bytes_grad = System.TypeSizes['float32'] if self.exe.grad_reduce_in_fp32 else bytes_param
    MG = int((W / max(bytes_param, 1)) * bytes_grad) if self.exe.training else 0

    # PP>1: Megatron-style reserved max — forward activations + main grad + act grad
    if self.exe.training and self.exe.pipeline_par > 1:
      return fw_peak + MG + AG, 0

    # PP=1
    overlap_threshold = 1.25
    if self.exe.training and (ACT <= overlap_threshold * MG):
      return max(fw_peak, fw_peak + MG + AG), 0
    else:
      return fw_peak, 0

  def get_mem_tier1_cap_req(self):
    return self._get_mem_cap_reqs()[0]

  def get_mem_tier2_cap_req(self):
    return self._get_mem_cap_reqs()[1]

  def get_act_offload_bw_req(self):
    # We should be able to offload (write) activation during FW pass and
    # prefetch it (read) during BW pass for block (i-1)
    # After BW pass activations are discarded
    if self.exe.activation_recompute != 'full':
      act_offload_size = self._block_act_storage_space
    else:
      act_offload_size = self._block_act_checkpoint_size
    offload_time = min(
      self._baseblock_fw_time_no_offload - self._block_fw_mem_time,
      self._edgeblock_fw_time_no_offload - self._block_fw_mem_time)
    return act_offload_size / max(offload_time, 1e-12)

  def get_weight_offload_bw_req(self):
    # We should be able to offload (write) and prefetch (read) weights both
    # during FW and BW passes for blocks (i-1) / (i+1).
    # We always keep weights, they cannot be discarded
    offload_time = min(
      self._baseblock_fw_time_no_offload - self._block_fw_mem_time,
      self._edgeblock_fw_time_no_offload - self._block_fw_mem_time)
    return self._block_weight_space / max(offload_time, 1e-12)

  def get_optim_offload_bw_req(self):
    # We should be able to offload (write) weight grads and optimizer state
    # and prefetch (read) optimizer state during BW passes for blocks
    # (i-1) / (i+1).
    if self.exe.training:
      offload_time = min(
        self._baseblock_bw_time_no_offload - (self._block_agrad_mem_time +
          self._block_wgrad_mem_time),
        self._edgeblock_bw_time_no_offload - (self._block_agrad_mem_time +
          self._block_wgrad_mem_time))
      return (self._block_weight_grad_space + self._block_optimizer_space) / \
        max(offload_time, 1e-12)
    else:
      return 0

  def get_offload_mem_bw_req(self):
    fw_offload_time = min(
      self._baseblock_fw_time_no_offload - self._block_fw_mem_time,
      self._edgeblock_fw_time_no_offload - self._block_fw_mem_time)
    if self.exe.training:
      bw_offload_time = min(
        self._baseblock_bw_time_no_offload - (self._block_agrad_mem_time +
          self._block_wgrad_mem_time),
        self._edgeblock_bw_time_no_offload - (self._block_agrad_mem_time +
          self._block_wgrad_mem_time))
      req_bw = max(self._get_fw_offload_size() / max(fw_offload_time, 1e-12),
                   self._get_bw_offload_size() / max(bw_offload_time, 1e-12))
      return req_bw
    else:
      return self._get_fw_offload_size() / max(fw_offload_time, 1e-12)

  def get_sample_rate(self):
    return self.exe.global_batch_size / self.get_total_time()

  def display_stats(self):
    stats = "=" * 80 + "\n"
    stats += "" \
      f"blocks={self.app.num_blocks}, " \
      f"hidden={self.app.hidden}, feedforward={self.app.feedforward}\n" \
      f"num attn heads: {self.app.attn_heads}, " \
      f"attn_size={self.app.attn_size}\n" \
      f"Run on {self.exe.num_procs} processors with:\n" \
      f"TP={self.exe.tensor_par}\n" \
      f"PP={self.exe.pipeline_par}\n" \
      f"DP={self.exe.data_par}\n" \
      f"Blocks per processor: {self._blocks_per_proc}\n" \
      f"Execution: {self.exe.get_json()};\n" \
      f"Weights: {human_format(self.get_weight_space(), 'bytes')};\n" \
      f"Act: {human_format(self.get_act_space(), 'bytes')};\n" \
      f"Act CP: {human_format(self.get_act_checkpoint_size(), 'bytes')};\n" \
      f"Act grad: {human_format(self.get_act_grad_space(), 'bytes')};\n" \
      f"Weight grad: {human_format(self.get_weight_grad_space(), 'bytes')};\n" \
      f"Optim space: {human_format(self.get_optimizer_space(), 'bytes')};\n" \
      f"Batch FW time: {self.get_fw_time():.4f};\n" \
      f"Batch BW time: {self.get_bw_time():.4f};\n" \
      f"Batch optim time: {self.get_optim_step_time():.4f};\n" \
      f"Batch FW offload overhead: {self.get_fw_offload_overhead():.4f};\n" \
      f"Batch BW offload overhead: {self.get_bw_offload_overhead():.4f};\n" \
      f"Batch recompute overhead: {self.get_recompute_time():.4f};\n" \
      f"Batch recomm overhead: {self.get_recomm_exposed_time():.4f};\n" \
      f"Batch bubble overhead: {self.get_bubble_time():.4f};\n" \
      f"Batch TP comm overhead: {self.get_tp_comm_exposed_time():.4f};\n" \
      f"Batch PP comm overhead: {self.get_pp_comm_exposed_time():.4f};\n" \
      f"Batch DP comm overhead: {self.get_dp_comm_exposed_time():.4f};\n" \
      f"Batch TP comm time on link: {self.get_tp_comm_link_time():.4f};\n" \
      f"Batch PP comm time on link: {self.get_pp_comm_link_time():.4f};\n" \
      f"Batch DP comm time on link: {self.get_dp_comm_link_time():.4f};\n" \
      f"Batch total time: {self.get_total_time():.4f};\n" \
      f"Activation offload required BW: " \
      f"{human_format(self.get_act_offload_bw_req(), 'bandwidth')};\n" \
      f"Weight offload required BW: " \
      f"{human_format(self.get_weight_offload_bw_req(), 'bandwidth')};\n" \
      f"Optimizer offload required BW: " \
      f"{human_format(self.get_optim_offload_bw_req(), 'bandwidth')};\n" \
      f"Total offload required BW: " \
      f"{human_format(self.get_offload_mem_bw_req(), 'bandwidth')};\n" \
      f"Mem tier1 capacity requirement: " \
      f"{human_format(self.get_mem_tier1_cap_req(), 'bytes')};\n" \
      f"Mem tier2 capacity requirement: " \
      f"{human_format(self.get_mem_tier2_cap_req(), 'bytes')};\n" \
      f"Compute efficiency: {self.get_compute_efficiency()*100:.2f}%;\n" \
      f"System efficiency: {self.get_system_efficiency()*100:.2f}%;\n" \
      f"Total efficiency: {self.get_total_efficiency()*100:.2f}%;\n" \
      f"Sample rate: {self.get_sample_rate():.2f};\n"
    self.log.info(stats)
