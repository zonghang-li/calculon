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


class Layer:
  """
  A single layer of a neural network. Has weights, activation space,
  gradients, and optimizer state associated with it. May invoke compute,
  memory access, or network operation.
  """

  def __init__(self, name, sys, m=0, n=0, k=0, batch=1, fw_flops=0, agrad_flops=0,
               wgrad_flops=0, inputs_size=0, output_size=0, activation_space=0,
               activation_grads=0, weight_space=0, weight_grads=0,
               optim_space=0, needs_recompute=False, needs_recomm=False,
               activation_reused=False, activation_stored=True,
               output_stored=True):
    self.name = name
    self.sys = sys

    # GEMM mulmat shape for Linear
    self.m = m
    self.n = n
    self.k = k
    self.batch = batch

    # FLOPs per training stage
    self.fw_flops = fw_flops        # Forward-pass FLOPs for this layer
    self.agrad_flops = agrad_flops  # FLOPs to backpropagate activation (output) gradients through this layer
    self.wgrad_flops = wgrad_flops  # FLOPs to compute weight gradients for this layer

    # Tensor sizes (counts of elements, not bytes)
    self.inputs_size = inputs_size  # Number of elements read as inputs in FW
    self.output_size = output_size  # Number of elements produced in FW
    # activations equal input size, we store them to compute Wgrad during BW
    self.activation_space = activation_space  # Elements of stored activations that must be kept for BW
    # activation grads equal output size and correspond grads w.r.t. the output
    self.activation_grads = activation_grads  # Elements of activation gradients flowing through this layer in BW

    # Parameter & optimizer footprints (elements)
    self.weight_space = weight_space  # Number of weight parameters for the layer
    self.weight_grads = weight_grads  # Elements in the weight-gradient tensor (same shape as weights)
    self.optim_space = optim_space    # Elements of optimizer state per weight
    self.optim_sharding_num_proc = 1  # DP sharding factor for optimizer/grads

    # Add optimizations and parallelization split
    self.needs_recompute = needs_recompute      # Mark this layer to be recomputed during BW (activation checkpointing)
    self.needs_recomm = needs_recomm            # Mark that an extra communication is needed during the W-grad stage (e.g., redo an AllGather with RS/AG sequence parallel)
    self.activation_reused = activation_reused  # The layer reuses an activation stored elsewhere (don’t double-count working space).
    self.activation_stored = activation_stored  # Whether to store this layer’s activation for BW
    self.output_stored = output_stored          # Whether the output tensor is explicitly kept (some fused ops don’t keep it)

    # Before bytes_per_element set by SW config, we operate with just parameter count, setting bytes_per_element to 1
    self.bytes_per_element = 1    # Placeholder scalar size; later set by set_bytes_per_element() from the execution datatype (e.g., bf16=2)
    self.processing_time = None   # Filled by timing routines after combining compute/memory/network overlap.
    self.net_exposed_time = None

  def get_stats_json(self):
    return {
      'name': self.name,
      'inputs_size': self.inputs_size,
      'outputs_size': self.output_size,
      'fw_flops': self.get_fw_flops(),
      'fw_mem_accessed': self.get_fw_mem_accessed(),
      'fw_arithmetic_intensity': self.get_fw_arithmetic_intensity(),
      'fw_processing_time': self.compute_processing_time('fw'),
      'baseblock_fw_tp_comm_tile': self.get_comm_tile('fw', baseblock=True),
      'edgeblock_fw_tp_comm_tile': self.get_comm_tile('fw', baseblock=False),
      'baseblock_fw_tp_comm_size': self.get_comm_bytes('fw', baseblock=True),
      'edgeblock_fw_tp_comm_size': self.get_comm_bytes('fw', baseblock=False),
      'baseblock_fw_tp_comm_time': self.compute_net_time('fw', baseblock=True),
      'edgeblock_fw_tp_comm_time': self.compute_net_time('fw',baseblock=False),
      'baseblock_fw_tp_comm_time_exposed': self.get_exposed_net_time(
        'fw', baseblock=True),
      'edgeblock_fw_tp_comm_time_exposed': self.get_exposed_net_time(
        'fw', baseblock=False),
      'agrad_flops': self.get_agrad_flops(),
      'agrad_mem_accessed': self.get_agrad_mem_accessed(),
      'agrad_arithmetic_intensity': self.get_agrad_arithmetic_intensity(),
      'agrad_processing_time': self.compute_processing_time('agrad'),
      'baseblock_bw_tp_comm_tile': self.get_comm_tile('agrad', baseblock=True),
      'edgeblock_bw_tp_comm_tile': self.get_comm_tile('agrad', baseblock=False),
      'baseblock_bw_tp_comm_size': self.get_comm_bytes('agrad', baseblock=True),
      'edgeblock_bw_tp_comm_size': self.get_comm_bytes('agrad', baseblock=False),
      'baseblock_bw_tp_comm_time': self.compute_net_time('agrad', baseblock=True),
      'edgeblock_bw_tp_comm_time': self.compute_net_time('agrad', baseblock=False),
      'baseblock_bw_tp_comm_time_exposed': self.get_exposed_net_time(
        'agrad', baseblock=True),
      'edgeblock_bw_tp_comm_time_exposed': self.get_exposed_net_time(
        'agrad', baseblock=False),
      'wgrad_flops': self.get_wgrad_flops(),
      'wgrad_mem_accessed': self.get_wgrad_mem_accessed(),
      'wgrad_arithmetic_intensity': self.get_wgrad_arithmetic_intensity(),
      'wgrad_processing_time': self.compute_processing_time('wgrad'),
      'baseblock_recomm_tile': self.get_comm_tile('wgrad', baseblock=True),
      'edgeblock_recomm_tile': self.get_comm_tile('wgrad', baseblock=False),
      'baseblock_recomm_size': self.get_comm_bytes('wgrad', baseblock=True),
      'edgeblock_recomm_size': self.get_comm_bytes('wgrad', baseblock=False),
      'baseblock_recomm_time': self.compute_net_time('wgrad', baseblock=True),
      'edgeblock_recomm_time': self.compute_net_time('wgrad', baseblock=False),
      'baseblock_recomm_time_exposed': self.get_exposed_net_time(
        'wgrad', baseblock=True),
      'edgeblock_recomm_time_exposed': self.get_exposed_net_time(
        'wgrad', baseblock=False),
      'optim_flops': self.get_optim_step_flops(),
      'optim_mem_accessed': self.get_optim_step_mem_accessed(),
      'optim_arithmetic_intensity': self.get_optim_step_arithmetic_intensity(),
      'optim_processing_time': self.compute_processing_time('optim'),
      'weight': self.get_weight(),
      'activation': self.get_activation(),
      'weight_grad': self.get_weight_grad(),
      'activation_grad': self.get_activation_grad(),
      'optimizer': self.get_optimizer()
    }

  def get_stats_str(self):
    stats = "Operation {0}:\n{1} FW flops, {2} FW bytes accessed,".format(
      self.name,
      human_format(self.get_fw_flops(), 'flops'),
      human_format(self.get_fw_mem_accessed(), 'bytes'))
    stats += " FW AI: {0:.3f}\n".format(self.get_fw_arithmetic_intensity())
    stats += "{0} BW Adrad flops, {1} BW Agrad bytes accessed,".format(
      human_format(self.get_agrad_flops(), 'flops'),
      human_format(self.get_agrad_mem_accessed(), 'bytes'))
    stats += " BW Agrad AI: {0:.3f}\n".format(
      self.get_agrad_arithmetic_intensity())
    stats += "{0} BW Wdrad flops, {1} BW Wgrad bytes accessed,".format(
      human_format(self.get_wgrad_flops(), 'flops'),
      human_format(self.get_wgrad_mem_accessed(), 'bytes'))
    stats += " BW Wgrad AI: {0:.3f}\n".format(
      self.get_wgrad_arithmetic_intensity())
    stats += "{0} Optim flops, {1} Optim bytes accessed,".format(
      human_format(self.get_optim_step_flops(), 'flops'),
      human_format(self.get_optim_step_mem_accessed(), 'bytes'))
    stats += " Optim AI: {0:.3f}\n".format(
      self.get_optim_step_arithmetic_intensity())
    stats += "W: {0}, Act: {1}, WGrad: {2}, AGrad: {3}, Optim: {4}".format(
      human_format(self.get_weight(), 'bytes'),
      human_format(self.get_activation(), 'bytes'),
      human_format(self.get_weight_grad(), 'bytes'),
      human_format(self.get_activation_grad(), 'bytes'),
      human_format(self.get_optimizer(), 'bytes'))
    return stats

  def set_bytes_per_element(self, bytes_per_element):
    self.bytes_per_element = bytes_per_element

  # Shard (distribute) optimizer and weight grads between data parallel nodes
  def shard_optimizer(self, num_procs):
    self.optim_sharding_num_proc = num_procs

  def get_gemm_shape(self):
    return self.m, self.n, self.k, self.batch

  # getters that will be called from Llm model class, can be rewritten
  def get_fw_flops(self):
    return self.fw_flops

  def get_fw_mem_accessed(self):
    mem_accessed = self.inputs_size + self.output_size + self.weight_space
    mem_accessed *= self.bytes_per_element
    return mem_accessed

  def get_fw_arithmetic_intensity(self):
    if self.fw_flops == 0:
      return 0
    if self.get_fw_mem_accessed() == 0:
      return float('inf')
    return self.fw_flops / self.get_fw_mem_accessed()

  def get_recompute_flag(self):
    return self.needs_recompute

  def get_recomm_flag(self):
    return self.needs_recomm

  def reuses_activation(self):
    return self.activation_reused

  def stores_activation(self):
    return self.activation_stored

  def stores_output(self):
    return self.output_stored

  def get_agrad_flops(self):
    return self.agrad_flops

  def get_agrad_mem_accessed(self):
    # activation grads equal output size and correspond grads w.r.t.
    # layer output; activations are equal to input size
    grad_mem = self.weight_space + (
      self.activation_space + self.activation_grads)
    grad_mem *= self.bytes_per_element
    return grad_mem

  def get_agrad_arithmetic_intensity(self):
    if self.agrad_flops == 0:
      return 0
    if self.get_agrad_mem_accessed() == 0:
      return float('inf')
    return self.agrad_flops / self.get_agrad_mem_accessed()

  def get_wgrad_flops(self):
    return self.wgrad_flops

  def get_wgrad_mem_accessed(self):
    if self.weight_space == 0:
      assert self.wgrad_flops == 0, \
        f"Haven't expected to see wgrad flops in layer {self.name}"
      return 0
    # activation grads equal output size and correspond grads w.r.t.
    # layer output; activations are equal to input size
    grad_mem = self.weight_grads + (
      self.activation_space + self.activation_grads)
    grad_mem *= self.bytes_per_element
    return grad_mem

  def get_wgrad_arithmetic_intensity(self):
    if self.wgrad_flops == 0:
      return 0
    if self.get_wgrad_mem_accessed() == 0:
      return float('inf')
    return self.wgrad_flops / self.get_wgrad_mem_accessed()

  # We use Adam optimizer. The amount of flops is based on the number of
  # weight grads to accommodate for possible weight_grad sharding
  # among data parallel nodes
  def get_optim_step_flops(self):
    optim_flops = self.weight_grads / self.optim_sharding_num_proc * 11
    return optim_flops

  def get_optim_step_mem_accessed(self):
    return self.get_optimizer()

  def get_optim_step_arithmetic_intensity(self):
    if self.get_optim_step_flops() == 0:
      return 0
    if self.get_optim_step_mem_accessed() == 0:
      return float('inf')
    return self.get_optim_step_flops() / self.get_optim_step_mem_accessed()

  def get_weight(self):
    return self.weight_space * self.bytes_per_element

  def get_activation(self):
    return self.activation_space * self.bytes_per_element

  def get_output(self):
    return self.output_size * self.bytes_per_element

  def get_weight_grad(self, sharded=True):
    # Keep lower precision copy of grads for mem and net transfers
    grads = self.weight_grads
    if sharded:
      # We keep grads in lower precision for communication
      grads *= self.bytes_per_element
      grads /= self.optim_sharding_num_proc
    else:
      # otherwise keep grads in 32 bit for accumulation
      grads *= 4
    return grads

  def get_activation_grad(self):
    return self.activation_grads * self.bytes_per_element

  def get_optimizer(self):
    # Keep 32-bits master copy of weights, plus both moments (m,v)
    # master copy for grads is accounted for in get_weight_grad()
    moments_size = self.optim_space * 4
    if self.bytes_per_element < 4:
      master_copy_size = self.weight_space * 4
    else:
      master_copy_size = 0
    return (master_copy_size + moments_size) / self.optim_sharding_num_proc

  def set_processing_time(self, processing_time):
    self.processing_time = processing_time

  def get_processing_time(self):
    return self.processing_time

  def use_matrix_engine(self):
    return False

  def get_comm_bytes(self, stage, baseblock=True):
    return 0

  def get_comm_tile(self, stage, baseblock=True):
    return self.get_comm_bytes(stage, baseblock)

  def compute_flops_time(self, stage):
    if stage == "fw":
      flops = self.get_fw_flops()
    elif stage == "agrad":
      flops = self.get_agrad_flops()
    elif stage == "wgrad":
      flops = self.get_wgrad_flops()
    elif stage == "optim":
      flops = self.get_optim_step_flops()
    else:
      raise Exception(f'Bad compute stage : {stage}')
    if self.use_matrix_engine() and stage != "optim":
      m, n, k, batch = self.get_gemm_shape()
      assert m > 0 and n > 0 and k > 0, "m,n,k must be set for GEMM engine."
      if isinstance(self, (BatchMatMul, BatchMatMulGQA)):
        if stage == "agrad":
          # A-grad = two GEMMs with different shapes
          flops_da = batch * 2 * m * k * n  # dA: (m, k, n)
          flops_db = batch * 2 * n * m * k  # dB: (n, m, k)
          assert flops_da + flops_db == flops
          t_da = flops_da / self.sys.get_matrix_throughput(batch, m, k, n)
          t_db = flops_db / self.sys.get_matrix_throughput(batch, n, m, k)
          return t_da + t_db
        elif stage == "wgrad":
          return 0
      assert batch * 2 * m * n * k == flops
      return flops / self.sys.get_matrix_throughput(batch, m, n, k)
    else:
      return flops / self.sys.get_vector_throughput(flops)

  def compute_mem_time(self, stage):
    if stage == "fw":
      mem = self.get_fw_mem_accessed()
    elif stage == "agrad":
      mem = self.get_agrad_mem_accessed()
    elif stage == "wgrad":
      mem = self.get_wgrad_mem_accessed()
    elif stage == "optim":
      mem = self.get_optim_step_mem_accessed()
    else:
      raise Exception(f'Bad compute stage : {stage}')
    return mem / self.sys.get_mem1_throughput(mem)

  def compute_net_time(self, stage, baseblock=True):
    return 0

  def get_exposed_net_time(self, stage, baseblock=True):
    return 0

  def get_required_bandwidth(self, stage, baseblock=True):
    return 0

  def compute_processing_time(self, stage):
    self.processing_time =  self.sys.get_processing_time(
      self.compute_flops_time(stage),
      self.compute_mem_time(stage)
    )
    return self.processing_time


class Linear(Layer):
  """
  FLOPs: 6 * batch_seq * c_in * c_out, where batch_seq = micro_batch_size * seq_len
    Forward: 2 * batch_seq * c_in * c_out.
    Backward (activation gradients): Same as forward.
    Backward (weight gradients): Same as forward.

  Tensor sizes (element counts, not bytes):
    Inputs read in FW: batch_seq * c_in.
    Outputs written in FW: batch_seq * c_out.
    Weights (parameters): c_in * c_out.
    Weight gradients: c_in * c_out.
    Stored activations (for wgrad): batch_seq * c_in.
    Activation gradients flowing through on BW: batch_seq * c_out.

  Memory traffic (bytes accessed, model’s accounting):
    Forward bytes: (batch_seq * c_in + batch_seq * c_out + c_in * c_out) * bytes_per_element.
    BW activation-grad bytes: Same as forward.
    BW weight-grad bytes: Same as forward.

  Because it’s a GEMM, the arithmetic intensity is high (often compute-bound on matrix units), which the model
  captures via use_matrix_engine().

  Optimizer state (Adam-style):
    optim_space: 2 * c_in * c_out (two moments for each weight).
    Internally, the model:
      keeps moments in FP32 (bytes = 8 * c_in * c_out).
      keeps an FP32 copy of weights if training in <FP32 (bytes = 4 * c_in * c_out).
      applies data-parallel sharding (dividing these by the DP shard count for storage and by the same factor
        for optimizer FLOPs).
  """

  def __init__(self, name, sys, micro_bs, seq_len, c_in, c_out,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    m, n, k = seq_len, c_in, c_out
    super().__init__(name,
                     sys,
                     m=micro_bs*m,  # Megatron‑LM flattens batch × sequence and uses a single 2‑D GEMM:
                                    # m = micro_bs * seq_len, batch = 1 (addmm / matmul on [B*S, H] × [H, K]).
                     n=n,
                     k=k,
                     batch=1,
                     fw_flops=micro_bs*2*m*n*k,
                     agrad_flops=micro_bs*2*m*n*k,
                     wgrad_flops=micro_bs*2*m*n*k,
                     inputs_size=micro_bs*m*n,
                     output_size=micro_bs*m*k,
                     weight_space=n*k,
                     weight_grads=n*k,
                     activation_space=micro_bs*m*n,
                     activation_grads=micro_bs*m*k,
                     optim_space=2*n*k,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    # Forward/agrad/wgrad times use the matrix throughput path (e.g., Tensor Cores),
    # while optimizer uses the vector path.
    return True

class LinearOverlapped(Layer):
  """
  This is a GEMM plus the tensor‑parallel collective (AllReduce / AllGather / ReduceScatter) with an
  overlap schedule (tiling) that hides network under compute and accounts for any exposed network time.
  It also knows which leg of a linear pair it is (“conjugate”) and whether to redo a forward collective
  during wgrad.

  If TP overlap is off, the model builds Linear + explicit TP comm layers around it.
  If TP overlap is on, it builds LinearOverlapped for the QKV/MLP legs that need AG/RS.

  FLOPs:
    Both standard and overlapped linear classes set per‑stage FLOPs to 2 * batch_seq * c_in * c_out for
    FW, A‑grad, and W‑grad, so a full FW+BW is 6 * batch_seq * c_in * c_out. The difference is which
    c_in and c_out they use:
      Linear: uses the global c_in and c_out.
      LinearOverlapped: uses the per-rank c_in and c_out after TP sharding.

  Activations kept vs. flowing grads:
    Linear:
      activation_space = batch_seq * c_in
      activation_grads = batch_seq * c_out
    LinearOverlapped:
      All-gather on forward pass (non‑conjugate leg):
        activation_space = batch_seq * c_in // num_tiles
        activation_grads = batch_seq * c_out
      Reduce-scatter on forward pass (conjugate leg):
        activation_space = batch_seq * c_in
        activation_grads = batch_seq * c_out / num_tiles
        plus an extra backward staging buffer of size batch_seq * c_out / num_tiles counted into activation_grads.
      All-reduce/identity variants follow the same pattern as above for their respective legs (e.g., AR on one
      leg; identity on the other).

  Memory traffic (bytes accessed):
    Forward: (batch_seq * c_in + batch_seq * c_out + c_in * c_out) * bytes_per_element.
    Activation-grad bytes: (c_in * c_out + activation_space + activation_grads) * bytes_per_element.
    Weight-grad bytes: Same as activation-grad bytes.
    Because LinearOverlapped changes activation_space/activation_grads, its backward bytes differ from Linear.
    Linear always uses batch_seq * c_in and batch_seq * c_out for those two terms.

  Communication bytes (only in LinearOverlapped):
    Linear never communicates. LinearOverlapped computes per‑stage communication. Let
      AG_size = (batch_seq * c_in) * bytes_per_element
      AR/RS_size = (batch_seq * c_out) * bytes_per_element
    Forward:
      Conjugate leg → RS or AR → move AR/RS_size.
      Non‑conjugate + split (RS/AG) → AG → move AG_size.
      Otherwise → identity → 0.
    Activation-grad bytes: symmetric to FW but with collectives swapped.
    Weight-grad bytes: optional redo of the FW collective (needs_recomm).
    This reflects RS/AG sequence parallel and AR tensor parallel cases; p2p_rs_ag uses RS/AG only on edge blocks.

  Overlap model (tiling) & “required bandwidth”:
    Linear: no overlap logic; no exposed comm.
    LinearOverlapped: splits compute & comm into num_tiles and overlaps them:
      If a net tile is longer than a (slowed) compute tile, the excess shows up as exposed network time;
        otherwise it’s hidden but slows compute proportionally (models orchestration cost).
      If tp_overlap == 'pipe', it adds one extra exposed comm tile (pipeline warm‑up); 'ring' avoids it.
      It also exposes a convenience metric:
        required_bandwidth = comm_tile_bytes / slowed_flop_tile_time
        i.e., the link BW needed to fully hide the TP collective under compute.

    Optimizer state & grads: Same as for Linear.
  """

  def __init__(self, name, sys, micro_bs, seq_len, c_in, c_out, tensor_par_comm_type,
               num_tiles, net_id, num_peers, conjugate=False,
               in_network_reduction=False, tp_overlap='pipe',
               needs_recompute=False, needs_recomm=False,
               activation_reused=False, activation_stored=True,
               output_stored=True):
    m, n, k = seq_len, c_in, c_out
    self.tensor_par_comm_type = tensor_par_comm_type
    self.num_tiles = num_tiles
    self.net = sys.get_network(net_id)
    self.num_peers = num_peers
    self.conjugate = conjugate
    self.in_network_reduction = in_network_reduction
    self.tp_overlap = tp_overlap
    self._processed_flag = False
    if self.tensor_par_comm_type == 'rs_ag':
      if not conjugate:
        # AllGather case
        assert k % self.num_peers == 0
        # assert (micro_bs * m) % self.num_peers == 0  # this should be true for seq_par
        k = k // self.num_peers
        act_space = micro_bs * m * n // num_tiles
        act_grad_space = micro_bs * m * k
        act_net_buffer = micro_bs * m * n // num_tiles
        act_grad_net_buffer = 0
      else:
        # ReduceScatter case
        assert n % self.num_peers == 0
        # assert (micro_bs * m) % self.num_peers == 0  # this should be true for seq_par
        n = n // self.num_peers
        act_space = micro_bs * m * n
        act_grad_space = micro_bs * m * k // num_tiles
        act_net_buffer = 0
        act_grad_net_buffer = micro_bs * m * k // num_tiles
        # act_net_buffer = micro_bs * m * k // num_tiles
    else:
      if not conjugate:
        # AllReduce case
        assert k % self.num_peers == 0
        k = k // self.num_peers
        act_space = micro_bs * m * n
        act_grad_space = 0
        act_net_buffer = micro_bs * m * n // num_tiles
        act_grad_net_buffer = 0
      else:
        # Identity case
        assert n % self.num_peers == 0
        n = n // self.num_peers
        act_space = 0
        act_grad_space = micro_bs * m * k
        act_net_buffer = 0
        act_grad_net_buffer = micro_bs * m * k

    super().__init__(name,
                     sys,
                     m=micro_bs*m,  # Megatron‑LM flattens batch × sequence and uses a single 2‑D GEMM:
                                    # m = micro_bs * seq_len, batch = 1 (addmm / matmul on [B*S, H] × [H, K]).
                     n=n,
                     k=k,
                     batch=1,
                     fw_flops=micro_bs*2*m*n*k,
                     agrad_flops=micro_bs*2*m*n*k,
                     wgrad_flops=micro_bs*2*m*n*k,
                     inputs_size=micro_bs*m*n,
                     output_size=micro_bs*m*k,
                     weight_space=n*k,
                     weight_grads=n*k,
                     activation_space=act_space, # + act_net_buffer,
                     activation_grads=act_grad_space + act_grad_net_buffer,
                     optim_space=2*n*k,
                     needs_recompute=needs_recompute,
                     needs_recomm=needs_recomm,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True

  def get_comm_bytes(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    ag_comm_size = self.inputs_size * self.bytes_per_element
    ar_rs_comm_size = self.output_size * self.bytes_per_element
    if stage == 'fw':
      if self.conjugate:
        # ReduceScatter or AllReduce on FW
        return ar_rs_comm_size
      else:
        if split_comm:
          # AllGather on FW
          return ag_comm_size
        else:
          # Identity on FW
          return 0
    if stage == 'agrad':
      # Comm sizes during FW and BW pass are the same
      if not self.conjugate:
        # ReduceScatter or AllReduce on BW
        return ag_comm_size
      else:
        if split_comm:
          # AllGather on BW
          return ar_rs_comm_size
        else:
          # Identity on BW
          return 0
    if stage == 'wgrad':
      if self.needs_recomm:
        return self.get_comm_bytes('fw', baseblock)
      else:
        return 0
    if stage == 'optim':
      return 0

  def get_comm_flops(self, stage, baseblock=True):
    return self.get_comm_bytes(stage, baseblock) / self.bytes_per_element

  def get_num_tiles(self):
    return self.num_tiles

  def get_comm_tile(self, stage, baseblock=True):
    return self.get_comm_bytes(stage, baseblock) / self.get_num_tiles()

  def compute_net_time(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    if self.conjugate:
      if split_comm:
        # ReduceScatter case
        fw_comm_type = 'reduce_scatter'
        bw_comm_type = 'all_gather'
      else:
        #AllReduce case
        fw_comm_type = 'all_reduce'
        bw_comm_type = None
      if not self.in_network_reduction:
        fw_flops = self.get_comm_flops(stage, baseblock) * (
          self.num_peers - 1) / self.num_peers
        fw_flop_time = fw_flops / self.sys.get_vector_throughput(fw_flops)
      else:
        fw_flop_time = 0
      bw_flop_time = 0
    else:
      if split_comm:
        #AllGather case
        fw_comm_type = 'all_gather'
        bw_comm_type = 'reduce_scatter'
      else:
        # Identity case
        fw_comm_type = None
        bw_comm_type = 'all_reduce'
      fw_flop_time = 0
      if not self.in_network_reduction:
        bw_flops = self.get_comm_flops(stage, baseblock) * (
          self.num_peers - 1) / self.num_peers
        bw_flop_time = bw_flops / self.sys.get_vector_throughput(bw_flops)
      else:
        bw_flop_time = 0
    if stage == 'fw':
      if fw_comm_type == None:
        return 0
      else:
        fw_net_time = self.net.time(
          fw_comm_type, self.get_comm_bytes(stage, baseblock), self.num_peers)
        return fw_net_time + fw_flop_time
    if stage == 'agrad':
      if bw_comm_type == None:
        return 0
      else:
        bw_net_time = self.net.time(
          bw_comm_type, self.get_comm_bytes(stage, baseblock), self.num_peers)
        return bw_net_time + bw_flop_time
    if stage == 'wgrad':
      if self.needs_recomm and fw_comm_type:
        # AllGather Redo (RS_AG only) or full recompute
        return self.net.time(
          fw_comm_type, self.get_comm_bytes(stage, baseblock), self.num_peers)
      else:
        return 0
    if stage == 'optim':
      return 0

  def compute_processing_time(self, stage):
    flop_time = self.compute_flops_time(stage)
    flop_time_slowed = flop_time / (1 - self.net.processor_usage)
    mem_time = self.compute_mem_time(stage)
    net_time = self.compute_net_time(stage)
    compute_time = self.sys.get_processing_time(flop_time, mem_time)
    if net_time == 0:
      time = compute_time
      net_exposed_time = 0
    else:
      compute_time_slowed = self.sys.get_processing_time(
        flop_time_slowed, mem_time)
      # Tiled time computed as fraction of full time, to model high effective
      # throughput when processing many consequitive tiles
      flop_tile = flop_time / self.num_tiles
      flop_tile_slowed = flop_time_slowed / self.num_tiles
      net_tile = net_time / self.num_tiles
      compute_tile = compute_time / self.num_tiles
      compute_tile_slowed = compute_time_slowed / self.num_tiles
      overlap_inflection = net_tile - flop_tile_slowed
      # we have one exposed comm tile if tp_comm is not ring,
      # one exposed compute tile, and
      # (Proc - 1) overlapped tiles, where either compute or comm is exposed
      if overlap_inflection > 0:
        # Tcomm is larger than compute, excess is exposed
        # compute time itself is the compute + mem
        time = compute_tile + (self.num_tiles - 1) * compute_tile_slowed
        net_exposed_time = (self.num_tiles - 1) * overlap_inflection
      else:
        # Tcomm is smaller than compute and hidden, but it contributes to
        # compute slowdown due part of compute resources orchestrating comm
        time = compute_tile + (self.num_tiles - 1) * compute_tile + (
          self.num_tiles - 1) * net_tile * self.net.processor_usage
        net_exposed_time = 0
      if self.tp_overlap == 'pipe':
        # If overlap type is pipe, we need to add an exposed comm tile
        # with ring-based overlap, we have a special schedule for comm and avoid
        # sending an extra tile we have in the beginning
        net_exposed_time += net_tile
        time += net_tile
    self.processing_time = time
    self.net_exposed_time = net_exposed_time
    self._processed_flag = True
    return self.processing_time

  def get_exposed_net_time(self, stage, baseblock=True):
    # only use after calling compute_processing_time(), otherwise it's set with None
    assert self._processed_flag
    return self.net_exposed_time

  def get_required_bandwidth(self, stage, baseblock=True):
    assert self._processed_flag
    net_tile_size = self.get_comm_tile(stage, baseblock)
    flop_time = self.compute_flops_time(stage)
    flop_time_slowed = flop_time / (1 - self.net.processor_usage)
    flop_tile_slowed = flop_time_slowed / self.num_tiles
    return net_tile_size / flop_tile_slowed


class BatchMatMul(Layer):
  """
  Linear layer is a learned affine transform (single GEMM) with learnable weights. It has parameters,
  weight‑grads, and optimizer state.

  BatchMatMul is a parameter‑free, batched matrix multiply. No weights, no optimizer.

  FLOPs:
    Forward:
      Linear: 2 * batch_seq * c_in * c_out
      BatchMatMul: batch * 2 * size_a * contraction_size * size_b
    Activation-grad:
      Linear: 2 * batch_seq * c_in * c_out (Linear only backprops through one GEMM output)
      BatchMatMul: batch * 4 * size_a * contraction_size * size_b (BatchMatMul computes both ∂A and ∂B (two GEMMs))
    Weight-grad:
      Linear: 2 * batch_seq * c_in * c_out (Linear computes weight grads X^⊤∂Y)
      BatchMatMul: 0 (BatchMatMul has no weights)

  Tensors (elements) each layer:
    Inputs read (FW):
      Linear: batch_seq * c_in
      BatchMatMul: batch * (size_a * contraction_size + contraction_size * size_b) (reads both A and B)
    Outputs written (FW):
      Linear: batch_seq * c_out
      BatchMatMul: batch * size_a * size_b
    Stored activations (for BW):
      Linear: batch_seq * c_in
      BatchMatMul: batch * (size_a * contraction_size + contraction_size * size_b) (stores both A and B)
    Activation grads (BW):
      Linear: batch_seq * c_out
      BatchMatMul: batch * size_a * size_b (∂C)
    Weights (params and grads):
      Linear: c_in * c_out
      BatchMatMul: 0 (BatchMatMul has no weights)

  Memory traffic (bytes):
    Linear: FW bytes, Activation-grad bytes, Weight-grad bytes:
      (batch_seq * c_in + batch_seq * c_out + c_in * c_out) * bytes_per_element
    BatchMatMul:
      FW and Activation-grad bytes (same three tensors as FW):
        batch * (size_a * contraction_size + size_a * size_b + contraction_size * size_b) * bytes_per_element
      Weight-grad bytes: 0 (no parameters)

  Optimizer & grads:
    Linear: keeps Adam moments for each weight (optim_space = 2 * c_in * c_out) and, if training in <FP32,
      a FP32 copy of the weights. Weight‑grads are also present.
    BatchMatMul: no optimizer state, no weight‑grads (parameter‑free op).
  """

  def __init__(self, name, sys, batch, size_a, contraction_size, size_b,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    m, n, k = size_a, contraction_size, size_b
    super().__init__(name,
                     sys,
                     m=m,
                     n=n,
                     k=k,
                     batch=batch,
                     fw_flops=batch*2*m*n*k,
                     agrad_flops=batch*2*2*m*n*k,
                     inputs_size=batch*(m*n+n*k),
                     output_size=batch*m*k,
                     activation_space=batch*(m*n+n*k),
                     activation_grads=batch*m*k,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True


class BatchMatMulGQA(Layer):
  """
  Batch matmul with grouped-query reuse. FLOPs identical to BatchMatMul; memory counts are corrected for GQA.

  New args:
    shared_operand: 'a' if the LEFT operand (A) is shared across heads
                    'b' if the RIGHT operand (B) is shared across heads
    reuse_ratio: kv_groups / attn_heads in (0, 1]; scales memory of the
                 shared operand to reflect reuse across heads in a group
  """
  def __init__(self, name, sys, batch, size_a, contraction_size, size_b,
               shared_operand='b', reuse_ratio=1.0,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    assert shared_operand in ('a', 'b'), "shared_operand must be 'a' or 'b'"
    assert 0.0 < reuse_ratio <= 1.0, "reuse_ratio must be in (0, 1]"
    m, n, k = size_a, contraction_size, size_b
    # Base element counts for inputs
    base_a = batch * m * n
    base_b = batch * n * k
    # Scale the shared operand by reuse_ratio (K or V reused across heads)
    if shared_operand == 'a':
        scaled_inputs = reuse_ratio * base_a + base_b
    else:  # shared_operand == 'b'
        scaled_inputs = base_a + reuse_ratio * base_b
    super().__init__(name,
                     sys,
                     m=m,
                     n=n,
                     k=k,
                     batch=batch,
                     fw_flops=batch*2*m*n*k,
                     agrad_flops=batch*2*2*m*n*k,
                     inputs_size=scaled_inputs,
                     output_size=batch*m*k,
                     activation_space=scaled_inputs,
                     activation_grads=batch*m*k,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True


# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://cthorey.github.io./blog/2016/backpropagation/
class LayerNorm(Layer):
  """
  LayerNorm has tiny parameter count (2 * hidden_size), but low arithmetic intensity. Its training cost is dominated
  by memory traffic for reading/writing activations and gradients, not the math itself. It’s a classic bandwidth-
  limited layer.

  Parameters:
    Weights: two learnable vectors, scale (gamma) and bias (beta), each of length hidden_size.
      Count: weight_space = 2 * hidden_size
    Weight gradients: same size as weights.
      Count: weight_grads = 2 * hidden_size
    Optimizer state: Adam keeps two moments (m, v), both FP32, per parameter.
      Count: optim_space = 4 * hidden_size

  Activations: This layer holds one activation tensor of size act_size during FW, plus gradients of the same size during BW.
    Input size: act_size.
    Output size: act_size.
    Activation storage: act_size.
    Activation gradients: act_size.

  FLOPs: 30 * act_size per pass.
    Forward pass: 9 * act_size. (Computing mean, variance, normalization, and affine transform)
    Backward – activation gradients: agrad_flops = 14 * act_size.
    Backward – weight gradients: wgrad_flops = 7 * act_size.
    Optimizer state FLOPs: The optimizer update is modeled separately, so FLOPs in forward/backward don’t include
      optimizer ops, they belong to a different stage (the “optimizer step”).

  Memory access (bytes accessed per pass): 6 * (act + hidden_size) * bytes_per_element.
    Forward: (input + output + weights) x bytes_per_element = 2 * (act_size + hidden_size) * bytes_per_element.
    Activation-grad pass: (weight + activation_space + activation_grads) * bytes_per_element = 2 * (act_size + hidden_size) * bytes_per_element.
    Weight-grad pass: (weight_grads + activation_space + weight_grads) * bytes_per_element = 2 * (act_size + hidden_size) * bytes_per_element.
    Optimizer state memory access: Optimizer buffers are not accessed during forward/backward passes. They are accessed
      once per training step when applying gradients to weights. The code uses a dedicated function get_optim_step_mem_accessed()
      to return the optimizer storage footprint. In other words, optimizer memory access is accounted in the optimizer
      step, not in FW/BW.
  """
  def __init__(self, name, sys, act_size, hidden,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     sys,
                     fw_flops=9*act_size,
                     agrad_flops=14*act_size,
                     wgrad_flops=7*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     weight_space=2*hidden,
                     weight_grads=2*hidden,
                     optim_space=2*2*hidden,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


class DropOut(Layer):
  """
  A parameter‑free dropout op whose only persistent “activation” for BW is the boolean mask.
  It times on the vector engine (memory bound).

  FLOPs per pass:
    Forward: act_size
    Activation-grad: act_size
    Weight-grad: 0

  Tensor sizes (elements):
    input_size: act_size
    output_size: act_size
    activation_space: act_size
    activation_grads: act_size

  No parameters/optimizer state.
  """

  def __init__(self, name, sys, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     sys,
                     fw_flops=act_size,
                     agrad_flops=act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


  # need to account for DropOut mask of bool type that takes 1 B per element
  # mask is the only DropOut activation
  def get_activation(self):
    return self.activation_space

  def get_activation_grad(self):
    return self.activation_grads

  def get_fw_mem_accessed(self):
    mask_size = self.activation_space
    mem_accessed = self.inputs_size + self.output_size
    mem_accessed *= self.bytes_per_element
    mem_accessed += mask_size
    return mem_accessed

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://mlfromscratch.com/activation-functions-explained/#/
class GeLU(Layer):
  """
  A parameter‑free activation with fixed per‑element costs. It runs on the vector engine.

  FLOPs per pass:
    Forward: 8 * act_size
    Activation-grad: 13 * act_size
    Weight-grad: 0 (no parameters)

  Tensor sizes (elements):
    Inputs read: act_size
    Outputs written: act_size
    Stored activations / flowing act‑grads:
      Unfused (fused=False) → activation_space = act_size, activation_grads = act_size
      Fused (fused=True) → activation_space = 0, activation_grads = 0
        Fused GeLU runs right after previous Linear layer and does not store activations or gradients.
    BW memory override: returns exactly the FW memory, ensuring BW bytes are modeled correctly
      even when fused storage is zero.

  Memory traffic (bytes):
    Forward bytes: 2 * act_size * bytes_per_element
    BW activation‑grad bytes (overridden): Same to the forward bytes
    BW weight‑grad bytes: 0 (no parameters)

  No parameters/optimizer state.
  """

  def __init__(self, name, sys, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               fused=False):
    # Fused GeLU runs right after previous Linear layer and does not store
    # activations or gradients
    self._fused = fused
    if fused:
      eff_act_space = 0
      eff_act_grads = 0
    else:
      eff_act_space = act_size
      eff_act_grads = act_size
    super().__init__(name, sys, fw_flops=8*act_size, agrad_flops=13*act_size,
                     inputs_size=act_size, output_size=act_size,
                     activation_space=eff_act_space,
                     activation_grads=eff_act_grads,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
class SoftMax(Layer):
  """
  A parameter‑free softmax with simple per‑element cost and symmetric FW/BW memory. It does not use the
  matrix engine (so it’s timed on the vector engine).

  FLOPs per pass:
    Forward: 5 * act_size
      A fixed per‑element cost for the numerically stable softmax steps (e.g., max/subtract/exp/sum/divide)
      modeled as 5 FLOPs/element.
    Backward (activation gradients): 8 * act_size
      Fixed per‑element cost for the softmax Jacobian–vector product.
    Backward (weight gradients): 0 (no parameters)

  Tensor sizes (elements):
    Inputs read in FW: act_size
    Outputs written in FW: act_size
    Stored activations for BW: act_size
    Activation gradients flowing on BW: act_size
    Weights / Weight grads / Optimizer: 0

  Memory traffic (bytes):
    Forward bytes: 2 * act_size * bytes_per_element
    BW activation‑grad bytes: Same as FW.
    BW weight‑grad bytes: 0 (no parameters)

  This operation has a low arithmetic intensity, so it is memory-bound in most regimes.
  """

  def __init__(self, name, sys, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     sys,
                     fw_flops=5*act_size,
                     agrad_flops=8*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://explained.ai/matrix-calculus/#sec:1.4.2
class ElementWise(Layer):
  """
  A parameter‑free, elementwise binary op (e.g., residual add) that reads two input tensors, writes one
  output tensor, and backpropagates gradients to both inputs. It runs on the vector engine (memory bound).

  FLOPs per pass:
    Forward: act_size (one op per output element)
    Activation-grad: operand1 + operand2 (one op per element of each input)
    Weight-grad: 0 (no parameters)

  Tensors (elements):
    Inputs read (FW): input_size = operand1 + operand2
    Output written (FW): output_size = act_size
    Stored activations (BW working set): activation_space = operand1 + operand2
    Activation gradients flowing on BW: act_size
    Parameters / weight grads / optimizer: 0

  Memory traffic (bytes):
    Forward bytes: (operand1 + operand2 + act_size) * bytes_per_element
    BW activation‑grad bytes: Same as forward bytes
    BW weight‑grad bytes: 0

  No parameters/optimizer state.
  """

  def __init__(self, name, sys, operand1, operand2,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    act_size = max(operand1, operand2)
    super().__init__(name,
                     sys,
                     fw_flops=act_size,
                     agrad_flops=(operand1+operand2),
                     inputs_size=(operand1+operand2),
                     output_size=act_size,
                     activation_space=(operand1+operand2),
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


# Splits activation on the forward pass, sums gradients on the backward
class Fork(Layer):
  """
  Fork is a fan-out/fan-in utility: free on FW, but adds BW compute and memory traffic proportional to the
  number of consumers, and (by default) increases activation residency, which can influence checkpointing
  and offload requirements.
  """
  def __init__(self, name, sys, act_size, num_users,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.num_users = num_users
    super().__init__(name,
                     sys,
                     inputs_size=act_size,
                     agrad_flops=num_users*act_size,
                     activation_space=act_size,
                     # Gradients from num_users accumulated in a single storage
                     # that's accounted in the other layers
                     # use 0 here to avoid double accounting
                     activation_grads=0,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_fw_mem_accessed(self):
    # Forward (fan-out): one input activation is fanned out to num_users consumers.
    # In this model, fan-out is treated as free (pointer-style), so forward memory traffic is 0 here.
    return 0

  def get_agrad_mem_accessed(self):
    # Even though Fork has no weights, it holds the activation (by default) so that backprop can route/sum gradients cleanly.
    # This increases the block’s activation footprint and can affect activation offload bandwidth if offloading is enabled.
    # Backward (fan-in): the num_users downstream gradients are summed element-wise into a single gradient and sent upstream.
    # The (num_users + 1) factor captures reading/writing an accumulation buffer of activation size while folding in
    # each user's gradient (plus the final write).
    return self.activation_space * self.bytes_per_element * (self.num_users + 1)


class TPComm(Layer):
  """
  A TPComm node models the collective that tensor-parallel ranks perform around a compute layer (e.g., AG/RS/AR).
  It mainly contributes network time (and optionally tiny reduction FLOPs).

  FLOPs:
    If num_peers == 1, all zero, otherwise:
      Non-conjugate instance (conjugate=False):
        FW: 0 FLOPs
        BW: activation_size * (num_peers - 1) / num_peers (only if in_network_reduction=False; otherwise 0)
      Conjugate instance (conjugate=True):
        FW: activation_size * (num_peers - 1) / num_peers (only if in_network_reduction=False; otherwise 0)
        BW: 0 FLOPs

  Forward/Backward memory access:
    TPComm suppresses memory on identity legs:
      FW memory is 0 when the FW leg is identity (e.g., ar with conjugate=False).
      BW (agrad) memory is 0 when the BW leg is identity (e.g., ar with conjugate=True).
    When not identity (e.g., rs_ag legs), FW/BW memory = (input_size + output_size) * bytes_per_element = 2 * act_size * bytes_per_element.

  Activation storage (capacity accounting):
    get_activation():
      rs_ag: activation bytes = act_size * bytes_per_element / num_peers (sequence parallel reduces per-rank activation footprint).
      otherwise:
        if conjugate=True: activation bytes = act_size * bytes_per_element.
        if conjugate=False: identity → 0 (don’t keep extra buffer).
    get_activation_grad() analogous:
      rs_ag: act_size * bytes_per_element / num_peers.
      else:
        if conjugate=True: identity → 0.
        if conjugate=False: act_size * bytes_per_element.

  Communication bytes:
    For num_peers > 1:
      If the leg is identity, get_comm_bytes(..) = 0.
      Else get_comm_bytes(stage) = act_size * bytes_per_element for FW and BW legs, regardless of the algorithm.
      If needs_recomm=True and (RS/AG is used or the instance is conjugate), then redo FW collective:
        get_comm_bytes('wgrad') = act_size * bytes_per_element
  """
  def __init__(self, name, sys, act_size, net_id, num_peers, tensor_par_comm_type,
               conjugate=False, in_network_reduction=False,
               needs_recomm=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.net = sys.get_network(net_id)
    self.num_peers = num_peers
    self.tensor_par_comm_type = tensor_par_comm_type
    self.comm_size = act_size
    self.conjugate = conjugate
    if self.num_peers == 1:
      fw_flops = 0
      bw_flops = 0
      in_size = 0
      out_size = 0
    else:
      if not self.conjugate:
        # FW pass Identity/AllGather, BW pass AllReduce/ReduceScatter
        fw_flops = 0
        if not in_network_reduction:
          bw_flops = act_size * (self.num_peers - 1) / self.num_peers
        else:
          bw_flops = 0
        in_size = act_size
        out_size = act_size
      else:
        # Conjugate function is opposite
        if not in_network_reduction:
          fw_flops = act_size * (self.num_peers - 1) / self.num_peers
        else:
          fw_flops = 0
        bw_flops = 0
        in_size = act_size
        out_size = act_size
    super().__init__(name,
                     sys,
                     fw_flops=fw_flops,
                     agrad_flops=bw_flops,
                     inputs_size=in_size,
                     output_size=out_size,
                     activation_space=in_size,
                     activation_grads=out_size,
                     needs_recomm=needs_recomm,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_activation(self):
    if self.tensor_par_comm_type == 'rs_ag':
      return self.activation_space * self.bytes_per_element / self.num_peers
    else:
      if self.conjugate:
        return self.activation_space * self.bytes_per_element
      else:
        # Identity
        return 0

  def get_fw_mem_accessed(self):
    if not self.tensor_par_comm_type == 'rs_ag' and not self.conjugate:
      # Identity
      return 0
    else:
      return super().get_fw_mem_accessed()

  def get_activation_grad(self):
    if self.tensor_par_comm_type == 'rs_ag':
      return self.activation_space * self.bytes_per_element / self.num_peers
    else:
      if not self.conjugate:
        return self.activation_grads * self.bytes_per_element
      else:
        # Identity
        return 0

  def get_agrad_mem_accessed(self):
    if not self.tensor_par_comm_type == 'rs_ag' and self.conjugate:
      # Identity
      return 0
    else:
      return super().get_agrad_mem_accessed()

  def get_comm_bytes(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    if (not split_comm and (self.conjugate and stage == 'agrad' or
        not self.conjugate and stage == 'fw')):
      # Identity FW or AllReduce BW
      return 0
    else:
      if stage == 'fw' or stage == 'agrad':
        return self.comm_size * self.bytes_per_element
      if stage == 'wgrad' and self.needs_recomm and (
          split_comm or self.conjugate):
        # with AG Redo, we need recomm both on FW pass (not self.conjugate)
        # and BW pass (self.conjugate)
        return self.comm_size * self.bytes_per_element
      else:
        # optim and wgrad stage has no comm if no ag_redo flag for RS_AG
        return 0

  def compute_net_time(self, stage, baseblock=True):
    if self.num_peers == 1:
      return 0
    split_comm = (self.tensor_par_comm_type == 'rs_ag') or (
      (self.tensor_par_comm_type == 'p2p_rs_ag') and not baseblock)
    net_compute_time = super().compute_processing_time(stage)
    if split_comm:
      if self.conjugate:
        # ReduceScatter case
        fw_net_time = self.net.time('reduce_scatter',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
        bw_net_time = self.net.time('all_gather',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
      else:
        #AllGather case
        fw_net_time = self.net.time('all_gather',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
        bw_net_time = self.net.time('reduce_scatter',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
    else:
      if self.conjugate:
        fw_net_time = self.net.time('all_reduce',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
        bw_net_time = 0
      else:
        fw_net_time = 0
        bw_net_time = self.net.time('all_reduce',
          self.get_comm_bytes(stage, baseblock), self.num_peers)
    if stage == 'fw':
      return fw_net_time + net_compute_time
    elif stage == 'agrad':
      return bw_net_time + net_compute_time
    elif stage == 'wgrad':
      # with AG Redo, we need recomm both on FW pass (not self.conjugate)
      # and BW pass (self.conjugate)
      if self.needs_recomm:
        return fw_net_time + net_compute_time
      else:
        return 0
    elif stage == 'optim':
      return 0
    else:
      raise Exception(f'Bad compute stage : {stage}')
    return 0

  def get_exposed_net_time(self, stage, baseblock=True):
    # only use after calling compute_processing_time(), otherwise it's set witth None
    return self.compute_net_time(stage, baseblock)

  def compute_processing_time(self, stage):
    return 0
