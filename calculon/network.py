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


class Network:
  """Configuration for a network."""

  kKeys = set(['bandwidth', 'pp_efficiency', 'ar_efficiency','size', 'latency', 'ops',
               'must_be_filled', 'processor_usage'])
  kNetOps = set(['p2p', 'reduce_scatter', 'all_gather', 'all_reduce'])
  kCollectives = set(['reduce_scatter', 'all_gather', 'all_reduce'])

  class Op:
    def __init__(self, scalar, offset):
      self.scalar = scalar
      self.offset = offset

  @staticmethod
  def _parse_op(op, scalar, offset):
    assert op in Network.kNetOps, f'Invalid network op: {op}'
    assert scalar > 0.0, f'Invalid network scalar for {op}: {scalar}'
    if op in Network.kCollectives:
      assert offset is not None, f'Must give offset for {op}'
      return Network.Op(scalar, offset)
    else:
      assert offset is None, f'Can\'t give offset for {op}'
      return Network.Op(scalar, 0)

  def __init__(self, cfg):
    assert Network.kKeys == set(cfg.keys())
    self._bw = cfg['bandwidth'] * 1e9  # Specified in GB/s
    assert self._bw > 0

    def _parse_net_efficiency(effs):
      return [[thr * 2**20, eff] for thr, eff in effs]

    self._pp_eff = _parse_net_efficiency(cfg['pp_efficiency'])
    self._ar_eff = _parse_net_efficiency(cfg['ar_efficiency'])
    self._size = cfg['size']
    assert self._size >= 0
    self._latency = cfg['latency']
    self._ops = {}
    for op in cfg['ops']:
      self._ops[op] = Network._parse_op(
        op, cfg['ops'][op][0], cfg['ops'][op][1])
    assert set(self._ops.keys()) == Network.kNetOps
    self._must_be_filled = cfg['must_be_filled']
    self._proc_usage = cfg['processor_usage']
    assert self._proc_usage >= 0.0 and self._proc_usage < 1.0

  @property
  def size(self):
    return self._size

  @property
  def must_be_filled(self):
    return self._must_be_filled

  @property
  def processor_usage(self):
    return self._proc_usage

  def get_efficiency(self, op, op_size):
    effs = self._pp_eff if op == "p2p" else self._ar_eff
    for thr, eff in effs:
      if op_size >= thr:
        return eff

  def time(self, op, op_size, comm_size):
    """ Computes the time taken for a network operation.

    Args:
      op (str)        : operation name
      op_size (int)   : operation size in bytes
      comm_size (int) : number of participants in operation

    Returns:
      time (float)    : time needed for operation
    """
    if op not in Network.kCollectives:
      assert comm_size == 2
    else:
      assert comm_size >= 2
    assert op in Network.kNetOps
    assert op_size >= 0
    op_eff = self.get_efficiency(op, op_size)
    # Bandwidth term:
    #   For P2P,   scalar=1, offset=0,  then op_size =             bytes;
    #   For AR,    scalar=2, offset=-1, then op_size = 2*(p-1)/p * bytes;
    #   For RS/AG, scalar=1, offset=-1, then op_size =   (p-1)/p * bytes.
    op_size *= self._ops[op].scalar
    chunk_size = 1 / comm_size * op_size
    op_size += chunk_size * self._ops[op].offset
    # Latency term:
    #   For P2P,      1  hop;
    #   For AR,  2*(p-1) stages;
    #   For RS/AG, (p-1) stages.
    if op == "p2p":
      latency = self._latency
    elif op == "all_reduce":
      latency = (comm_size - 1) * 2 * self._latency
    elif op in ("reduce_scatter", "all_gather"):
      latency = (comm_size - 1) * self._latency
    else:
      raise NotImplementedError("Unknown communication type: {}".format(op))
    return latency + op_size / (self._bw * op_eff)
