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

import math

class Processor:
  """Configuration for a processing engine."""

  def __init__(self, cfg):
    self._datatypes = {}
    for datatype in cfg.keys():
      self._datatypes[datatype] = {
        'flops': cfg[datatype]['tflops'] * 1e12,
      }
      bins = cfg[datatype]['gflops_efficiency']
      if type(bins) == dict:  # parse matrix
        self._datatypes[datatype]['efficiency'] = {}
        for batch, batch_bin in bins.items():
          self._datatypes[datatype]['efficiency'][int(batch)] = {}
          last_bin_flops = None
          for flops, flops_bin in batch_bin.items():
            flops = int(flops) * 1e9
            if last_bin_flops:
              assert flops < last_bin_flops
            last_bin_flops = flops
            self._datatypes[datatype]['efficiency'][int(batch)][int(flops)] = flops_bin
      else:  # parse vector
        self._datatypes[datatype]['efficiency'] = []
        last_bin_flops = None
        for ib in range(len(bins)):
          cur_bin = bins[ib]
          flops = cur_bin[0] * 1e9
          if last_bin_flops:
            assert flops < last_bin_flops
          last_bin_flops = flops
          self._datatypes[datatype]['efficiency'].append([flops] + cur_bin[1:])

  def flops(self, datatype):
    return self._datatypes[datatype]['flops']

  def efficiency(self, datatype, flops=0, batch=1, m=0, n=0, k=0):
    shape_aware = m > 0 and n > 0 and k > 0
    op_flops = batch * 2 * m * n * k if shape_aware else flops
    bins = self._datatypes[datatype]['efficiency']
    if type(bins) == list:
      for ib in range(len(bins)):
        cur_bin = bins[ib]
        flops = cur_bin[0]
        if op_flops >= flops:
          # or size-only: [flops, eff]
          return cur_bin[1]
    elif type(bins) == dict:
      batch_bin = bins[batch]
      for flops in sorted(batch_bin.keys(), reverse=True):
        if op_flops >= flops:
          flops_bin = batch_bin[flops]
          assert len(flops_bin) > 0, f"This bin {flops/1e9} is empty, please check your system configuration file."
          # If exact match
          shape_key = f"{m},{n},{k}"
          if shape_key in flops_bin:
            return flops_bin[shape_key]

          # Not exact match, choose most-similar shape and flops
          def _safelog(x: float) -> float:
            return math.log(max(1.0, float(x)))

          def _gf_from_mnk(B: int, M: int, N: int, K: int) -> float:
            return (B * 2 * M * N * K) / 1e9

          # Find the most-similar shape and flops
          lm, ln, lk = _safelog(m), _safelog(n), _safelog(k)
          lgf_ref = _safelog(_gf_from_mnk(batch, m, n, k))
          best = None        # (shape_dist, flops_dist, eff)
          best_shape = None  # (M, N, K)
          for shape_key_ in flops_bin.keys():
            M_, N_, K_ = map(int, shape_key_.split(','))
            # Log-space shape distance (scale-invariant)
            dm = lm - _safelog(M_)
            dn = ln - _safelog(N_)
            dk = lk - _safelog(K_)
            shape_dist = dm * dm + dn * dn + dk * dk
            # FLOPs closeness in log-space
            lgf = _safelog(_gf_from_mnk(batch, M_, N_, K_))
            flops_dist = abs(lgf - lgf_ref)
            eff_ = flops_bin[shape_key_]
            cand = (shape_dist, flops_dist, -float(eff_))
            if best is None or cand < best:
              best = cand
              best_shape = (M_, N_, K_)
          if best is not None:
            return -best[2]
    assert False, f'{op_flops} wasn\'t covered in {datatype} efficiency curve'

  def throughput(self, datatype, flops=0, batch=1, m=0, n=0, k=0):
    assert datatype in self._datatypes, f'Unsupported type: {datatype}'
    return self.flops(datatype) * self.efficiency(datatype, flops, batch, m, n, k)
