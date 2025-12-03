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

import datetime
import logging
import multiprocessing as mp
import psutil
from tqdm import tqdm

import calculon
from calculon.llm import *


def _search_star(args):
  return OptimalExecution.search(*args)


class OptimalExecution(calculon.CommandLine):
  NAME = 'llm-optimal-execution'
  ALIASES = ['loe']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      OptimalExecution.NAME, aliases=OptimalExecution.ALIASES,
      help='run a search to find the optimal llm execution')
    sp.set_defaults(func=OptimalExecution.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('datatype', type=str, choices=System.supported_datatypes(),
                    help='The datatype to use')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('output', type=str,
                    help='File path to the output file'
                    " ('*.csv', '*.csv.gz', '*.json', '*.json.gz')")
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')
    sp.add_argument('-m', '--mbs-break', action='store_true',
                    help='Search across MBS and break earlier when possible')
    sp.add_argument('-t', '--top-n', type=int, default=1,
                    help='Number of best outputs')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')
    sp.add_argument('-f', '--fused_activation', type=arg_true_false_all,
                    default='true', help='Mode of fused activation')
    sp.add_argument('--no-tp-overlap', action='store_true',
                    help='Don\'t allow TP overlap')
    sp.add_argument('--no-dp-overlap', action='store_true',
                    help='Don\'t allow DP overlap')
    sp.add_argument('--activation-recompute', action='store_true',
                    help='Search activation_recompute (otherwise fixed to a default)')
    sp.add_argument('--optimizer-sharding', action='store_true',
                    help='Search optimizer_sharding when dp>1 (otherwise fixed False)')
    sp.add_argument('--tensor-par-comm-type', action='store_true',
                    help='Search tensor_par_comm_type (otherwise fixed to \"ar\")')
    sp.add_argument('--seq-par-ag-redo', action='store_true',
                    help='Search seq_par_ag_redo (otherwise fixed False)')
    sp.add_argument('--weight-offload', action='store_true',
                    help='Search weight_offload (otherwise fixed False)')
    sp.add_argument('--activations-offload', action='store_true',
                    help='Search activations_offload when allowed (otherwise fixed False)')
    sp.add_argument('--optimizer-offload', action='store_true',
                    help='Search optimizer_offload (otherwise fixed False)')
    sp.add_argument('--pipeline-interleaving', action='store_true',
                    help='Search pipeline_interleaving (otherwise fixed to default)')

  @staticmethod
  def run_command(logger, args):
    assert args.top_n > 0, 'top-n must be > 0'
    app = Llm.Application(calculon.io.read_json_file(args.application))
    syst = System(calculon.io.read_json_file(args.system))
    activation_recompute_choices = ['full', 'attn_only', 'none'] if args.activation_recompute else ['none']
    tensor_par_comm_type_choices = ['ar', 'p2p_rs_ag', 'rs_ag'] if args.tensor_par_comm_type else ['ar']
    params = []
    for tp in Llm.get_all_tensor_parallelisms(args.num_procs, app.hidden, app.attn_heads, app.kv_groups):
      for pp in Llm.get_all_pipeline_parallelisms(args.num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(args.num_procs, tp, pp)
        valid_ppints = Llm.get_valid_pipeline_interleavings(app.num_blocks, pp)
        if not args.pipeline_interleaving: valid_ppints = list(valid_ppints)[:1]
        for ppint in valid_ppints:
          batch_size = OptimalExecution.get_batch_size(dp, args.max_batch_size)
          if batch_size is None: continue
          for activation_recompute in activation_recompute_choices:
            optimizer_sharding_choices = pick(dp > 1, [True, False], [False]) if args.optimizer_sharding else [False]
            for optimizer_sharding in optimizer_sharding_choices:
              for tensor_par_comm_type in tensor_par_comm_type_choices:
                params.append(
                  (args.debug, args.top_n, args.layers, args.num_procs,
                   args.max_batch_size, args.datatype, app, syst, tp, pp, dp,
                   ppint, batch_size, activation_recompute, optimizer_sharding,
                   tensor_par_comm_type, args.fused_activation, args.mbs_break,
                   not args.no_tp_overlap, not args.no_dp_overlap, args.seq_par_ag_redo,
                   args.weight_offload, args.activations_offload, args.optimizer_offload))

    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = []
      for res in tqdm(pool.imap_unordered(_search_star, params),
                      total=len(params), desc='Searching', smoothing=0.05):
        searches.append(res)
    end_time = datetime.datetime.now()

    # Combines parallel search result into one data structure
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for cbest, ec, gec, bec, tp, pp in searches:
      best = OptimalExecution.update_list(best, cbest, args.top_n)
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    if args.debug:
      return 0

    if len(best) == 0:
      if not args.noneok:
        logger.fatal('No acceptable configurations found :(')
        return -1
      else:
        logger.info('No acceptable configurations found :(')
    else:
      logger.info(f'Best sample rate: {best[0][0]}')

    output = {}
    for index, run in enumerate(best):
      _, execution, stats = run
      output[index] = {
        'execution': execution,
        'stats': stats
      }

    if calculon.io.is_json_extension(args.output):
      logger.info(f'Output: {args.output}')
      calculon.io.write_json_file(output, args.output)
    elif args.output.endswith('.csv') or args.output.endswith('.csv.gz'):
      logger.info(f'Output: {args.output}')
      exe_keys = list(output[0]['execution'].keys())
      stats_keys = list(output[0]['stats'].keys())
      opener = gzip.open if args.output.endswith('.gz') else open
      with opener(args.output, 'wb') as fd:
        fd.write(bytes(f',{",".join(exe_keys)},{",".join(stats_keys)}\n',
                       'utf-8'))
        for index in sorted(output.keys()):
          fd.write(bytes(f'{index}', 'utf-8'))
          for exe_key in exe_keys:
            fd.write(bytes(f',{output[index]["execution"][exe_key]}', 'utf-8'))
          for stats_key in stats_keys:
            fd.write(bytes(f',{output[index]["stats"][stats_key]}', 'utf-8'))
          fd.write(bytes('\n', 'utf-8'))
    else:
      assert False, f'Unknown file type: {args.output}'

    return 0

  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    if data_par > max_batch_size:
      return None
    last = data_par
    while True:
      if last + data_par > max_batch_size:
        return last
      else:
        last += data_par

  @staticmethod
  def infer_networks(syst, tp, pp, dp):
    networks = []
    for idx in range(syst.num_networks):
      net = syst.get_network(idx)
      networks.append((idx, net.size, net.must_be_filled))
    if not networks:
      return 0, 0, 0
    networks_sorted = sorted(networks, key=lambda x: x[1])
    must_fill = [t for t in networks_sorted if t[2]]
    if must_fill:
      default_idx = must_fill[0][0]
    else:
      default_idx = networks_sorted[0][0]

    def candidates_ge(val):
      return [t for t in networks_sorted if t[1] >= val]

    if tp > 1:
      cand = candidates_ge(tp)
      if not cand:
        raise Llm.Error(f"Tensor parallelism {tp} exceeds all network sizes")
      cand_mf = [t for t in cand if t[2]]
      if cand_mf:
        tn = cand_mf[0][0]
      else:
        tn = cand[0][0]
    else:
      tn = default_idx

    if pp > 1:
      group = tp * pp
      cand = candidates_ge(group)
      if not cand:
        raise Llm.Error(f"Pipeline parallelism {pp} with tensor parallelism {tp} exceeds all network sizes")
      cand_mf = [t for t in cand if t[2]]
      if cand_mf:
        pn = cand_mf[0][0]
      else:
        pn = cand[0][0]
    else:
      pn = default_idx

    if dp > 1:
      cand = candidates_ge(dp)
      if not cand:
        raise Llm.Error(f"Data parallelism {dp} exceeds all network sizes")
      dn = cand[-1][0]
    else:
      dn = networks_sorted[-1][0]

    return tn, pn, dn

  @staticmethod
  def search(debug, top_n, layers, num_procs, max_batch_size, datatype,
             app, syst, tp, pp, dp, ppint, batch_size, activation_recompute,
             optimizer_sharding, tensor_par_comm_type, fused_acts, mbs_break,
             allow_tp_overlap, allow_dp_overlap, search_seq_par_ag_redo,
             search_weight_offload, search_activations_offload, search_optimizer_offload):
    num_nets = syst.num_networks
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    has_mem2 = syst.mem2.capacity > 0
    can_redo = Llm.can_redo_ag(tensor_par_comm_type, activation_recompute)
    seq_par_choices = pick(can_redo and search_seq_par_ag_redo, [True, False],[False])
    data_par_choices = pick(dp > 1 and allow_dp_overlap,[True, False],[False])
    tensor_par_choices = pick(tp > 1 and allow_tp_overlap, ['none', 'ring', 'pipe'],['none'])
    weight_offload_choices = pick(has_mem2 and search_weight_offload,[True, False],[False])
    optimizer_offload_choices = pick(has_mem2 and search_optimizer_offload, [True, False], [False])
    if activation_recompute == 'full' or not has_mem2:
      activations_offloads = [False]
    else:
      activations_offloads = [True, False] if search_activations_offload else [False]

    for seq_par_ag_redo in seq_par_choices:
      for data_par_overlap in data_par_choices:
        for tensor_par_overlap in tensor_par_choices:
          for weight_offload in weight_offload_choices:
            for activations_offload in activations_offloads:
              for optimizer_offload in optimizer_offload_choices:
                for fused_act in fused_acts:
                  for microbatch_size in Llm.get_valid_microbatch_sizes(
                      app.seq_size, tp, dp, batch_size, pp, tensor_par_comm_type):
                    mbs_break_good = good_exe_count
                    tn, pn, dn = OptimalExecution.infer_networks(syst, tp, pp, dp)
                    exe_count += 1
                    exe_json = {
                      'num_procs': num_procs,
                      'tensor_par': tp,
                      'pipeline_par': pp,
                      'data_par': dp,
                      'tensor_par_net': tn,
                      'pipeline_par_net': pn,
                      'data_par_net': dn,
                      'batch_size': batch_size,
                      'microbatch_size': microbatch_size,
                      'datatype': datatype,
                      'fused_activation': fused_act,
                      'qkv_packing': True,
                      'grad_reduce_in_fp32': False,
                      'attention_type': 'groupquery',
                      'activation_recompute': activation_recompute,
                      'pipeline_interleaving': ppint,
                      'optimizer_sharding': optimizer_sharding,
                      'tensor_par_comm_type': tensor_par_comm_type,
                      'tensor_par_overlap': tensor_par_overlap,
                      'seq_par_ag_redo': seq_par_ag_redo,
                      'data_par_overlap': data_par_overlap,
                      'weight_offload': weight_offload,
                      'activations_offload': activations_offload,
                      'optimizer_offload': optimizer_offload,
                      'training': True
                    }
                    if not debug:
                      try:
                        logger = logging.Logger('sub')
                        model = Llm(app, logger)
                        model.compile(
                          syst,
                          Llm.Execution.from_json(exe_json))
                        model.run(syst)
                        stats = model.get_stats_json(layers)
                        good_exe_count += 1
                        curr = (stats['sample_rate'], exe_json, stats)
                        best = OptimalExecution.update_list(best, curr,
                                                            top_n)
                      except Llm.Error as ex:
                        logger = logging.getLogger()
                        logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
                        bad_exe_count += 1
                    if mbs_break and good_exe_count == mbs_break_good:
                      break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp)

  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    current.sort(reverse=True, key=lambda x: x[0])
    return current[:quantity]


calculon.CommandLine.register(OptimalExecution)
