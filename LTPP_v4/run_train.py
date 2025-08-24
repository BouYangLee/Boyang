# run_train.py
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import mental_predicate_set, action_predicate_set, total_predicate_set
from dataset import EventData, collate_fn
from models.model import LA_TPP_Model, Rule
from trainer import Trainer
from utils import plot_loss, plot_W_history
import torch.nn.functional as F


def make_discrete_seq():
    """
    生成测试序列：t=0时action 5，t=0.5时action 6
    注意：在LA-TPP中，mental states不是输入事件，而是通过推理推导出来的
    """
    seq = [
        {'time_since_start': 0.0, 'type_event': 5},  # action 5 at t=0
        {'time_since_start': 0.5, 'type_event': 6},  # action 6 at t=0.5
    ]
    return seq


def build_dataset(n_seqs=10):
    """构建数据集，每个序列都是相同的离散时间模式"""
    return [make_discrete_seq() for _ in range(n_seqs)]


def seq_dicts_to_action_list(seq_dicts, action_predicate_set):
    """Return list of (action_id, time) for action events in seq_dicts, in chronological order."""
    out = []
    for e in sorted(seq_dicts, key=lambda x: x['time_since_start']):
        if e['type_event'] in action_predicate_set:
            out.append((int(e['type_event']), float(e['time_since_start'])))
    return out


def seq_dicts_to_all_events(seq_dicts):
    """Return list of (event_id, time) for all events, sorted by time"""
    out = []
    for e in sorted(seq_dicts, key=lambda x: x['time_since_start']):
        out.append((int(e['type_event']), float(e['time_since_start'])))
    return out


def sanitize_event_traces_for_json(event_traces):
    sanitized = []
    for ev in event_traces:
        ev_s = {
            'time': float(ev.get('time', 0.0)),
            'events': ev.get('events', []),
            'mental_before': ev.get('mental_before', []),
            'mental_after': ev.get('mental_after', []),
            'reasoning_chain': ev.get('reasoning_chain', []),
        }
        trace_list = []
        for inf in ev.get('trace', []):
            inf_s = {
                'iter': int(inf.get('iter', 0)),
                'clause_key': inf.get('clause_key'),
                'head': inf.get('head'),
                'g': float(inf.get('g', 0.0)),
                'matched_predicates': inf.get('matched_predicates', []),
                'prev_val': float(inf.get('prev_val', 0.0)),
                'new_val': float(inf.get('new_val', 0.0)),
                'delta': float(inf.get('delta', 0.0))
            }
            trace_list.append(inf_s)
        ev_s['trace'] = trace_list
        sanitized.append(ev_s)
    return sanitized


def analyze_reasoning_logic(rules, trace_data, g_threshold=0.01):
    """
    分析推理逻辑，构建真实的因果链
    trace_data: list of event trace dictionaries
    """
    if not trace_data:
        return []

    print("=== Reasoning Logic Analysis ===")

    reasoning_chains = []

    for timestep in trace_data:
        t = timestep.get('time', 0.0)
        action = timestep.get('action', -1)
        trace = timestep.get('trace', [])

        print(f"\nTime t={t}:")
        print(f"Action: {action}")

        # 分析激活的规则
        active_rules = []
        for inf in trace:
            g = float(inf.get('g', 0.0))
            delta = float(inf.get('delta', 0.0))
            if g >= g_threshold or delta >= 1e-6:
                head = inf.get('head')
                matched = inf.get('matched_predicates', [])
                active_rules.append((head, matched, g, delta))

        print(f"Active rules:")
        for head, matched, g, delta in active_rules:
            print(f"  {matched} -> {head}, g={g:.4f}, delta={delta:.6f}")

        # 构建时间步的推理链
        timestep_chain = [f"action:{action}"]

        for head, matched, g, delta in active_rules:
            if g >= g_threshold:
                timestep_chain.append(f"pred:{head}")

        reasoning_chains.append({
            'time': t,
            'chain': timestep_chain,
            'active_rules': active_rules
        })

    return reasoning_chains


def main():
    outdir = 'outputs'
    os.makedirs(outdir, exist_ok=True)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # 构建数据集
    data = build_dataset(n_seqs=8)
    ds = EventData(data)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=True, num_workers=0)

    # 使用您预定义的规则
    rules = [
        Rule(head=4, body_predicate_idx=[5, 2, 3], rule_type='M->M', name='5_2_3->4'),
        Rule(head=4, body_predicate_idx=[5, 1], rule_type='M->M', name='5_1->4'),
        Rule(head=2, body_predicate_idx=[1, 4], rule_type='M->M', name='1_4->2'),
        Rule(head=6, body_predicate_idx=[2, 1, 4], rule_type='M->A', name='2_1_4->6'),
        Rule(head=1, body_predicate_idx=[3, 6], rule_type='A->M', name='3_6->1')
    ]

    print("=== Defined Rules ===")
    for rule in rules:
        print(f"{rule.name}: {rule.body} -> {rule.head} ({rule.type})")
    print()

    device = 'cpu'
    model = LA_TPP_Model(
        rules=rules,
        mental_predicates=mental_predicate_set,
        action_predicates=action_predicate_set,
        predicate_list=total_predicate_set,
        d_pred=6,
        device=device,
        learnable_K=False
    )

    trainer = Trainer(model, lr=3e-3, device=device)

    n_epochs = 40
    eval_interval = 5
    example_idx = 0

    losses = []
    param_names = [model.key_to_str[k] for k in model.clause_keys]
    W_history = {name: [] for name in param_names}

    print("=== Starting Training ===")
    for ep in range(n_epochs):
        avg_loss = trainer.train_epoch(loader)
        losses.append(avg_loss)

        # 记录W历史
        for key in model.clause_keys:
            param_name = model.key_to_str[key]
            Theta = model.get_theta(key).detach()
            Kn = F.normalize(model.K.detach(), dim=1)
            Thetan = F.normalize(Theta, dim=0)
            S = Kn @ Thetan
            W = F.softmax(S / model.engine.T_match, dim=0)
            W_sum = W.sum(dim=1)
            W_history[param_name].append(W_sum.cpu().numpy())

        print(f"Epoch {ep + 1}/{n_epochs}, avg loss={avg_loss:.4f}")

        # 定期评估和调试
        if (ep + 1) % eval_interval == 0 or ep == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                example_seq_dicts = data[example_idx]
                example_action_list = seq_dicts_to_action_list(example_seq_dicts, action_predicate_set)

                print(f"\n=== Evaluation at Epoch {ep + 1} ===")
                print("Example sequence actions:", example_action_list)

                try:
                    # 使用修正后的事件处理方法，严格按照LA-TPP算法
                    res = model.handle_event_sequence(
                        example_action_list,
                        return_trace=True,
                        g_threshold=0.02,
                        delta_threshold=1e-6,
                        debug_verbose=(ep == n_epochs - 1)  # 最后一轮启用详细调试
                    )

                    print(f"Log-likelihood: {float(loglik):.4f}, Survival: {float(surv):.4f}")

                    # 分析推理逻辑
                    reasoning_chains = analyze_reasoning_logic(rules, event_traces, g_threshold=0.01)

                    print("\n=== Reasoning Chains ===")
                    for chain_info in reasoning_chains:
                        t = chain_info['time']
                        chain = chain_info['chain']
                        print(f"t={t}: {' -> '.join(chain)}")

                    # 保存详细trace
                    if ep == n_epochs - 1:
                        sanitized = sanitize_event_traces_for_json(event_traces)
                        outpath = os.path.join(outdir, f"final_trace_epoch_{ep + 1}.json")
                        with open(outpath, 'w') as fh:
                            json.dump({
                                'epoch': ep + 1,
                                'loglik': float(loglik),
                                'survival': float(surv),
                                'reasoning_chains': reasoning_chains,
                                'event_traces': sanitized
                            }, fh, indent=2)
                        print(f"Detailed trace saved to {outpath}")

                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()

            model.train()

    # 最终图表
    plot_loss(losses, outpath=os.path.join(outdir, 'loss.png'))
    plot_W_history(W_history, predicate_list=total_predicate_set, outdir=outdir)

    print(f"\nTraining completed. Results saved to {outdir}")


if __name__ == '__main__':
    main()