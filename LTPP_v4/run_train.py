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


# --- 改进的合成序列生成器 ---
def make_seq_with_reasoning_potential(n_events=6, t0=0.0):
    """
    生成具有更强推理潜力的序列，确保能触发多跳推理
    """
    seq = []
    t = t0

    # 第一个事件：action 5，应该能通过规则触发后续推理
    t += random.random() * 0.2 + 0.1
    seq.append({'time_since_start': t, 'type_event': 5})

    # 如果需要，可以添加一些mental事件来丰富推理路径
    if random.random() < 0.8:  # 80%概率添加mental 1
        seq.append({'time_since_start': t + 0.01, 'type_event': 1})

    if random.random() < 0.6:  # 60%概率添加mental 3
        seq.append({'time_since_start': t + 0.02, 'type_event': 3})

    # 添加更多events来测试长链推理
    for i in range(1, n_events):
        t += random.random() * 0.3 + 0.15

        # 交替或随机选择actions
        if i % 3 == 1:
            action = 6
        else:
            action = random.choice([5, 6])

        seq.append({'time_since_start': t, 'type_event': action})

        # 随机添加一些mental events
        if random.random() < 0.5:
            seq.append({'time_since_start': t + 0.01, 'type_event': random.choice([1, 2, 3])})

    return sorted(seq, key=lambda x: x['time_since_start'])


def build_dataset(n_seqs=10, seq_len=6):
    return [make_seq_with_reasoning_potential(seq_len) for _ in range(n_seqs)]


def seq_dicts_to_action_list(seq_dicts, action_predicate_set):
    """Return list of (action_id, time) for action events in seq_dicts, in chronological order."""
    out = []
    for e in sorted(seq_dicts, key=lambda x: x['time_since_start']):
        if e['type_event'] in action_predicate_set:
            out.append((int(e['type_event']), float(e['time_since_start'])))
    return out


def sanitize_event_traces_for_json(event_traces):
    sanitized = []
    for ev in event_traces:
        ev_s = {
            'time': float(ev.get('time', 0.0)),
            'action': int(ev.get('action', -1)),
            'chains': ev.get('chains', []),
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


def analyze_reasoning_chain(rules, chain):
    """
    分析推理链，验证逻辑一致性
    """
    print("=== Reasoning Chain Analysis ===")
    print("Chain:", " -> ".join(chain))

    # 解析链中的元素
    parsed_chain = []
    for item in chain:
        if '@' in item:
            pred_part, time_part = item.split('@')
            pred_type, pred_id = pred_part.split(':')
            parsed_chain.append((pred_type, int(pred_id), float(time_part)))

    # 验证规则触发逻辑
    print("\nRule validation:")
    for i, rule in enumerate(rules):
        body_preds = set(rule.body)
        head_pred = rule.head

        # 检查链中是否有满足此规则的序列
        for j in range(len(parsed_chain)):
            current_time = parsed_chain[j][2]
            # 在同一时刻查找body中的所有谓词
            same_time_preds = set()
            for k in range(len(parsed_chain)):
                if abs(parsed_chain[k][2] - current_time) < 0.01:  # 同一时刻
                    same_time_preds.add(parsed_chain[k][1])

            # 检查是否满足body条件
            if body_preds.issubset(same_time_preds):
                # 检查head是否在后续出现
                head_found = False
                for k in range(j, len(parsed_chain)):
                    if parsed_chain[k][1] == head_pred and abs(parsed_chain[k][2] - current_time) < 0.01:
                        head_found = True
                        break

                status = "✓" if head_found else "✗"
                print(f"  Rule {rule.name}: {rule.body} -> {rule.head} {status}")
                if body_preds.issubset(same_time_preds):
                    print(f"    Body satisfied at t={current_time:.3f}: {same_time_preds}")
                    if head_found:
                        print(f"    Head {head_pred} found in chain")
                    else:
                        print(f"    Head {head_pred} NOT found - potential issue!")


def main():
    outdir = 'outputs'
    os.makedirs(outdir, exist_ok=True)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # 构建数据集
    data = build_dataset(n_seqs=8, seq_len=6)
    ds = EventData(data)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=True, num_workers=0)

    # 定义规则 - 确保能形成完整推理链
    rules = [
        Rule(head=4, body_predicate_idx=[5, 1], rule_type='A->M', name='5_1->4'),  # action 5 + mental 1 -> mental 4
        Rule(head=2, body_predicate_idx=[4], rule_type='M->M', name='4->2'),  # mental 4 -> mental 2
        Rule(head=6, body_predicate_idx=[2, 4], rule_type='M->A', name='2_4->6'),  # mental 2 + mental 4 -> action 6
        Rule(head=1, body_predicate_idx=[6], rule_type='A->M', name='6->1'),  # action 6 -> mental 1
        Rule(head=3, body_predicate_idx=[1, 2], rule_type='M->M', name='1_2->3'),  # mental 1 + mental 2 -> mental 3
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
                    # 启用详细调试
                    res = model.handle_event_sequence(
                        example_action_list,
                        return_trace=True,
                        g_threshold=0.02,
                        delta_threshold=1e-6,
                        debug_verbose=(ep == n_epochs - 1)  # 最后一轮启用详细调试
                    )

                    if isinstance(res, tuple):
                        if len(res) == 2:
                            loglik, surv = res
                            event_traces = []
                            global_chain = []
                        elif len(res) == 3:
                            loglik, surv, event_traces = res
                            global_chain = []
                        elif len(res) == 4:
                            loglik, surv, event_traces, global_chain = res
                        else:
                            raise ValueError(f"Unexpected return tuple length: {len(res)}")
                    else:
                        raise ValueError("Unexpected non-tuple return")

                    print(f"Log-likelihood: {float(loglik):.4f}, Survival: {float(surv):.4f}")

                    # 分析推理trace
                    total_significant_activations = 0
                    for ev_idx, ev in enumerate(event_traces):
                        t = ev.get('time', 0.0)
                        action = ev.get('action', -1)
                        trace = ev.get('trace', [])

                        significant_activations = [
                            inf for inf in trace
                            if float(inf.get('g', 0.0)) >= 0.02 or
                               float(inf.get('delta', 0.0)) >= 1e-6
                        ]

                        total_significant_activations += len(significant_activations)
                        print(
                            f"  Event {ev_idx}: action={action}@{t:.3f}, significant activations={len(significant_activations)}")

                        for inf in significant_activations[:5]:  # 显示前5个
                            head = inf.get('head', 'unknown')
                            g = float(inf.get('g', 0.0))
                            delta = float(inf.get('delta', 0.0))
                            matched = inf.get('matched_predicates', [])
                            print(f"    -> head={head}, g={g:.4f}, delta={delta:.6f}, matched={matched}")

                    print(f"Total significant activations: {total_significant_activations}")

                    # 显示全局推理链
                    if global_chain:
                        print("Global reasoning chain:")
                        print("  " + " -> ".join(global_chain))
                        # 分析推理链逻辑
                        analyze_reasoning_chain(rules, global_chain)
                    else:
                        print("No global reasoning chain extracted")

                    # 保存详细trace
                    if ep == n_epochs - 1:  # 只在最后保存详细trace
                        sanitized = sanitize_event_traces_for_json(event_traces)
                        outpath = os.path.join(outdir, f"final_trace_epoch_{ep + 1}.json")
                        with open(outpath, 'w') as fh:
                            json.dump({
                                'epoch': ep + 1,
                                'loglik': float(loglik),
                                'survival': float(surv),
                                'global_chain': global_chain,
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

    # 手动测试一个简单的推理案例
    print("\n" + "=" * 60)
    print("MANUAL REASONING TEST")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # 构造一个理想情况：action 5发生，同时有mental 1
        test_events = [(5, 0.0)]  # 只有action 5在t=0

        print("Test events:", test_events)
        print("Expected reasoning chain based on rules:")
        print("  action:5 -> (should trigger via 5_1->4 if mental 1 exists)")
        print("  -> mental:4 -> (via 4->2) -> mental:2")
        print("  -> (via 2_4->6 if both 2 and 4 exist) -> action:6")
        print("  -> (via 6->1) -> mental:1")
        print("  -> (via 1_2->3 if both 1 and 2 exist) -> mental:3")

        res = model.handle_event_sequence(
            test_events,
            return_trace=True,
            g_threshold=0.01,
            delta_threshold=1e-6,
            debug_verbose=True
        )

        if len(res) >= 4:
            _, _, _, final_chain = res
            print(f"\nActual extracted chain: {' -> '.join(final_chain) if final_chain else 'None'}")
            if final_chain:
                analyze_reasoning_chain(rules, final_chain)
        else:
            print("No chain extracted in manual test")

    print(f"\nTraining completed. Results saved to {outdir}")


if __name__ == '__main__':
    main()