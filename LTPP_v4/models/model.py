import torch
import torch.nn as nn
import torch.nn.functional as F
from .softlogic import SoftLogicEngine
from constants import total_predicate_set, mental_predicate_set, action_predicate_set, grid_length


class Rule:
    def __init__(self, head, body_predicate_idx, rule_type, name=None):
        self.head = head
        self.body = list(body_predicate_idx)
        self.type = rule_type
        self.name = name if name is not None else f"r{self.head}_{'_'.join(map(str, self.body))}"


class LA_TPP_Model(nn.Module):
    def __init__(self, rules, mental_predicates, action_predicates,
                 predicate_list=total_predicate_set, d_pred=6, device='cpu',
                 learnable_K=False):
        super().__init__()
        self.device = device
        self.rules = rules
        self.predicate_list = list(predicate_list)
        self.pred_to_pos = {p: i for i, p in enumerate(self.predicate_list)}
        self.mental_predicates = mental_predicates
        self.action_predicates = action_predicates

        # 构建clause keys和映射
        self.clause_keys = []
        self.clause_key_to_rule = {}
        self.key_to_str = {}
        used_names = set()
        for idx, r in enumerate(rules):
            key = (r.head, idx)
            self.clause_keys.append(key)
            self.clause_key_to_rule[key] = r
            base_name = r.name
            name = base_name
            suffix = 0
            while name in used_names:
                suffix += 1
                name = f"{base_name}_{suffix}"
            used_names.add(name)
            self.key_to_str[key] = name

        # embedding参数
        self.d = d_pred
        K0 = torch.randn(len(self.predicate_list), self.d) * 0.1
        self.K = nn.Parameter(K0, requires_grad=bool(learnable_K))

        # 规则参数 - 改进初始化
        self.Thetas = nn.ParameterDict()
        for key in self.clause_keys:
            rule = self.clause_key_to_rule[key]
            h = len(rule.body)
            param_name = self.key_to_str[key]
            # 使用更好的初始化策略
            theta_init = torch.randn(self.d, h) * 0.1
            # 为每个slot添加一些随机性，但保持合理的相似度
            for slot_idx, pred_id in enumerate(rule.body):
                if pred_id in self.pred_to_pos:
                    pred_pos = self.pred_to_pos[pred_id]
                    # 让theta向对应谓词的K嵌入靠近
                    theta_init[:, slot_idx] += 0.3 * K0[pred_pos, :] + torch.randn(self.d) * 0.05
            self.Thetas[param_name] = nn.Parameter(theta_init)

        # mental state索引
        self.mental_to_idx = {p: i for i, p in enumerate(self.mental_predicates)}

        # 动力学参数 - 改进初始化
        self.alpha_un = nn.Parameter(torch.ones(len(self.mental_predicates)) * (-0.5))  # 较慢的衰减

        # gamma参数 - 更好的初始化
        self.gamma_AtoM = nn.ParameterDict()
        self.gamma_MtoA = nn.ParameterDict()
        for key in self.clause_keys:
            ks = self.key_to_str[key]
            self.gamma_AtoM[ks] = nn.Parameter(torch.zeros(len(self.mental_predicates)))
            self.gamma_MtoA[ks] = nn.Parameter(torch.zeros(len(self.action_predicates)))

            rule = self.clause_key_to_rule[key]
            if rule.head in self.mental_predicates:
                j_idx = self.mental_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_AtoM[ks].zero_()
                    self.gamma_AtoM[ks].data[j_idx] = 0.2  # 增强初始影响
            if rule.head in self.action_predicates:
                k_idx = self.action_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_MtoA[ks].zero_()
                    self.gamma_MtoA[ks].data[k_idx] = 0.2  # 增强初始影响

        self.b = nn.Parameter(torch.ones(len(self.action_predicates)) * (-2.0))  # 较低的基准强度
        self.w = nn.Parameter(0.2 * torch.randn(len(self.action_predicates), len(self.mental_predicates)))

        # 使用改进的softlogic引擎
        self.engine = SoftLogicEngine(
            predicate_list=self.predicate_list,
            d=self.d,
            K_init=None,
            T_match=0.6,  # 降低匹配温度，增强选择性
            tau=0.1,  # 稍微增大tau，使softmin更平滑
            beta=5.0  # 降低beta，使soft-OR更包容
        )
        # 共享K张量
        self.engine.K = self.K
        self.to(device)

    def get_theta(self, key):
        param_name = self.key_to_str[key]
        return self.Thetas[param_name]

    def build_val_from_history(self, H_t_events, M_vec, t_now, recent_window=None):
        """
        改进的history构建方法
        """
        if recent_window is None:
            recent_window = grid_length

        val = torch.zeros(len(self.predicate_list), device=self.K.device)

        # 填充mental states
        for p in self.mental_predicates:
            if p in self.pred_to_pos:
                val[self.pred_to_pos[p]] = M_vec[self.mental_to_idx[p]]

        # 标记最近的actions - 使用更宽松的时间窗口
        for (pred, ts) in H_t_events:
            if pred in self.pred_to_pos and pred in self.action_predicates:
                time_diff = t_now - ts
                # 使用软衰减而不是硬截断
                if time_diff <= recent_window + 1e-9:
                    # action在时间窗口内保持较高激活
                    decay_factor = max(0.1, 1.0 - (time_diff / recent_window) * 0.5)
                    val[self.pred_to_pos[pred]] = decay_factor

        return val

    def compute_S(self, g_cache):
        """
        计算A->M贡献的S向量
        """
        m = len(self.mental_predicates)
        S = torch.zeros(m, device=self.K.device)
        for key, g in g_cache.items():
            ks = self.key_to_str[key]
            gamma = self.gamma_AtoM[ks]
            # 确保g是标量tensor
            if isinstance(g, (int, float)):
                g = torch.tensor(g, device=self.K.device)
            S = S + gamma * g
        return F.relu(S)  # 确保S非负

    def alpha(self):
        return F.softplus(self.alpha_un)

    def compute_intensity_vectors(self, M_vec, val_vec, g_cache):
        """
        计算强度向量
        """
        linear = self.b + (self.w @ M_vec)
        lambda_ment = F.softplus(linear)

        lambda_logic = torch.zeros_like(lambda_ment)
        for key, g in g_cache.items():
            ks = self.key_to_str[key]
            gamma_vec = self.gamma_MtoA[ks]
            if isinstance(g, (int, float)):
                g = torch.tensor(g, device=self.K.device)
            lambda_logic = lambda_logic + gamma_vec * g

        lambda_logic = F.relu(lambda_logic)  # 确保非负
        return lambda_ment + lambda_logic, lambda_ment, lambda_logic

    def forward_chaining_all(self, val_init, M_vec, max_iters=4, return_trace=False, debug_verbose=False):
        Thetas = {}
        for key in self.clause_keys:
            Thetas[key] = self.get_theta(key)

        val_out, g_cache, trace = self.engine.forward_chaining_with_trace(
            Thetas, val_init, max_iters=max_iters, debug_verbose=debug_verbose
        )

        if return_trace:
            return val_out, g_cache, trace
        else:
            return val_out, g_cache

    def handle_event_sequence(self, events, recent_window=None, return_trace=False,
                              g_threshold=0.05, delta_threshold=1e-4, debug_verbose=False):
        """
        改进的事件序列处理方法，支持详细调试
        """
        if recent_window is None:
            recent_window = grid_length

        device = self.K.device
        M = torch.zeros(len(self.mental_predicates), device=device)
        history = []
        t_prev = 0.0
        total_loglik = torch.tensor(0.0, device=device)
        total_survival = torch.tensor(0.0, device=device)

        event_trace_list = []

        if debug_verbose:
            print(f"\n=== Processing event sequence with {len(events)} events ===")
            print("Rules defined:")
            for key in self.clause_keys:
                rule = self.clause_key_to_rule[key]
                print(f"  {self.key_to_str[key]}: {rule.body} -> {rule.head}")

        for event_idx, (action_id, t_i) in enumerate(events):
            delta_t = t_i - t_prev

            if debug_verbose:
                print(f"\n--- Event {event_idx}: action={action_id} at t={t_i:.3f} ---")
                print(f"Mental states before: {M.cpu().numpy()}")

            # 构建事件前的val向量
            val_pre = self.build_val_from_history(history, M, t_prev, recent_window)
            val_pre_fc, g_pre = self.forward_chaining_all(val_pre, M, debug_verbose=debug_verbose)
            S_prev = self.compute_S(g_pre)
            alpha_vec = self.alpha()

            # ODE积分
            M_minus = M + delta_t * (-alpha_vec * M + (1.0 - M) * S_prev)
            M_minus = torch.clamp(M_minus, 0.0, 1.0)

            # 事件前的强度计算
            val_pre_at_ti = self.build_val_from_history(history, M_minus, t_i - 1e-9, recent_window)
            val_fc_before, g_before = self.forward_chaining_all(val_pre_at_ti, M_minus)
            lambda_k, _, _ = self.compute_intensity_vectors(M_minus, val_fc_before, g_before)

            # 计算对数似然
            if action_id in self.action_predicates:
                k_idx = self.action_predicates.index(action_id)
                lam = lambda_k[k_idx]
                lam = torch.clamp(lam, min=1e-8)
                total_loglik = total_loglik + torch.log(lam)

            total_survival = total_survival + delta_t * torch.sum(lambda_k)

            # 将action添加到历史
            history.append((action_id, t_i))

            # 构建插入action后的val向量
            val_after_insert = self.build_val_from_history(history, M_minus, t_i, recent_window)

            if debug_verbose:
                print(f"Val after inserting action: {val_after_insert.cpu().numpy()}")
                print("Running forward chaining...")

            # 执行前向链式推理
            val_post, g_post, trace = self.forward_chaining_all(
                val_after_insert, M_minus, max_iters=8, return_trace=True, debug_verbose=debug_verbose
            )

            event_trace_list.append({
                'time': t_i,
                'action': action_id,
                'trace': trace
            })

            # 更新mental状态
            S_post = self.compute_S(g_post)
            M_plus = M_minus + delta_t * (-alpha_vec * M_minus + (1.0 - M_minus) * S_post)
            M_plus = torch.clamp(M_plus, 0.0, 1.0)

            if debug_verbose:
                print(f"Mental states after: {M_plus.cpu().numpy()}")
                print(f"Significant rule activations (g >= {g_threshold}):")
                for key, g_val in g_post.items():
                    if g_val >= g_threshold:
                        rule_name = self.key_to_str[key]
                        rule = self.clause_key_to_rule[key]
                        print(f"  {rule_name}: {rule.body} -> {rule.head}, g={g_val:.4f}")

            # 步进到下一时刻
            M = M_plus
            t_prev = t_i

        if return_trace:
            global_chain = self.extract_single_global_chain_from_events(
                event_trace_list, g_threshold=g_threshold, delta_threshold=delta_threshold
            )
            return total_loglik, total_survival, event_trace_list, global_chain
        else:
            return total_loglik, total_survival

    # 改进的全局链提取方法
    def extract_single_global_chain_from_events(self, event_trace_list, g_threshold=0.05, delta_threshold=1e-4):
        """
        改进的全局推理链提取，更准确地识别因果关系
        """
        global_chain = []
        tiny_g_inclusion = 1e-6

        for ev in event_trace_list:
            t = float(ev.get('time', 0.0))
            action = int(ev.get('action'))
            trace = ev.get('trace', []) or []

            # 添加触发action
            global_chain.append(f"action:{action}@{t:.3f}")
            active_set = set([action])

            # 按head分组trace条目
            head_map = {}
            for inf in trace:
                try:
                    head = int(inf.get('head'))
                except Exception:
                    continue
                head_map.setdefault(head, []).append(inf)

            # 计算每个head的最终激活值
            final_head_val = {}
            for head, recs in head_map.items():
                # 使用更新中最大的new_val作为final值
                updates = [r for r in recs if
                           (float(r.get('new_val', 0.0)) - float(r.get('prev_val', 0.0))) > delta_threshold]
                if updates:
                    final_head_val[head] = max(float(r.get('new_val', 0.0)) for r in updates)
                else:
                    # 如果没有显著更新，使用最大的g值
                    final_head_val[head] = max(float(r.get('g', 0.0)) for r in recs) if recs else 0.0

            # 构建候选列表
            candidates = []
            for inf in trace:
                g = float(inf.get('g', 0.0))
                prev_val = float(inf.get('prev_val', 0.0))
                new_val = float(inf.get('new_val', 0.0))
                delta = new_val - prev_val
                head = int(inf.get('head'))
                matched_raw = inf.get('matched_predicates', []) or []

                matched = []
                for p in matched_raw:
                    try:
                        matched.append(int(p))
                    except Exception:
                        pass

                # 更严格的包含条件
                significant_g = g >= g_threshold
                significant_delta = delta >= delta_threshold
                significant_final = final_head_val.get(head, 0.0) >= g_threshold

                if significant_g or significant_delta or significant_final:
                    candidates.append({
                        'head': head,
                        'matched': matched,
                        'g': g,
                        'delta': delta,
                        'new_val': new_val,
                        'final_head_val': final_head_val.get(head, 0.0),
                        'raw': inf
                    })

            # 贪婪扩展
            used_heads = set()
            while True:
                # 找到与active_set有连接且未使用的候选
                valid_candidates = [
                    c for c in candidates
                    if (set(c['matched']) & active_set) and (c['head'] not in used_heads)
                ]

                if not valid_candidates:
                    break

                # 按最终头部值和g值排序选择最佳候选
                valid_candidates.sort(
                    key=lambda x: (x['final_head_val'], x['g'], x['delta']),
                    reverse=True
                )

                chosen = valid_candidates[0]
                head = chosen['head']

                # 确定谓词类型标签
                if head in self.action_predicates:
                    tag = "action"
                elif head in self.mental_predicates:
                    tag = "mental"
                else:
                    tag = "pred"

                global_chain.append(f"{tag}:{head}@{t:.3f}")
                active_set.add(head)
                used_heads.add(head)

        return global_chain

    def extract_global_chains(self, event_trace_list, g_threshold=0.05, delta_threshold=1e-4):
        chain = self.extract_single_global_chain_from_events(
            event_trace_list, g_threshold=g_threshold, delta_threshold=delta_threshold
        )
        return [chain] if chain else []