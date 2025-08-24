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

        # 规则参数
        self.Thetas = nn.ParameterDict()
        for key in self.clause_keys:
            rule = self.clause_key_to_rule[key]
            h = len(rule.body)
            param_name = self.key_to_str[key]
            theta_init = torch.randn(self.d, h) * 0.1
            for slot_idx, pred_id in enumerate(rule.body):
                if pred_id in self.pred_to_pos:
                    pred_pos = self.pred_to_pos[pred_id]
                    theta_init[:, slot_idx] += 0.3 * K0[pred_pos, :] + torch.randn(self.d) * 0.05
            self.Thetas[param_name] = nn.Parameter(theta_init)

        # mental state索引
        self.mental_to_idx = {p: i for i, p in enumerate(self.mental_predicates)}

        # 动力学参数
        self.alpha_un = nn.Parameter(torch.ones(len(self.mental_predicates)) * (-0.5))

        # gamma参数
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
                    self.gamma_AtoM[ks].data[j_idx] = 0.2
            if rule.head in self.action_predicates:
                k_idx = self.action_predicates.index(rule.head)
                with torch.no_grad():
                    self.gamma_MtoA[ks].zero_()
                    self.gamma_MtoA[ks].data[k_idx] = 0.2

        self.b = nn.Parameter(torch.ones(len(self.action_predicates)) * (-2.0))
        self.w = nn.Parameter(0.2 * torch.randn(len(self.action_predicates), len(self.mental_predicates)))

        self.engine = SoftLogicEngine(
            predicate_list=self.predicate_list,
            d=self.d,
            K_init=None,
            T_match=0.6,
            tau=0.1,
            beta=5.0
        )
        self.engine.K = self.K
        self.to(device)

    def get_theta(self, key):
        param_name = self.key_to_str[key]
        return self.Thetas[param_name]
    #
    # def compute_S(self, g_cache):
    #     """计算A->M贡献的S向量"""
    #     m = len(self.mental_predicates)
    #     S = torch.zeros(m, device=self.K.device)
    #     for key, g in g_cache.items():
    #         ks = self.key_to_str[key]
    #         gamma = self.gamma_AtoM[ks]
    #         if isinstance(g, (int, float)):
    #             g = torch.tensor(g, device=self.K.device)
    #         S = S + gamma * g
    #     return F.relu(S)
    #
    # def alpha(self):
    #     return F.softplus(self.alpha_un)
    #
    # def compute_intensity_vectors(self, M_vec, g_cache):
    #     """计算强度向量"""
    #     linear = self.b + (self.w @ M_vec)
    #     lambda_ment = F.softplus(linear)
    #
    #     lambda_logic = torch.zeros_like(lambda_ment)
    #     for key, g in g_cache.items():
    #         ks = self.key_to_str[key]
    #         gamma_vec = self.gamma_MtoA[ks]
    #         if isinstance(g, (int, float)):
    #             g = torch.tensor(g, device=self.K.device)
    #         lambda_logic = lambda_logic + gamma_vec * g
    #
    #     lambda_logic = F.relu(lambda_logic)
    #     return lambda_ment + lambda_logic, lambda_ment, lambda_logic
    #
    # def forward_chaining_all(self, val_init, max_iters=8, return_trace=False, debug_verbose=False):
    #     Thetas = {}
    #     for key in self.clause_keys:
    #         Thetas[key] = self.get_theta(key)
    #
    #     val_out, g_cache, trace = self.engine.forward_chaining_with_trace(
    #         Thetas, val_init, max_iters=max_iters, debug_verbose=debug_verbose
    #     )
    #
    #     if return_trace:
    #         return val_out, g_cache, trace
    #     else:
    #         return val_out, g_cache
    def build_val_from_history(self, H_t_events, M_vec, t_now, recent_window=None):
        """
        从历史事件和当前mental状态构建val向量
        H_t_events: list of (pred_id, timestamp) tuples
        M_vec: current mental state vector
        t_now: current time
        recent_window: time window for considering recent actions
        """
        if recent_window is None:
            recent_window = grid_length

        val = torch.zeros(len(self.predicate_list), device=self.K.device)

        # 填充当前的mental states
        for i, p in enumerate(self.mental_predicates):
            if p in self.pred_to_pos:
                val[self.pred_to_pos[p]] = M_vec[i]

        # 标记最近的actions
        for (pred_id, ts) in H_t_events:
            if pred_id in self.pred_to_pos and pred_id in self.action_predicates:
                time_diff = t_now - ts
                # 在时间窗口内的actions被激活
                if time_diff <= recent_window + 1e-9:
                    # 使用衰减因子
                    decay_factor = max(0.1, 1.0 - (time_diff / recent_window) * 0.5)
                    val[self.pred_to_pos[pred_id]] = decay_factor

        return val

    #计算ODE
    def compute_S(self, g_cache):
        """
        计算A->M规则贡献的excitation sum S向量
        g_cache: dict of {rule_key: g_value}
        """
        m = len(self.mental_predicates)
        S = torch.zeros(m, device=self.K.device)

        for key, g in g_cache.items():
            rule = self.clause_key_to_rule[key]
            # 只考虑A->M类型的规则（头部是mental predicate）
            if rule.head in self.mental_predicates:
                ks = self.key_to_str[key]
                gamma = self.gamma_AtoM[ks]

                # 确保g是tensor
                if isinstance(g, (int, float)):
                    g = torch.tensor(g, device=self.K.device)

                S = S + gamma * g

        return F.relu(S)  # 确保S非负

    def alpha(self):
        """返回mental state的衰减参数α"""
        return F.softplus(self.alpha_un)

    def compute_intensity_vectors(self, M_vec, val_vec, g_cache):
        """
        计算action intensity vectors
        M_vec: mental state vector
        val_vec: current val vector after forward chaining
        g_cache: rule activation cache
        """
        # Direct mental drive
        linear = self.b + (self.w @ M_vec)
        lambda_ment = F.softplus(linear)

        # Rule-based boost
        lambda_logic = torch.zeros_like(lambda_ment)
        for key, g in g_cache.items():
            rule = self.clause_key_to_rule[key]
            # 只考虑M->A类型的规则（头部是action predicate）
            if rule.head in self.action_predicates:
                ks = self.key_to_str[key]
                gamma_vec = self.gamma_MtoA[ks]

                if isinstance(g, (int, float)):
                    g = torch.tensor(g, device=self.K.device)

                lambda_logic = lambda_logic + gamma_vec * g

        lambda_logic = F.relu(lambda_logic)  # 确保非负
        total_intensity = lambda_ment + lambda_logic

        return total_intensity, lambda_ment, lambda_logic

    def forward_chaining_all(self, val_init, M_vec, max_iters=4, return_trace=False, debug_verbose=False):
        """
        执行前向链推理的包装方法
        """
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

    def handle_mixed_event_sequence(self, action_events, mental_events, return_trace=False,
                                    g_threshold=0.05, delta_threshold=1e-4, debug_verbose=False):
        """
        按照LA-TPP论文正确处理混合事件序列
        """
        device = self.K.device
        M = torch.zeros(len(self.mental_predicates), device=device)

        # 合并并排序所有事件
        all_events = []
        for action_id, t in action_events:
            all_events.append((t, 'action', action_id))
        for mental_id, t in mental_events:
            all_events.append((t, 'mental', mental_id))
        all_events.sort(key=lambda x: x[0])

        # 按时刻分组
        time_groups = {}
        for t, event_type, event_id in all_events:
            if t not in time_groups:
                time_groups[t] = {'actions': [], 'mentals': []}
            if event_type == 'action':
                time_groups[t]['actions'].append(event_id)
            else:
                time_groups[t]['mentals'].append(event_id)

        timestamps = sorted(time_groups.keys())

        total_loglik = torch.tensor(0.0, device=device)
        total_survival = torch.tensor(0.0, device=device)
        event_trace_list = []
        t_prev = 0.0

        if debug_verbose:
            print(f"\n=== LA-TPP Event Processing ===")
            print("Time groups:", {t: time_groups[t] for t in timestamps})
            print(f"Initial mental states: {M.cpu().numpy()}")

        for t_i in timestamps:
            delta_t = t_i - t_prev

            if debug_verbose:
                print(f"\n--- Processing time t={t_i:.3f} (Δt={delta_t:.3f}) ---")

            # Step 1: ODE Integration from t_{i-1}^+ to t_i^-
            if delta_t > 0:
                # 使用当前mental状态构建val用于计算S
                val_for_ode = torch.zeros(len(self.predicate_list), device=device)
                for j, p in enumerate(self.mental_predicates):
                    if p in self.pred_to_pos:
                        val_for_ode[self.pred_to_pos[p]] = M[j]

                _, g_for_ode = self.forward_chaining_all(val_for_ode, debug_verbose=False)
                S_current = self.compute_S(g_for_ode)
                alpha_vec = self.alpha()

                # ODE积分: dM/dt = -α*M + (1-M)*S
                M_new = M + delta_t * (-alpha_vec * M + (1.0 - M) * S_current)
                M = torch.clamp(M_new, 0.0, 1.0)

                if debug_verbose:
                    print(f"ODE evolution: {M.cpu().numpy()}")

            # Step 2: Compute intensity at t_i^- (before event)
            val_before = torch.zeros(len(self.predicate_list), device=device)
            for j, p in enumerate(self.mental_predicates):
                if p in self.pred_to_pos:
                    val_before[self.pred_to_pos[p]] = M[j]

            _, g_before = self.forward_chaining_all(val_before, debug_verbose=False)
            lambda_k, _, _ = self.compute_intensity_vectors(M, g_before)

            # Step 3: Compute log-likelihood for actions
            actions_at_t = time_groups[t_i]['actions']
            for action_id in actions_at_t:
                if action_id in self.action_predicates:
                    k_idx = self.action_predicates.index(action_id)
                    lam = lambda_k[k_idx]
                    lam = torch.clamp(lam, min=1e-8)
                    total_loglik = total_loglik + torch.log(lam)
                    if debug_verbose:
                        print(f"Action {action_id}: λ={lam:.4f}, log-lik={torch.log(lam):.4f}")

            total_survival = total_survival + delta_t * torch.sum(lambda_k)

            # Step 4: Insert events and perform forward chaining
            mentals_at_t = time_groups[t_i]['mentals']

            # 构建包含新事件的val向量
            val_with_events = torch.zeros(len(self.predicate_list), device=device)
            # 复制当前mental状态
            for j, p in enumerate(self.mental_predicates):
                if p in self.pred_to_pos:
                    val_with_events[self.pred_to_pos[p]] = M[j]

            # 标记新发生的actions和mentals为1
            for action_id in actions_at_t:
                if action_id in self.pred_to_pos:
                    val_with_events[self.pred_to_pos[action_id]] = 1.0
            for mental_id in mentals_at_t:
                if mental_id in self.pred_to_pos:
                    val_with_events[self.pred_to_pos[mental_id]] = 1.0

            if debug_verbose:
                print(f"Val before forward chaining: {val_with_events.cpu().numpy()}")

            # Step 5: Forward chaining (递归推理)
            val_after, g_after, trace = self.forward_chaining_all(
                val_with_events, max_iters=8, return_trace=True, debug_verbose=debug_verbose
            )

            if debug_verbose:
                print(f"Val after forward chaining: {val_after.cpu().numpy()}")
                print("Significant rule activations:")
                for key, g_val in g_after.items():
                    if g_val >= g_threshold:
                        rule_name = self.key_to_str[key]
                        rule = self.clause_key_to_rule[key]
                        print(f"  {rule_name}: {rule.body} -> {rule.head}, g={g_val:.4f}")

            # Step 6: Update mental states based on forward chaining results
            for j, p in enumerate(self.mental_predicates):
                if p in self.pred_to_pos:
                    new_val = float(val_after[self.pred_to_pos[p]])
                    M[j] = max(M[j].item(), new_val)  # 取最大值确保激活传播

            M = torch.clamp(M, 0.0, 1.0)

            if debug_verbose:
                print(f"Final mental states at t={t_i:.3f}: {M.cpu().numpy()}")

            # 记录trace
            event_trace_list.append({
                'time': t_i,
                'actions': actions_at_t,
                'mentals': mentals_at_t,
                'mental_states_after': M.clone(),
                'trace': trace
            })

            t_prev = t_i

        if return_trace:
            global_chain = self.extract_simple_reasoning_chain(
                event_trace_list, g_threshold=g_threshold, delta_threshold=delta_threshold
            )
            return total_loglik, total_survival, event_trace_list, global_chain
        else:
            return total_loglik, total_survival

    def extract_simple_reasoning_chain(self, event_trace_list, g_threshold=0.05, delta_threshold=1e-4):
        """
        提取简化的推理链：只显示主要的因果关系
        """
        chain = []

        for ev in event_trace_list:
            t = float(ev.get('time', 0.0))
            actions = ev.get('actions', [])
            mentals = ev.get('mentals', [])
            trace = ev.get('trace', [])

            # 添加直接事件
            for action_id in actions:
                chain.append(f"action:{action_id}@{t:.3f}")
            for mental_id in mentals:
                chain.append(f"mental:{mental_id}@{t:.3f}")

            # 添加通过推理激发的谓词
            activated_heads = set()
            for inf in trace:
                g = float(inf.get('g', 0.0))
                delta = float(inf.get('new_val', 0.0)) - float(inf.get('prev_val', 0.0))
                head = inf.get('head')

                if (g >= g_threshold or delta >= delta_threshold) and head not in activated_heads:
                    if head in self.mental_predicates:
                        chain.append(f"mental:{head}@{t:.3f}")
                        activated_heads.add(head)
                    elif head in self.action_predicates:
                        chain.append(f"action:{head}@{t:.3f}")
                        activated_heads.add(head)

        return chain

    def handle_event_sequence(self, events, recent_window=None, return_trace=False,
                              g_threshold=0.05, delta_threshold=1e-4, debug_verbose=False):
        """
        严格按照LA-TPP论文算法处理事件序列
        events: list of (action_id, time) tuples

        四步循环：
        1. Pre-event estimation: 计算 M(t_i^-) 和 λ_k(t_i^-)
        2. Action sampling and insertion: 计算log-likelihood
        3. Micro-reasoning-step: 递归前向链推理
        4. Post-event mental state: 更新 M(t_i^+)
        """
        if recent_window is None:
            recent_window = grid_length

        device = self.K.device
        M = torch.zeros(len(self.mental_predicates), device=device)  # M(0) = 0
        history = []  # H_t
        t_prev = 0.0
        total_loglik = torch.tensor(0.0, device=device)
        total_survival = torch.tensor(0.0, device=device)

        event_trace_list = []

        if debug_verbose:
            print(f"\n=== Processing event sequence with {len(events)} events ===")
            print(f"Initial mental states M(0): {M.cpu().numpy()}")

        for event_idx, (action_id, t_i) in enumerate(events):
            delta_t = t_i - t_prev

            if debug_verbose:
                print(f"\n{'=' * 50}")
                print(f"Event {event_idx}: action={action_id} at t={t_i:.3f}")
                print(f"Delta t: {delta_t:.3f}")
                print(f"Mental states M(t_{{{event_idx - 1 if event_idx > 0 else 0}}}^+): {M.cpu().numpy()}")

            # ============ Step 1: Pre-event estimation ============
            if debug_verbose:
                print(f"\n--- Step 1: Pre-event estimation ---")

            # 1.1: 使用ODE从 M(t_{i-1}^+) 积分到 M(t_i^-)
            if delta_t > 0:
                # 计算当前的S向量（基于上一时刻的状态）
                val_prev = self.build_val_from_history(history, M, t_prev, recent_window)
                val_prev_fc, g_prev = self.forward_chaining_all(val_prev, M, debug_verbose=False)
                S_prev = self.compute_S(g_prev)
                alpha_vec = self.alpha()

                # ODE积分: dM/dt = -α*M + (1-M)*S
                M_minus = M + delta_t * (-alpha_vec * M + (1.0 - M) * S_prev)
                M_minus = torch.clamp(M_minus, 0.0, 1.0)
            else:
                M_minus = M.clone()

            if debug_verbose:
                print(f"M(t_i^-) after ODE integration: {M_minus.cpu().numpy()}")

            # 1.2: 计算 λ_k(t_i^-)
            val_pre_event = self.build_val_from_history(history, M_minus, t_i - 1e-9, recent_window)
            val_pre_fc, g_pre = self.forward_chaining_all(val_pre_event, M_minus, debug_verbose=False)
            lambda_pre, lambda_ment_pre, lambda_logic_pre = self.compute_intensity_vectors(M_minus, val_pre_fc, g_pre)

            if debug_verbose:
                print(f"Intensities λ(t_i^-): {lambda_pre.cpu().numpy()}")

            # ============ Step 2: Action sampling and insertion ============
            if debug_verbose:
                print(f"\n--- Step 2: Action sampling and insertion ---")

            # 2.1: 计算log-likelihood
            if action_id in self.action_predicates:
                k_idx = self.action_predicates.index(action_id)
                lam_k = lambda_pre[k_idx]
                lam_k = torch.clamp(lam_k, min=1e-8)
                loglik_contrib = torch.log(lam_k)
                total_loglik = total_loglik + loglik_contrib

                if debug_verbose:
                    print(f"Log-likelihood contribution for action {action_id}: {float(loglik_contrib):.6f}")

            # 2.2: 累积survival integral
            survival_contrib = delta_t * torch.sum(lambda_pre)
            total_survival = total_survival + survival_contrib

            if debug_verbose:
                print(f"Survival integral contribution: {float(survival_contrib):.6f}")

            # 2.3: 插入action到历史中
            history.append((action_id, t_i))
            if debug_verbose:
                print(f"Updated history: {history}")

            # ============ Step 3: Micro-reasoning-step ============
            if debug_verbose:
                print(f"\n--- Step 3: Micro-reasoning-step ---")

            # 3.1: 构建插入action后的val向量
            val_after_insert = self.build_val_from_history(history, M_minus, t_i, recent_window)

            if debug_verbose:
                print(f"Val vector after action insertion: {val_after_insert.cpu().numpy()}")

            # 3.2: 执行递归前向链推理 (Equation 9)
            val_post_reasoning, g_post_reasoning, trace = self.forward_chaining_all(
                val_after_insert, M_minus, max_iters=8, return_trace=True, debug_verbose=debug_verbose
            )

            if debug_verbose:
                print(f"Val vector after reasoning: {val_post_reasoning.cpu().numpy()}")
                significant_rules = [(k, g) for k, g in g_post_reasoning.items() if g >= g_threshold]
                if significant_rules:
                    print("Significant rule activations:")
                    for key, g_val in significant_rules:
                        rule_name = self.key_to_str[key]
                        rule = self.clause_key_to_rule[key]
                        print(f"  {rule_name}: {rule.body} -> {rule.head}, g={g_val:.4f}")

            # ============ Step 4: Post-event mental state ============
            if debug_verbose:
                print(f"\n--- Step 4: Post-event mental state ---")

            # 4.1: 计算excitation sum S(t_i^+)
            S_post = self.compute_S(g_post_reasoning)

            if debug_verbose:
                print(f"Excitation sum S(t_i^+): {S_post.cpu().numpy()}")

            # 4.2: 更新 M(t_i^+) - 这里应该基于推理结果更新mental状态
            # 根据论文，mental状态应该反映推理后的结果
            M_plus = M_minus.clone()
            for i, pred_id in enumerate(self.mental_predicates):
                if pred_id in self.pred_to_pos:
                    pos = self.pred_to_pos[pred_id]
                    # 使用推理后的val作为新的mental状态
                    M_plus[i] = val_post_reasoning[pos]

            # 确保mental状态在[0,1]范围内
            M_plus = torch.clamp(M_plus, 0.0, 1.0)

            if debug_verbose:
                print(f"Updated mental states M(t_i^+): {M_plus.cpu().numpy()}")

            # 记录trace信息
            event_trace_list.append({
                'time': t_i,
                'action': action_id,
                'M_minus': M_minus.detach().cpu().numpy().tolist(),
                'M_plus': M_plus.detach().cpu().numpy().tolist(),
                'S_post': S_post.detach().cpu().numpy().tolist(),
                'trace': trace,
                'lambda_pre': lambda_pre.detach().cpu().numpy().tolist(),
                'loglik_contrib': float(loglik_contrib.detach()) if 'loglik_contrib' in locals() else 0.0
            })

            # 更新状态为下一轮迭代做准备
            M = M_plus
            t_prev = t_i

        if debug_verbose:
            print(f"\n{'=' * 50}")
            print(f"Final results:")
            print(f"Total log-likelihood: {float(total_loglik):.6f}")
            print(f"Total survival integral: {float(total_survival):.6f}")
            print(f"Final mental states: {M.cpu().numpy()}")

        if return_trace:
            global_chain = self.extract_reasoning_chain_from_traces(
                event_trace_list, g_threshold=g_threshold, delta_threshold=delta_threshold
            )
            return total_loglik, total_survival, event_trace_list, global_chain
        else:
            return total_loglik, total_survival

    def extract_reasoning_chain_from_traces(self, event_trace_list, g_threshold=0.05, delta_threshold=1e-4):
        """
        从trace中提取推理链，按照真实的因果关系构建
        """
        global_chain = []

        for trace_data in event_trace_list:
            t = trace_data['time']
            action = trace_data['action']
            trace = trace_data['trace']

            # 添加触发action
            global_chain.append(f"action:{action}@{t:.3f}")

            # 按推理顺序添加激活的mental states
            reasoning_steps = []
            for inf in trace:
                g = float(inf.get('g', 0.0))
                delta = float(inf.get('delta', 0.0))
                head = int(inf.get('head', -1))

                if (g >= g_threshold or delta >= delta_threshold) and head in self.mental_predicates:
                    reasoning_steps.append((head, g, delta, inf.get('iter', 0)))

            # 按iteration顺序排序
            reasoning_steps.sort(key=lambda x: x[3])

            for head, g, delta, iter_num in reasoning_steps:
                global_chain.append(f"mental:{head}@{t:.3f}")

        return global_chain