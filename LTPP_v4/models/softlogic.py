# models/softlogic.py
import math
import torch
import torch.nn.functional as F
from torch import nn


class SoftLogicEngine(nn.Module):
    def __init__(self, predicate_list, d=None, K_init=None, T_match=0.8, tau=0.05, beta=8.0):
        super().__init__()
        self.predicate_list = list(predicate_list)
        self.P = len(self.predicate_list)
        self.pred_to_pos = {p: i for i, p in enumerate(self.predicate_list)}
        self.T_match = T_match
        self.tau = tau
        self.beta = beta
        if K_init is not None:
            self.d = K_init.shape[1]
            self.K = nn.Parameter(K_init.clone(), requires_grad=False)
        else:
            self.d = self.P if d is None else d
            self.K = nn.Parameter(torch.randn(self.P, self.d) * 0.1, requires_grad=False)

    @staticmethod
    def softmin_tau(x, tau):
        # differentiable soft approximation of min
        n = x.shape[-1]
        return -tau * (torch.logsumexp(-x / tau, dim=-1) - math.log(n))

    def eval_single_clause(self, Theta_f, val_vec, T_match=None, tau=None):
        """
        Original simple evaluator returning g_f^{AND} scalar.
        """
        if T_match is None:
            T_match = self.T_match
        if tau is None:
            tau = self.tau
        Kn = F.normalize(self.K, dim=1)  # (P, d)
        Thetan = F.normalize(Theta_f, dim=0)  # (d, h)
        S = Kn @ Thetan  # (P, h)
        W = F.softmax(S / T_match, dim=0)  # (P, h)
        conf = (W * S).sum(dim=0)  # (h,)
        matched = (W * val_vec.unsqueeze(1)).sum(dim=0)  # (h,)
        x = torch.cat([conf, matched], dim=0)
        g_and = self.softmin_tau(x, tau)
        return torch.clamp(g_and, 0.0, 1.0)

    def soft_or(self, g_list, beta=None):
        """
        Soft OR (LogSumExp) over a list of scalar tensors.
        """
        if beta is None:
            beta = self.beta
        g_stack = torch.stack(g_list)
        return (1.0 / beta) * torch.logsumexp(beta * g_stack, dim=0)

    # -----------------------------
    # 修改的匹配和推理方法
    # -----------------------------
    def eval_single_clause_with_match(self, Theta_f, val_vec, T_match=None, tau=None):
        """
        改进的规则评估方法，解决推理链断裂问题
        """
        if T_match is None:
            T_match = self.T_match
        if tau is None:
            tau = self.tau

        Kn = F.normalize(self.K, dim=1)  # (P, d)
        Thetan = F.normalize(Theta_f, dim=0)  # (d, h)
        S = Kn @ Thetan  # (P, h)
        W = F.softmax(S / T_match, dim=0)  # (P, h)

        conf = (W * S).sum(dim=0)  # (h,)
        matched = (W * val_vec.unsqueeze(1)).sum(dim=0)  # (h,)
        x = torch.cat([conf, matched], dim=0)
        g_and = self.softmin_tau(x, tau)

        # 改进的argmax选择策略
        # 结合相似度分数和真值，但不让bias完全主导
        eps = 1e-6
        # 使用加权策略：70%相似度 + 30%真值bias
        similarity_weight = 0.7
        truth_weight = 0.3

        # 相似度分数 (normalized)
        sim_scores = F.softmax(S, dim=0)  # (P, h)

        # 真值bias (normalized)
        val_expanded = val_vec.unsqueeze(1).expand(-1, S.shape[1])  # (P, h)
        val_scores = F.softmax(val_expanded + eps, dim=0)  # (P, h)

        # 组合分数
        combined_scores = similarity_weight * sim_scores + truth_weight * val_scores

        argmax_pos = torch.argmax(combined_scores, dim=0)  # (h,) indices
        argmax_predicates = [self.predicate_list[int(idx)] for idx in argmax_pos]

        return torch.clamp(g_and, 0.0, 1.0), W, argmax_predicates

    def forward_chaining_with_trace(self, Thetas, initial_val, max_iters=6, tol=1e-6, return_W=False,
                                    debug_verbose=False):
        """
        改进的递归前向链式推理，增强调试功能和推理稳定性
        """
        device = initial_val.device
        val = initial_val.clone().to(device)
        g_cache = {}
        trace = []

        # 构建head到clauses的映射
        head_to_clauses = {}
        for k in Thetas.keys():
            head = k[0]
            head_to_clauses.setdefault(head, []).append(k)

        if debug_verbose:
            print(f"=== Forward chaining started with initial_val: {val.cpu().numpy()} ===")

        for it in range(max_iters):
            changed = False
            val_new = val.clone()
            iteration_changes = []

            if debug_verbose:
                print(f"\n--- Iteration {it} ---")
                print(f"Current val: {val.cpu().numpy()}")

            for head, clause_keys in head_to_clauses.items():
                clause_infos = []

                # 评估该head的所有子句
                for key in clause_keys:
                    Theta_f = Thetas[key].to(device)
                    g, W, argmax_preds = self.eval_single_clause_with_match(Theta_f, val)
                    clause_infos.append((key, g, W, argmax_preds))

                    if debug_verbose:
                        print(f"  Clause {key}: g={g.item():.6f}, matched={argmax_preds}")

                # 计算所有g值的soft-OR聚合
                g_list = [ci[1] for ci in clause_infos]
                g_stack = torch.stack(g_list)  # (R,)

                if g_stack.shape[0] == 1:
                    g_head = g_stack[0]
                else:
                    # 使用更稳定的soft-OR实现
                    beta_effective = max(self.beta, 1.0)  # 避免beta过小
                    g_head = (1.0 / beta_effective) * torch.logsumexp(beta_effective * g_stack, dim=0)

                pos = self.pred_to_pos[head]
                old_val = float(val[pos].item())

                # 更新策略：取max而不是简单加法，避免数值溢出
                new_val = max(old_val, float(g_head.item()))

                # 降低更新阈值，使推理更敏感
                update_threshold = max(tol, 1e-8)
                delta = new_val - old_val

                if delta > update_threshold:
                    val_new[pos] = new_val
                    changed = True

                    # 选择g值最高的子句作为触发子句
                    max_idx = int(torch.argmax(g_stack).item())
                    key_chosen, g_chosen, W_chosen, argmax_chosen = clause_infos[max_idx]

                    ev = {
                        'iter': int(it),
                        'clause_key': key_chosen,
                        'head': head,
                        'g': float(g_chosen.item()),
                        'matched_predicates': list(argmax_chosen),
                        'prev_val': old_val,
                        'new_val': new_val,
                        'delta': delta
                    }

                    if return_W:
                        ev['W'] = W_chosen.detach().cpu().numpy()

                    trace.append(ev)
                    iteration_changes.append((head, old_val, new_val, delta))

                    if debug_verbose:
                        print(f"  >> HEAD {head} updated: {old_val:.6f} -> {new_val:.6f} (delta={delta:.6f})")
                        print(f"     Triggered by clause {key_chosen} with matched={argmax_chosen}")

                # 更新g_cache (存储所有子句的g值)
                for key, g, _, _ in clause_infos:
                    g_cache[key] = float(g.item())

            val = val_new

            if debug_verbose:
                print(f"End of iteration {it}: changed={changed}, total_changes={len(iteration_changes)}")
                if iteration_changes:
                    for head, old, new, delta in iteration_changes:
                        print(f"  Changed: head={head}, {old:.6f}->{new:.6f} (Δ={delta:.6f})")

            if not changed:
                if debug_verbose:
                    print(f"Converged after {it + 1} iterations")
                break
        else:
            if debug_verbose:
                print(f"Reached max_iters={max_iters} without full convergence")

        if debug_verbose:
            print(f"=== Final val: {val.cpu().numpy()} ===")
            print(f"Total trace entries: {len(trace)}")

        return val, g_cache, trace

    # 保持向后兼容性的包装器
    def forward_chaining(self, Thetas, initial_val, max_iters=6, tol=1e-6):
        val, g_cache, _ = self.forward_chaining_with_trace(
            Thetas, initial_val, max_iters=max_iters, tol=tol, return_W=False, debug_verbose=False
        )
        return val, g_cache

