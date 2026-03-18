import copy
import numpy as np
import torch
from torch.distributions import Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class PixelWiseA3C_InnerState:
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        t_max,
        gamma,
        beta=1e-2,
        phi=lambda x: x,
        pi_loss_coef=1.0,
        v_loss_coef=0.5,
        average_reward_tau=1e-2,
        act_deterministically=False,
        average_entropy_decay=0.999,
        average_value_decay=0.999,
        grad_clip_norm=None,
        eps=1e-8,
    ):
        self.shared_model = model.to(device)
        self.model = copy.deepcopy(self.shared_model).to(device)
        self.sync_parameters()

        self.optimizer = optimizer
        self.batch_size = batch_size

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.grad_clip_norm = grad_clip_norm
        self.eps = eps

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.average_reward = 0

        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

    def _clear_memory(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def _to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device)
        x = self.phi(x)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _forward_model(self, statevar):
        out = self.model(statevar)
        if isinstance(out, (tuple, list)):
            if len(out) == 2:
                pout, vout = out
            elif len(out) >= 3:
                # 兼容旧版 forward: (_, value, policy) 或 (policy, value, aux)
                # 优先取形状更像 policy/value 的两个张量。
                candidates = [x for x in out if torch.is_tensor(x)]
                if len(candidates) < 2:
                    raise ValueError("Model forward returned insufficient tensor outputs.")
                pout, vout = candidates[0], candidates[1]
            else:
                raise ValueError("Model forward returned an empty tuple/list.")
        else:
            raise ValueError("Model forward must return (policy, value) or compatible tuple/list.")
        return pout, vout

    def _normalize_policy(self, pout):
        if torch.isnan(pout).any() or torch.isinf(pout).any():
            raise FloatingPointError("NaN/Inf detected in policy output.")

        # 若已是概率分布则直接轻微裁剪；否则按 logits 做 softmax。
        with torch.no_grad():
            sum_prob = pout.sum(dim=1, keepdim=True)
            looks_like_prob = (
                torch.all(pout >= 0).item()
                and torch.allclose(sum_prob, torch.ones_like(sum_prob), atol=1e-4, rtol=1e-4)
            )

        if looks_like_prob:
            probs = pout.clamp_min(self.eps)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)
        else:
            probs = torch.softmax(pout, dim=1)
            probs = probs.clamp_min(self.eps)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return probs

    def sync_parameters(self):
        self.model.load_state_dict(self.shared_model.state_dict())

    def update_grad(self, target, source):
        target_params = dict(target.named_parameters())
        for name, src_param in source.named_parameters():
            tgt_param = target_params[name]
            if src_param.grad is None:
                tgt_param.grad = None
            else:
                tgt_param.grad = src_param.grad.detach().clone().to(tgt_param.device)

    def update(self, statevar):
        assert self.t_start < self.t

        if statevar is None:
            R = torch.zeros_like(self.past_values[self.t - 1])
        else:
            _, vout = self._forward_model(statevar)
            R = vout.detach()

        pi_loss = 0.0
        v_loss = 0.0

        for i in reversed(range(self.t_start, self.t)):
            R = self.gamma * R + self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            pi_loss = pi_loss - log_prob * advantage.detach() - self.beta * entropy
            v_loss = v_loss + 0.5 * (v - R).pow(2)

        if self.pi_loss_coef != 1.0:
            pi_loss = pi_loss * self.pi_loss_coef
        if self.v_loss_coef != 1.0:
            v_loss = v_loss * self.v_loss_coef

        total_loss = (pi_loss + v_loss).mean()
        print(f"pi_loss:{pi_loss.mean().item():.6f}   v_loss:{v_loss.mean().item():.6f}")

        # 1) 清 local grad
        self.model.zero_grad(set_to_none=True)
        # 2) 清 shared grad
        self.optimizer.zero_grad(set_to_none=True)
        # 3) backward on local model
        total_loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        # 4) 将 local grad 拷贝到 shared model
        self.update_grad(self.shared_model, self.model)
        # 5) optimizer step on shared model
        self.optimizer.step()
        # 6) sync shared -> local
        self.sync_parameters()

        self._clear_memory()
        self.t_start = self.t

    def act_and_train(self, state, reward):
        statevar = self._to_tensor(state)
        reward_tensor = self._to_tensor(reward)

        if self.t > 0:
            self.past_rewards[self.t - 1] = reward_tensor

        if self.t - self.t_start == self.t_max:
            self.update(statevar)

        self.past_states[self.t] = statevar
        pout, vout = self._forward_model(statevar)
        probs = self._normalize_policy(pout)

        p_trans = probs.permute(0, 2, 3, 1).contiguous()
        if torch.isnan(p_trans).any() or torch.isinf(p_trans).any():
            raise FloatingPointError("NaN/Inf detected in transformed policy tensor.")

        dist = Categorical(probs=p_trans)
        if self.act_deterministically:
            action = p_trans.argmax(dim=-1)
        else:
            action = dist.sample()

        action_ch = action.unsqueeze(1)
        action_prob = probs.gather(1, action_ch).clamp_min(self.eps)
        log_action_prob = action_prob.log()
        entropy = dist.entropy().unsqueeze(1)

        self.past_action_log_prob[self.t] = log_action_prob
        self.past_action_entropy[self.t] = entropy
        self.past_values[self.t] = vout
        self.t += 1

        return action.detach().cpu().numpy()

    def stop_episode_and_train(self, state, reward, done=False):
        reward_tensor = self._to_tensor(reward)
        if self.t > 0:
            self.past_rewards[self.t - 1] = reward_tensor

        if done:
            self.update(None)
        else:
            statevar = self._to_tensor(state)
            self.update(statevar)

    def act(self, state):
        statevar = self._to_tensor(state)
        with torch.no_grad():
            pout, _ = self._forward_model(statevar)
            probs = self._normalize_policy(pout)
            p_trans = probs.permute(0, 2, 3, 1).contiguous()

            if self.act_deterministically:
                action = p_trans.argmax(dim=-1)
            else:
                dist = Categorical(probs=p_trans)
                action = dist.sample()

        return action.detach().cpu().numpy()
