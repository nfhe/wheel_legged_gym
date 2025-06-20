# Copyright 2024 nfhe

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from wheeled_bipedal_gym.rsl_rl.modules import ActorCritic
from wheeled_bipedal_gym.rsl_rl.storage import RolloutStorage, RolloutStorageWithCost

class P3O:
    """P3O (Proximal Policy Optimization with Proximal Value Function) 算法实现"""
    actor_critic: ActorCritic
    def __init__(
        self,
        actor_critic,
        k_value,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        extra_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        kl_decay=0,
        device="cpu",
        # P3O特定参数
        p3o_alpha=0.1,  # 策略更新的保守程度参数
        p3o_beta=0.5,   # 价值函数更新的保守程度参数
        p3o_gamma=0.95, # P3O折扣因子
        cost_viol_loss_coef=1.0,
        cost_value_loss_coef=1.0,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.kl_decay = max(kl_decay, 0)
        self.schedule = schedule
        self.learning_rate = learning_rate

        # P3O特定参数
        self.p3o_alpha = p3o_alpha
        self.p3o_beta = p3o_beta
        self.p3o_gamma = p3o_gamma

        # P3O components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.actor.parameters()},
                {"params": self.actor_critic.critic.parameters()},
                {"params": self.actor_critic.std},
            ],
            lr=learning_rate,
        )
        self.extra_optimizer = None
        if self.actor_critic.is_sequence:
            self.extra_optimizer = optim.Adam(
                [
                    {"params": self.actor_critic.encoder.parameters()},
                ],
                lr=extra_learning_rate,
            )
        self.transition = RolloutStorage.Transition()

        # P3O parameters (继承PPO参数)
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.k_value = k_value
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        action_shape,
        cost_shape,
        cost_d_values
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            action_shape,
            cost_shape,
            cost_d_values,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        if self.actor_critic.is_sequence:
            self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
            latent = self.actor_critic.get_latent()
            critic_obs = torch.cat((critic_obs, latent), dim=-1)
        else:
            self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.clone()
        self.transition.observation_history = obs_history.clone()
        self.transition.critic_observations = critic_obs.clone()
        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
            self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_cost_returns(last_values, self.gamma, self.lam)

    def update_k_value(self,i):
        self.k_value = torch.min(torch.ones_like(self.k_value),self.k_value*(1.0004**i))
        return self.k_value

    def compute_surrogate_loss(self,actions_log_prob_batch,old_actions_log_prob_batch,advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss

    def compute_cost_surrogate_loss(self,actions_log_prob_batch,old_actions_log_prob_batch,cost_advantages_batch):
        # cost_advantages_batch : batch_size,num_type_costs
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        # (batch_size,num_type_costs) * (batch_size,1) = (batch_size,num_type_costs)
        surrogate = cost_advantages_batch*ratio.view(-1,1)
        surrogate_clipped = cost_advantages_batch*torch.clamp(ratio.view(-1,1), 1.0 - self.clip_param,1.0 + self.clip_param)
        # num_type_costs
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean(0)
        return surrogate_loss

    def compute_viol(self,actions_log_prob_batch,old_actions_log_prob_batch,cost_advantages_batch,cost_volation_batch):

        # compute cliped cost advantage
        cost_surrogate_loss = self.compute_cost_surrogate_loss(actions_log_prob_batch=actions_log_prob_batch,
                                                          old_actions_log_prob_batch=old_actions_log_prob_batch,
                                                          cost_advantages_batch=cost_advantages_batch)
        # compute the violation term,d_values :(num_type_costs)
        # cost_volation = (1-self.gamma)*(torch.squeeze(cost_returns_batch).mean() - self.d_values)
        cost_volation_loss = cost_volation_batch.mean()
        # combine the result
        cost_loss = cost_surrogate_loss + cost_volation_loss
        # do max and sum over
        #cost_loss = self.k_value*torch.sum(F.relu(cost_loss))
        cost_loss = torch.sum(self.k_value*F.relu(cost_loss))
        return cost_loss

    def compute_value_loss(self,target_values_batch,value_batch,returns_batch):
        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss

    def update(self, current_learning_iteration):
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_viol_loss = 0
        mean_surrogate_loss = 0
        obs_batch_max = -math.inf
        obs_batch_min = math.inf

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, obs_history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, target_cost_values_batch,cost_advantages_batch,cost_returns_batch,cost_violation_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                surrogate_loss = self.compute_surrogate_loss(actions_log_prob_batch=actions_log_prob_batch,
                                                         old_actions_log_prob_batch=old_actions_log_prob_batch,
                                                         advantages_batch=advantages_batch)

                # Cost voilation
                viol_loss = self.compute_viol(actions_log_prob_batch=actions_log_prob_batch,
                                old_actions_log_prob_batch=old_actions_log_prob_batch,
                                cost_advantages_batch=cost_advantages_batch,
                                cost_volation_batch=cost_violation_batch)
                # value function loss
                value_loss = self.compute_value_loss(target_values_batch=target_values_batch,
                                        value_batch=value_batch,
                                        returns_batch=returns_batch)

                # Cost value function loss
                cost_value_loss = self.compute_value_loss(target_values_batch=target_cost_values_batch,
                                                        value_batch=cost_value_batch,
                                                        returns_batch=cost_returns_batch)

                main_loss = surrogate_loss + self.cost_viol_loss_coef * viol_loss
                combine_value_loss = self.cost_value_loss_coef * cost_value_loss + self.value_loss_coef * value_loss
                entropy_loss = - self.entropy_coef * entropy_batch.mean()

                loss = main_loss + combine_value_loss + entropy_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_viol_loss += viol_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                current_max = obs_batch.max().item()
                current_min = obs_batch.min().item()
                if current_max > obs_batch_max:
                    obs_batch_max = current_max
                if current_min < obs_batch_min:
                    obs_batch_min = current_min

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_viol_loss /= num_updates
        mean_surrogate_loss /= num_updates

        self.storage.clear()

        return mean_value_loss,mean_cost_value_loss,mean_viol_loss,mean_surrogate_loss,obs_batch_min,obs_batch_max
