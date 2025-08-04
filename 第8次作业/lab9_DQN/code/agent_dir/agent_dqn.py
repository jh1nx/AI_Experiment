import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from collections import deque
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        hidden1 = torch.relu(self.layer1(inputs))
        hidden2 = torch.relu(self.layer2(hidden1))
        return self.head(hidden2)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.storage = deque(maxlen=buffer_size)
        
    def __len__(self):
        return len(self.storage)

    def push(self, *transition):
        self.storage.append(transition)

    def sample(self, batch_size):
        return random.sample(self.storage, batch_size)

    def clean(self):
        self.storage.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        self.env = env
        self.args = args
        self.historical_returns = []

        # 设置随机种子
        self.seed = args.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # 配置设备（使用GPU如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化经验回放缓冲区和Q网络
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.q_net = QNetwork(env.observation_space.shape[0], args.hidden_size, env.action_space.n).to(self.device)
        self.target_q_net = QNetwork(env.observation_space.shape[0], args.hidden_size, env.action_space.n).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        
        # 初始化参数
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.update_target_freq = args.update_target_freq
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.lr_min = args.lr_min
        self.lr_decay = args.lr_decay
        
        self.train_step = 0
    
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        # 测试模式下不需要额外初始化
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        # 调用run方法实现训练过程
        self.run()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        # 将状态张量转移到正确的设备上
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # 如果是测试模式或符合贪婪策略条件，选择最佳动作
        if test or random.random() > self.epsilon:
            with torch.no_grad():
                action = self.q_net(state_tensor).max(1)[1].item()
        else:
            # 随机探索
            action = self.env.action_space.sample()
        
        return action

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        current_state, _ = self.env.reset()
        time_step = 0
        
        for episode_idx in range(1, self.args.n_frames + 1):
            terminated_flag = False
            episode_reward = 0
            
            while not terminated_flag:
                time_step += 1
                
                # 选择动作
                chosen_action = self.make_action(current_state, test=False)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(chosen_action)
                terminated_flag = terminated or truncated
                
                # 存储经验
                self.replay_buffer.push(current_state, chosen_action, reward, next_state, terminated_flag)
                
                # 如果经验缓冲区足够大，进行训练
                if len(self.replay_buffer) > self.batch_size:
                    self._update_network()
                    
                    # 根据频率更新目标网络
                    if time_step % self.update_target_freq == 0:
                        self.target_q_net.load_state_dict(self.q_net.state_dict())
                    
                    # 更新探索率
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                current_state = next_state
                episode_reward += reward
                
                if terminated_flag:
                    self.historical_returns.append(episode_reward)
                    current_state, _ = self.env.reset()
                    
                    # 每10个episode打印一次进度
                    if episode_idx % 10 == 0:
                        avg_reward = sum(self.historical_returns[-10:]) / min(10, len(self.historical_returns))
                        print(f"Episode: {episode_idx}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
    
    def _update_network(self):
        """更新Q网络"""
        # 从回放缓冲区中采样一批经验
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # 提取批次数据并确保它们在正确的设备上
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).to(self.device)
        rewards = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(np.array(batch[4])).to(self.device)
        
        # 计算当前Q值和目标Q值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
    
    def _adjust_learning_rate(self):
        """学习率衰减"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_min, param_group['lr'] * self.lr_decay)