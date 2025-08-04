import argparse
import warnings
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from agent_dir.agent_dqn import AgentDQN

# 忽略无关警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 设置matplotlib中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]


def parse():
    parser = argparse.ArgumentParser(description="Run Experiments with Different Seeds")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1111, 2222, 6179], 
        help="list of seeds",
    )
    parser.add_argument(
        "--log_dir", default="./result", type=str, help="directory for results"
    ) 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse()
    
    # 确保结果目录存在
    os.makedirs(config.log_dir, exist_ok=True)

    experiment_results = []  # 存储所有实验的奖励记录
    # 对于每个种子，运行一次完整实验
    for current_seed in config.seeds:
        # 创建CartPole环境
        environment = gym.make("CartPole-v0", render_mode="rgb_array")
        # 将参数直接传递给Agent
        agent_config = argparse.Namespace(
            env_name="CartPole-v0",
            hidden_size=128,         # 神经网络隐藏层大小
            lr=0.001,                # 初始学习率
            lr_min=0.0001,           # 最小学习率限制
            lr_decay=0.9,            # 学习率衰减因子
            lr_decay_freq=500,       # 每500步衰减一次学习率
            gamma=0.95,              # 折扣因子，用于计算未来奖励的权重
            n_frames=50,             # 训练的总帧数（episode数量）
            seed=current_seed,       # 设置随机种子，确保实验可复现
            batch_size=128,          # 每次训练的批量大小
            buffer_size=4000,        # 经验回放缓冲区大小
            update_target_freq=10,   # 目标网络更新频率
            epsilon=1.0,             # 初始探索率（ε-贪心策略）
            epsilon_min=0.0001,      # 最小探索率
            epsilon_decay=0.996,     # 探索率衰减因子
            log_dir=config.log_dir,
        )

        # 创建并训练DQN智能体
        dqn_agent = AgentDQN(environment, agent_config)
        dqn_agent.train()
        episode_returns = dqn_agent.historical_returns  # 获取训练过程中的奖励记录
        experiment_results.append(episode_returns)

    # 统一所有奖励序列的长度（取最短的长度）
    shortest_len = min([len(returns) for returns in experiment_results])
    experiment_results = [returns[:shortest_len] for returns in experiment_results]

    # 转换为numpy数组，便于计算统计量
    experiment_results = np.array(experiment_results)
    return_deviation = np.std(experiment_results, axis=0)  # 计算每个时间步的奖励标准差

    # 创建结果可视化图表
    plt.figure()

    # 绘制每个实验的奖励曲线
    for trial_idx, returns in enumerate(experiment_results):
        plt.plot(returns[:shortest_len], label=f"实验{trial_idx+1} 奖励")

    # 绘制奖励标准差曲线
    plt.plot(return_deviation[:shortest_len], label="奖励标准差", color="gray")

    # 添加图表元素
    plt.legend()
    plt.xlabel("批次")
    plt.ylabel("奖励值")
    plt.title("实验奖励和标准差")

    # 保存图表到文件
    plt.savefig(config.log_dir + "/result.png")
    
    # 创建子图布局的结果图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("DQN训练结果分析", fontsize=16)
    
    # 绘制三次实验的结果，每个实验一个子图
    for trial_idx in range(3):
        row, col = trial_idx // 2, trial_idx % 2
        current_ax = axs[row, col]
        current_ax.plot(experiment_results[trial_idx][:shortest_len], label=f"实验{trial_idx+1}")
        current_ax.set_title(f"实验{trial_idx+1}的训练奖励")
        current_ax.set_xlabel("批次")
        current_ax.set_ylabel("奖励值")
        current_ax.legend()
        current_ax.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制奖励标准差
    std_ax = axs[1, 1]
    std_ax.plot(return_deviation[:shortest_len], color="red")
    std_ax.set_title("三次实验的奖励标准差")
    std_ax.set_xlabel("批次")
    std_ax.set_ylabel("标准差")
    std_ax.set_ylim(0, 200)  # 设置标准差纵坐标最大值为200
    std_ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存子图结果
    plt.savefig(config.log_dir + "/module.png")
    plt.close()

    # 打印结果数据
    print(f"Rewards: {experiment_results}")
    print(f"奖励标准差: {return_deviation}")
