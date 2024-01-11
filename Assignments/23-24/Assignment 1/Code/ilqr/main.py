import argparse
import gymnasium as gym
import autograd.numpy as np
from matplotlib import pyplot as plt


from student import ILqr, pendulum_dyn, cost


def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    render = args.render
    ilqr = ILqr(pendulum_dyn, cost, horizon=20)

    env = gym.make("Pendulum-v1", render_mode="human" if render else "rgb_array")

    rewards = []
    for i in range(args.episodes):
        print(f'Starting Episode {i}')
        episode_reward = episode(env, ilqr)
        print(f'Episode {i} reward: {episode_reward}')
        rewards.append(episode_reward)
    print(f'Average reward: {sum(rewards) / len(rewards)}')

    env.close()


def episode(env, ilqr):
    state, _ = env.reset()
    total_reward = 0.0

    u_seq = [np.array((0.0,)) for _ in range(ilqr.horizon)]

    st_seq = []
    th = np.arctan2(state[1],state[0])
    st_seq.append(np.array((th,state[2])))

    c_seq = []
    plt.figure(figsize=(15,5))
    done = False
    while not done:
        
        x_seq = [np.array((th,state[2]))]
        c_seq.append(cost(x_seq[0],u_seq[0]))
        
        for t in range(ilqr.horizon):
            x_seq.append(pendulum_dyn(x_seq[-1], u_seq[t]))

        # do 3 iterations of iLQR
        for rep in range(3):
            k_seq, K_seq = ilqr.backward(x_seq, u_seq)
            x_seq, u_seq = ilqr.forward(x_seq, u_seq, k_seq, K_seq)
        
        st_seq.append(np.array((th,state[2])))
        
        state,reward,term,trunc,_ = env.step(u_seq[0])
        done = term or trunc
        total_reward += reward
        
        th = np.arctan2(state[1],state[0])

    return total_reward


if __name__ == "__main__":
    main()
