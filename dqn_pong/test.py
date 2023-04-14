import gym
from  replay_memory import ReplayBuffer
from utils import RepeatActionAndMaxFrame

def test_env():
    name = 'PongDeterministic-v4'
    env = RepeatActionAndMaxFrame(gym.make(name))
    batch_size = 6
    mem_size = 48
    r = ReplayBuffer(mem_size, env.observation_space.shape, env.action_space)
    T = 10
    s_t = env.reset()[0]
    for t in range(T):
        a_t = env.action_space.sample()
        s_t_, r_t, done, info = env.step(a_t)
        r.store_transition(s_t, a_t, r_t,s_t_, done)
        print(s_t)
        s_t = s_t_
        if done:
            s_t = env.reset()
        if r.mem_cntr >= batch_size:
            states, actions, rewards, states_, dones  = r.sample_memory(batch_size)
    assert(True)




