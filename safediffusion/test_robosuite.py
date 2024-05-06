import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="PickPlace",
    robots="Kinova3",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
)

env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) * 0.1
    obs, reward, done, info = env.step(action)
    env.render()

    if done:
        env.reset()