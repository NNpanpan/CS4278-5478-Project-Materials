from gym_duckietown.envs import DuckietownEnv

def launch_env(map_name='map5', max_steps=1500, seed=11):
    env = DuckietownEnv(
    map_name = map_name,
    domain_rand = False,
    draw_bbox = False,
    max_steps = max_steps,
    seed = seed
    )
