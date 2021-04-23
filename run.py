import argparse
import sys
import time
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from modular.decide import Controller
from modular.lane_detection.lane import detect_lane

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map4')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--load-file', type=str, default=None)
args = parser.parse_args()

print('Loading map {} with seed {}'.format(args.map_name, args.seed))

env = DuckietownEnv(
    map_name = args.map_name,
    domain_rand = False,
    draw_bbox = False,
    max_steps = args.max_steps,
    seed = args.seed
)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

obs = env.reset()
print(obs.shape)

env.render()

total_reward = 0

from_file = False

if args.load_file is not None:
    actions = np.loadtxt(args.load_file, delimiter=',')
    from_file = True
else:
    actions = []

policy = Controller(has_pit=False, has_intersection=True, has_stop_sign=True) \
    if not from_file else None

done = False

if from_file:
    for (speed, steering) in actions:
        print(speed, steering)

        obs, reward, done, info = env.step([speed, steering])
        total_reward += reward
    
        print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

        env.render()

        time.sleep(0.1)
else:
    try:
        while not done:
            rgbobs = env.render('rgb_array')

            action = policy.predict(rgb_array=np.array(rgbobs), raw_obs=obs)
            print('--Action: ', action)
            actions.append(action)

            # Perform action
            obs, reward, done, _ = env.step(np.array(action))
            env.render()

            total_reward += reward

            print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f\n' % (env.step_count, reward, total_reward))
    except:
        print("Unexpected error:", sys.exc_info()[0])
    finally:
        print("Total Steps", len(actions))
        print("Total Reward", total_reward)

        # dump the controls using numpy
        np.savetxt('./results/{}_seed{}.txt'.format(args.map_name, args.seed), actions, delimiter=',')
