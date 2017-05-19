import cv2
from gym.spaces.box import Box
import numpy as np
import gym
from gym import spaces
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger, Monitor
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()


from gym.envs.registration import register

register(
    id='wob.mini.DebugChaseCircle-v0',
    entry_point='debug_chase_circle_env:DebugChaseCircleEnv',
    max_episode_steps=200,
    tags={
        'debug': True
    },
    kwargs={
        'horizon': 200
    }
)

register(
    id='wob.mini.DebugStationaryChaseCircle-v0',
    entry_point='debug_chase_circle_env:DebugChaseCircleEnv',
    max_episode_steps=200,
    tags={
        'debug': True
    },
    kwargs={
        'horizon': 200,
        'velscale': 0
    }
)

register(
    id='wob.mini.DebugStationaryScaledRewardChaseCircle-v0',
    entry_point='debug_chase_circle_env:DebugChaseCircleEnv',
    max_episode_steps=200,
    tags={
        'debug': True
    },
    kwargs={
        'horizon': 200,
        'velscale': 0,
        'scaled_reward': True
    }
)

register(
    id='wob.mini.DebugStationaryCenteredChaseCircle-v0',
    entry_point='debug_chase_circle_env:DebugChaseCircleEnv',
    max_episode_steps=200,
    tags={
        'debug': True
    },
    kwargs={
        'horizon': 200,
        'velscale': 0,
        'centered': True
    }
)

register(
    id='wob.mini.DebugStationaryScaledRewardCenteredChaseCircle-v0',
    entry_point='debug_chase_circle_env:DebugChaseCircleEnv',
    max_episode_steps=200,
    tags={
        'debug': True
    },
    kwargs={
        'horizon': 200,
        'velscale': 0,
        'scaled_reward': True,
        'centered': True
    }
)

def create_env(env_id, client_id, remotes, **kwargs):
    spec = gym.spec(env_id)

    print(spec)
    print(spec.tags)

    if spec.tags.get('flashgames', False):
        return create_flash_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('wob', False):
        return create_wob_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('debug', False):
        return create_debug_env(env_id)
    else:
        # Assume atari.
        assert "." not in env_id  # universe environments have dots in names.
        return create_atari_env(env_id)

def create_debug_env(env_id):
    env = gym.make(env_id)
    env = Monitor(env, 'videos/', force=True)
    env = Logger(env)
    # height and width the same for all mini wob envs
    height = 160
    width = 160
    env = CropScreen(env, height, width, 0, 0)
    reduced_height = 42
    reduced_width = 42
    env = MiniWOBRescale(env, width=reduced_width, height=reduced_height)
    # limit actions to key locations and clicks
    # pass the original width and height because those are used 
    # to map the discrete actions back to mouse locations in the screen
    action_width = 160
    action_height = 160
    env = DiscreteToMouseCoordVNCActions(
        env, n_xbins=16, n_ybins=16, width=action_width, height=action_height,
            top=0, left=0)
    # env = DiscreteToMouseMovementVNCActions(
    #     env, width=action_width, height=action_height, step_size=15)
    # low = np.array([10., 50. + 75.])
    # high = low + np.array([action_width, action_height])
    # coord_space = gym.spaces.Box(low, high)
    # env = ContinuousToMouseCoordVNCActions(env, coord_space)
    env = Unvectorize(env)
    return env

def create_wob_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Monitor(env, 'videos/', force=True)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    # height and width the same for all mini wob envs
    height = 210
    width = 160
    env = CropScreen(env, height, width, 75, 10)
    reduced_height = 42
    reduced_width = 42
    env = MiniWOBRescale(env, width=reduced_width, height=reduced_height)

    # limit actions to key locations and clicks
    # pass the original width and height because those are used 
    # to map the discrete actions back to mouse locations in the screen
    action_width = 160
    action_height = 160
    env = DiscreteToMouseCoordVNCActions(
        env, n_xbins=16, n_ybins=16, width=action_width, height=action_height)
    # env = DiscreteToMouseMovementVNCActions(
    #     env, width=action_width, height=action_height, step_size=15)
    # low = np.array([10., 50. + 75.])
    # high = low + np.array([action_width, action_height])
    # coord_space = gym.spaces.Box(low, high)
    # env = ContinuousToMouseCoordVNCActions(env, coord_space)
    logger.info('creating MINI WOB env: {}\n'.format(env_id))

    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    logger.info('configuring MINI WOB env')
    env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
        vnc_driver='go', vnc_kwargs={
            'encoding': 'tight', 'compress_level': 0,
            'fine_quality_level': 50, 'subsample_level': 3})
    logger.info('configuration finished for MINI WOB env')
    return env

def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    if env_id == 'flashgames.NeonRace-v0':
        # Better key space for this game.
        keys = ['left', 'right', 'up', 'left up', 'right up', 'down', 'up x']
    logger.info('create_flash_env(%s): keys=%s', env_id, keys)

    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                  vnc_driver='go', vnc_kwargs={
                    'encoding': 'tight', 'compress_level': 0,
                    'fine_quality_level': 50, 'subsample_level': 3})
    return env    

def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env

def create_atari_env(env_id):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log

def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]

class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n

def box2box(a, src, dest):
    # subtract min, divide by range
    out = (a - src.low) / (src.high - src.low)
    # multiply by range, add min
    out = out * (dest.high - dest.low) + dest.low
    out = np.clip(out, dest.low, dest.high)
    return out

class ContinuousToMouseCoordVNCActions(vectorized.ActionWrapper):
    def __init__(self, env, coord_space):
        super(ContinuousToMouseCoordVNCActions, self).__init__(env)
        self.coord_space = coord_space
        # maps from ~gaussian to space
        self.action_space = spaces.Box(
            low=np.array([-3.,-3.]),
            high=np.array([3.,3.]))
    
    def _action(self, action_n):
        actions = []
        for action in action_n:
            coords = box2box(action, self.action_space, self.coord_space).astype(int)
            actions.append([vnc_spaces.PointerEvent(coords[0], coords[1], 0)])
        return actions

class DiscreteToMouseMovementVNCActions(vectorized.ActionWrapper):
    def __init__(self, env, width, height, step_size=15):
        super(DiscreteToMouseMovementVNCActions, self).__init__(env)
        # x, y
        self.coords = np.array([int(width/2), int(height/2)])
        self.width = width
        self.height = height
        self.action_map = {
            0: np.array([step_size, 0]), # move right 
            1: np.array([0, step_size]), # move down
            2: np.array([-step_size, 0]), # move left
            3: np.array([0, -step_size]) # move up
        }
        self.action_space = spaces.Discrete(len(self.action_map.keys()))

    def _action(self, action_n):
        actions = []
        for action_idx in action_n:
            new_coords = self.coords + self.action_map[action_idx]
            new_coords[0] = int(min(max(new_coords[0], 0), self.width))
            new_coords[1] = int(min(max(new_coords[1], 0), self.height))
            self.coords = new_coords # update coordinates
            actions.append([universe.spaces.PointerEvent(
                new_coords[0] + 15, 
                new_coords[1] + 75 + 50 + 10, 
                0)]) # build action
        return actions

# class DiscreteToMouseMovementVNCActions(vectorized.ActionWrapper):
#     def __init__(self, env, width, height, max_step_size=15, n_bins=10):
#         super(DiscreteToMouseMovementVNCActions, self).__init__(env)
#         # x, y
#         self.coords = np.array([int(width/2), int(height/2)])
#         self.width = width
#         self.height = height
#         self.action_map = {
#             0: np.array([step_size, 0]), # move right 
#             1: np.array([0, step_size]), # move down
#             2: np.array([-step_size, 0]), # move left
#             3: np.array([0, -step_size]) # move up
#         }
#         self.action_space = spaces.MultiDiscrete()

#     def _action(self, action_n):
#         actions = []
#         for action_idx in action_n:
#             new_coords = self.coords + self.action_map[action_idx]
#             new_coords[0] = int(min(max(new_coords[0], 0), self.width))
#             new_coords[1] = int(min(max(new_coords[1], 0), self.height))
#             self.coords = new_coords # update coordinates
#             actions.append([universe.spaces.PointerEvent(
#                 new_coords[0] + 15, 
#                 new_coords[1] + 75 + 50 + 10, 
#                 0)]) # build action
#         return actions

class DiscreteToMouseCoordVNCActions(vectorized.ActionWrapper):
    def __init__(self, env, n_xbins=16, n_ybins=16, width=155, height=155,
            top=125, left=10):
        super(DiscreteToMouseCoordVNCActions, self).__init__(env)
        self._n_x_bins = n_xbins
        self._n_y_bins = n_ybins
        self._top = top
        self._left = left
        self._width = width
        self._height = height
        self._generate_actions()
        self.action_space = spaces.Discrete(n_xbins * n_ybins)
        logger.info('Discrete Mouse Coord Actions: {}\n'.format(self._actions))

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        acts = [self._actions[int(action)] for action in action_n]
        return acts

    def _generate_actions(self):
        self._actions = []
        # add offset to place in middle of bin
        x_offset = 0 #(self._width / self._n_x_bins) / 2.
        y_offset = 0 #(self._height / self._n_y_bins) / 2.
        for xcoord in np.linspace(0, self._width, self._n_x_bins):
            for ycoord in np.linspace(0, self._height, self._n_y_bins):
                # add 10 to xcoord to move into mini wob region
                xcoord_val = int(xcoord + self._left + x_offset)
                # add 75 to ycoord to move into mini wob region
                # add 50 to ycoord to move out of text region
                ycoord_val = int(ycoord + self._top + y_offset)
                # set action as mouse movement to coordinates
                # mouse not pressed down
                action = universe.spaces.PointerEvent(xcoord_val, ycoord_val, 0)
                self._actions.append([action])

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys

    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]

    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]

class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                for ob in observation_n]

def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame

class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [128, 200, 1])

    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]

def _process_frame_mini_wob(frame, width, height):
    frame = cv2.resize(frame, (height*2, width*2))
    frame = cv2.resize(frame, (height, width))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [width, height, 1])
    return frame

class MiniWOBRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None, width=160, height=210):
        super(MiniWOBRescale, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(0.0, 1.0, [width, height, 1])

    def _observation(self, observation_n):
        return [_process_frame_mini_wob(observation, self.width, self.height) 
            for observation in observation_n]

