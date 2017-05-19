
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import universe
from universe.spaces.vnc_event import PointerEvent

logger = logging.getLogger(__name__)

class DebugChaseCircleEnv(universe.envs.VNCEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }
    def __init__(self, 
            screen_width=160, 
            screen_height=160,
            horizon=200,
            radius=30,
            velscale=5,
            mouse_s=10,
            info_height=0,
            scaled_reward=False,
            centered=False
            ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.info_height = info_height
        self.horizon = horizon
        self.radius = radius
        self.mouse_s = mouse_s
        self.velscale = velscale
        self.scaled_reward = scaled_reward
        self.centered = centered
        
        self.action_space = universe.spaces.VNCActionSpace()
        self.observation_space = gym.spaces.Box(low=0, high=255, 
            shape=(screen_height, screen_width, 3))

        self.viewer = None
        self._state = ((0, 0), self.horizon)
        self._mouse = (0, 0)

    def _in_circle(self, ax, ay, x, y):
        # euclidean distance less than radius
        return np.sqrt((x - ax) ** 2 + (y - ay) ** 2) < self.radius

    def _scaled_reward(self, ax, ay, x, y):
        # quadratic scaling from center of circle
        dist = np.sqrt((x - ax) ** 2 + (y - ay) ** 2)
        max_dist = np.sqrt(self.screen_height ** 2 + self.screen_width ** 2)
        return 1. / self.horizon * (1 - dist / max_dist) ** 2

    def _propagate(self, x, y, xdot, ydot):
        # check for contacting wall and reverse if so
        if (x + self.radius > self.screen_width 
            or x - self.radius < 0):
            xdot *= -1
        if (y + self.radius > self.screen_height - self.info_height
            or y - self.radius < 0):
            ydot *= -1

        x += xdot
        y += ydot
        return x, y, xdot, ydot

    def _get_obs(self):
        return self.render(mode='rgb_array')

    def _step(self, action):
        x, y, xdot, ydot, t = self.state
        t -= 1

        # check for end of episode
        if t <= 0:
            return self._get_obs(), 0, True, {}
        done = False

        # only use first action provided
        if isinstance(action, list):
            action = action[0]
        # map action from
        ax, ay = action.x, action.y
        ay -= self.info_height

        # reward for action inside circle or nearby circle
        reward = 0
        if self.scaled_reward:
            reward = self._scaled_reward(ax, ay, x, y)
        else:
            if self._in_circle(ax, ay, x, y):
                reward = 1. / self.horizon
        # set this value for rendering of the mouse
        self._mouse = (ax, ay)

        # propagate circle
        x, y, xdot, ydot = self._propagate(x, y, xdot, ydot)
        self.state = (x, y, xdot, ydot, t)
        return self._get_obs(), reward, done, {'reward':reward}

    def _reset(self):
        if self.centered:
            x = self.screen_width / 2.
            y = self.screen_height / 2.
        else:
            x = np.random.randint(0 + self.radius, 
                self.screen_width - self.radius)
            y = np.random.randint(0 + self.radius, 
                self.screen_height - self.radius - self.info_height)
        xdot = (np.random.rand() - .5) * self.velscale
        ydot = (np.random.rand() - .5) * self.velscale
        self.state = (x, y, xdot, ydot, self.horizon)
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            # mouse
            s = self.mouse_s
            mouse = rendering.make_polygon(
                [(0,s/2.),(s/2.,0),(s, s)], filled=False)
            self.mouse_trans = rendering.Transform()
            mouse.add_attr(self.mouse_trans)
            self.viewer.add_geom(mouse)
            # circle
            circle = rendering.make_circle(radius=self.radius, filled=True)
            self.circle_trans = rendering.Transform()
            circle.add_attr(self.circle_trans)
            self.viewer.add_geom(circle)
            # info section 
            info_section = self.viewer.draw_polygon([
                (0,self.screen_height), 
                (self.screen_width,self.screen_height),
                (self.screen_width,self.screen_height - self.info_height),
                (0,self.screen_height - self.info_height),
            ])
            info_section.set_color(.8,.9,1.)
            self.viewer.add_geom(info_section)

        if self.state is None: return None

        x, y, _, _, _ = self.state
        self.circle_trans.set_translation(x, y)
        x, y = self._mouse
        self.mouse_trans.set_translation(x, y)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def configure(self, **_):
        # overwrite base class configure and do nothing so that we can
        # treat this environment as a wob.m
        pass

if __name__ == '__main__':
    env = DebugCircleChaseEnv()   
    x = env.reset()
    totalr = 0
    t = 0 
    while True:
        t += 1
        print('avg reward: {}'.format(totalr / t))
        a = PointerEvent(np.random.randint(0,160), np.random.randint(0,160))
        # a = PointerEvent(env.state[0], env.state[1])
        x, r, done, info = env.step(a)
        totalr += r
        if done:
            env.reset()
        env.render()
