import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
import gym
import einops

from safediffusion.utils.env_utils import load_environment

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#
def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(img_as_string, dtype='uint8').reshape((height, width, 4))


#-----------------------------------------------------------------------------#
#------------------- maze 2d custom renderers---------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    'maze2d-umaze-dense-v1': (0, 5, 0, 5),
    'maze2d-medium-dense-v1': (0, 8, 0, 8),
    'maze2d-large-dense-v1': (0, 9, 0, 12)
}

class MazeRenderer:

    def __init__(self, env):
        if type(env) is str: env = load_environment(env)
        self._config = env._config
        self._background = self._config != ' '
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, plans=None, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5.12, 5.12)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
        if plans is not None:
            plt.plot(plans[1, :], plans[0, :], c = 'blue', zorder = 30)
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

class Maze2dRenderer(MazeRenderer):

    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
        
        self.observation_history = []
    
    def render(self, observation, plan=None, title=None):
        """
        Single observation, append it to the histories, and render the rollouts
        """
        self.observation_history.append(observation)

        return self.renders(np.array(self.observation_history), plan, title=title)

    def renders(self, observations, plans=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        observations = observations + .5
        if plans is not None:
            plans = plans + .5

        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
            if plans is not None:
                plans /= scale

        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale

            if plans is not None:
                plans[0, :] /= iscale
                plans[1, :] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

        return super().renders(observations, plans, **kwargs)
    
def plot_history_maze(stat, save_dir):
    """
    Plot core plots from the stat: 1) control input, 2) tracking-error, 3) velocity
    """
    # Plot "stamps-actions"
    if "actions" in stat.keys():
        plt.figure()
        for i in range(stat["actions"].shape[1]):
            plt.plot(stat["stamps"], stat["actions"][:, i], label=f'Action {i+1}')
        plt.xlabel('Stamps'); plt.ylabel('Actions')
        plt.title('Stamps vs Actions')
        plt.grid(True); plt.legend()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'actions.png'))
        plt.close()

    # Plot "stamps-tracking_errors"
    if "tracking_errors" in stat.keys():
        plt.figure()
        for i in range(stat["tracking_errors"].shape[1]):
            plt.plot(stat["stamps"], stat["tracking_errors"][:, i], label=f'Tracking Error {i+1}')
        plt.xlabel('Stamps'); plt.ylabel('Tracking Errors')
        plt.title('Stamps vs Tracking Errors')
        plt.grid(True); plt.legend()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'tracking_errors.png'))
        plt.close()

    # Plot "velocity"
    if "rollout" in stat.keys():
        plt.figure()
        plt.plot(stat["stamps"], stat["rollout"][:-1, 0], label=f'p_x')
        plt.plot(stat["stamps"], stat["rollout"][:-1, 1], label=f'p_y')
        plt.plot(stat["stamps"], stat["rollout"][:-1, 2], label=f'v_x')
        plt.plot(stat["stamps"], stat["rollout"][:-1, 3], label=f'v_y')
        plt.xlabel('Stamps'); plt.ylabel('Velocity')
        plt.title('Stamps vs Velocity')
        plt.grid(True); plt.legend()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'velocity.png'))
        plt.close()