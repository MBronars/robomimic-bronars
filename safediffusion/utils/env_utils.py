import os
import gym

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

#-----------------------------------------------------------------------------#
#------------------------------ d4rl -----------------------------------------#
#-----------------------------------------------------------------------------#
@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#--------------------------------- load_env ----------------------------------#
#-----------------------------------------------------------------------------#
def load_environment(name):
    if type(name) != str:
        return name
    with suppress_output():
        wrapped_env = gym.make(name)

    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env