# @time 2024/04/08
import torch.cuda
import torch.nn.functional as F
from env import TreeEnv
from prunEnv import PrunEnvWrapper

class Config(object):
    """ Object to hold the config requirements for an agent. """

    def __init__(self):
        self.seed = None
        self.train = True
        self.environment = PrunEnvWrapper() #TreeEnv()
        self.file_to_save = None
        self.hyperparameters = None
        self.env_parameters = None
        self.standard_deviation_results = 1.0
        self.save_freq = 5 # save model every n episods
        self.rolling_score_window = 5
        self.runs_per_agent = 1
        self.use_GPU = True
        self.agent_name = 'P-DQN'
        self.ceil = True

        self.hyperparameters = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epsilon_initial': 0.3,
            'epsilon_final': 0.01,
            'epsilon_decay': 5000,
            'replay_memory_size': 1e6,
            'batch_size': 64,
            'gamma': 0.99,
            'lr_critic': 1e-5,
            'lr_actor': 1e-4,
            'lr_alpha': 1e-2,
            'tau_actor': 0.01,
            'tau_critic': 0.01,
            'critic_hidden_layers': (256, 128, 64),
            'actor_hidden_layers': (256, 128, 64),
            'random_pick_steps': 10000,
            'updates_per_step': 2,
            'maximum_episodes': 2000,
            'alpha': 0.2,
        }

        self.agent_to_color_dictionary = {
            'P-DQN': '#0000FF',
            'intelligent_light': '#0E0E0F',
        }
