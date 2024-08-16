import torch
from src import config
from src import trainer
from src import prunEnv_V2_binary

config = config.Config()
config.seed = 42 #123456
config.train = True
config.environment = prunEnv_V2_binary.PrunEnvWrapper() #TreeEnv()
config.file_to_save = 'results/'
config.standard_deviation_results = 1.0 # for visualization
config.save_freq = 5
config.rolling_score_window = 5
config.runs_per_agent = 1
config.agent_name = 'P-DQN'
config.use_GPU = True
config.ceil = True
config.existing_actor = 'episode40'
config.existing_actor_param = 'episode40'

config.hyperparameters = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epsilon_initial': 0.3, #0.3,
    'epsilon_final': 0,
    'epsilon_decay': 3000, #3000,
    'replay_memory_size': 10000, #1e5, see if this can help save memory
    'batch_size': 256, # 128 
    'gamma': 0.98, # higher more exploitation
    'lr_critic': 1e-3, #1e-4,
    'lr_actor': 1e-4, # 1e-6,
    'lr_alpha': 1e-4, # 1e-4,
    'tau_critic': 0.01,
    'tau_actor': 0.01,
    'critic_hidden_layers': (256, 128, 64, 32), #(256, 128, 64),
    'actor_hidden_layers': (256, 128, 64, 32), #(256, 128, 64),
    'updates_per_step': 1,
    'maximum_episodes': 5, # temp
    'alpha': 0.2, # lower more exploitation
}

if __name__ == "__main__":
    print("Running")
    trainer = trainer.Train_and_Evaluate(config=config)
    trainer.train_agent(train_existing=False)
