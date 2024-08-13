import os.path

import gym
from agents.pdqn import P_DQN
from utilities.memory import ReplayBuffer
from utilities.utilities import *
from logger import logger
import time
import math

class Train_and_Evaluate(object):

    def __init__(self, config):
        # Environment
        self.env = config.environment

        # Agent
        self.agent = P_DQN(config, self.env)

        # Memory
        self.replay_memory_size = config.hyperparameters['replay_memory_size']
        self.batch_size = config.hyperparameters['batch_size']
        self.updates_per_step = config.hyperparameters['updates_per_step']
        self.memory = ReplayBuffer(self.replay_memory_size)

        self.total_steps = 0
        self.total_updates = 0

        self.save_freq = config.save_freq
        self.file_to_save = config.file_to_save
        self.maximum_episodes = config.hyperparameters['maximum_episodes']

        self.train = config.train

        self.agent_to_color_dictionary = config.agent_to_color_dictionary
        self.standard_deviation_results = config.standard_deviation_results

        self.colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        self.color_idx = 0

        self.rolling_score_window = config.rolling_score_window
        self.runs_per_agent = config.runs_per_agent
        self.agent_name = config.agent_name
        self.ceil = config.ceil
        
        self.existing_actor = config.existing_actor
        self.existing_actor_param = config.existing_actor_param
        # Training Loop
        
    def scale_action_param(self, action):
        
        if action[0]==0: # THINNING
            act_param = action[1]/10 * 0.5
            action[1] = act_param
        if action[0]==0: # RAISING
            act_param = int(action[1])
            action[1] = act_param
        elif action[0]==1: # REDUCTION_EAST
            act_param = int(action[1])
            action[1] = act_param
        elif action[0]==2: # REDUCTION_SOUTH
            act_param = int(action[1])
            action[1] = act_param
        elif action[0]==3: # REDUCTION_WEST
            act_param = int(action[1])
            action[1] = act_param
        elif action[0]==4: # REDUCTION_NORTH
            act_param = int(action[1])
            action[1] = act_param
        elif action[0]==5: # REDUCTION_TOP
            act_param = int(action[1])
            action[1] = act_param
        elif action[0]==6: # Topping
            act_param = action[1]/10 * 5
            action[1] = act_param
        elif action[0]==7: # NOACTION
            act_param = 0.0
            action[1] = act_param
        else:               # ENDING
            act_param = 0.0
            action[1] = act_param
            
        return action

    def train_agent(self, train_existing=False):
        """

        :return:
        """

        rolling_scores_for_diff_runs = []
        mean_rolling_scores_for_diff_runs = []
        cummulative_rolling_scores_for_diff_runs = []
        
        file_to_save_actor = os.path.join(self.file_to_save, 'actor/')
        file_to_save_actor_param = os.path.join(self.file_to_save, 'actor_param/')
        file_to_save_runs = os.path.join(self.file_to_save, 'runs_1/')
        file_to_save_rolling_scores = os.path.join(self.file_to_save, 'rolling_scores/')
        os.makedirs(file_to_save_actor, exist_ok=True)
        os.makedirs(file_to_save_actor_param, exist_ok=True)
        os.makedirs(file_to_save_runs, exist_ok=True)
        os.makedirs(file_to_save_rolling_scores, exist_ok=True)
        
        if train_existing:
            actor_path = file_to_save_actor + self.existing_actor
            actor_param_path = file_to_save_actor_param + self.existing_actor_param 
            self.agent.load_models(actor_path, actor_param_path)

        #for run in range(self.runs_per_agent):
        # game_full_episodes_scores = []
        game_full_episodes_rolling_scores = []
        game_full_episodes_mean_rolling_scores = []
        game_full_episodes_cumtlv_rolling_scores = []
        step_scores = []
        
        start_time = time.time()

        for i_episode in range(self.maximum_episodes):
            logger.info(f"******************** Running episode {i_episode} ********************")

            # if self.save_freq > 0 and i_episode % self.save_freq == 0:
            #     actor_path = os.path.join(file_to_save_actor, 'episode{}.pth'.format(i_episode))
            #     actor_param_path = os.path.join(file_to_save_actor_param, 'episode{}.pth '.format(i_episode))
            #     self.agent.save_models(actor_path, actor_param_path)

            episode_score = []
            episode_steps = 0
            done = 0

            self.env.episodes = i_episode
            self.env.episode_steps = 0
            state = self.env.reset()  # n_steps  
            #print("state shape: ",state.shape )
            
            if np.isnan(state).any():
                pass
            
            while not done:
                start = time.time()
                if len(self.memory) > self.batch_size:
                    action, action_params = self.agent.select_action(state, self.train)
                    if self.ceil:
                        action_params = np.ceil(action_params).squeeze(0)
                    else:
                        action_params = action_params.squeeze(0)
                    
                    # limit upper action param value
                    action_param = 0.0
                    if math.isnan(action_params[action]):
                        action_param = 10.0
                    else:
                        action_param = action_params[action]
                    
                    action_for_env = [action, action_param]
                    #action_for_env = [action, action_params[action]]
                    #logger.info(f"++ Chosen action (Network): {action_for_env}")
                    for i in range(self.updates_per_step):
                        self.agent.update(self.memory)
                        self.total_updates += 1
                else:
                    action_params = np.random.uniform(low=0, high=10, size=10)
                    action = np.random.randint(10, size=1)[0]
                    action_for_env = [action, action_params[action]]
                    #logger.info(f"++ Chosen action (Random): {action_for_env}")
                    
                    # action 7 must be integer, action 0 range between (0 - 0.05)
                
                self.env.episode_steps = episode_steps+1
                
                action_for_env = self.scale_action_param(action_for_env)
                
                next_state, reward, done, info = self.env.step(action_for_env)
                # print("valid next state: ",np.isnan(next_state).any())
                end = time.time()
                length = end - start
                logger.info(f"Step: {self.env.episode_steps}, Step reward: {reward:.4f}, time taken: {length:.4f} seconds")

                episode_steps += 1

                episode_score.append(reward) # record last round reward
                step_scores.append(reward)
                
                if not np.isnan(next_state).any():
                    self.total_steps += 1
                    self.memory.push(state, action, action_params, reward, next_state, done)

                    state = next_state
            
            # Record mean reward
            # episode_score_so_far = np.mean(episode_score) # use cummulative reward, not mean
            # game_full_episodes_scores.append(episode_score_so_far)
            # game_full_episodes_rolling_scores.append(
            #     np.mean(game_full_episodes_scores))
            
            # Record reward of final round
            #episode_score_so_far = episode_score[-1] # episode_score_so_far is final reward of round
            # game_full_episodes_rolling_scores.append(episode_score_so_far)
            # game_full_episodes_rolling_scores.append(
            #     np.mean(game_full_episodes_scores[self.rolling_score_window:]))
            
            #if len(self.memory) > self.batch_size:
            game_full_episodes_rolling_scores.append(episode_score[-1])
            game_full_episodes_mean_rolling_scores.append(np.mean(episode_score))
            game_full_episodes_cumtlv_rolling_scores.append(np.sum(episode_score))

            logger.info("## (Episode summary): Episode: {}, total steps:{}, episode steps:{}, final score:{}, mean score: {:.4f}, cummulative score: {:.4f}".format(
                i_episode, self.total_steps, episode_steps, episode_score[-1], np.mean(episode_score), np.sum(episode_score)))

            self.env.close()
            #file_path_for_pic = os.path.join(file_to_save_runs, 'episode{}_run{}.jpg'.format(i_episode, run))
            # file_path_for_pic = os.path.join(file_to_save_runs, 'episode{}.jpg'.format(i_episode))
            # visualize_results_per_run(agent_results=game_full_episodes_scores,
            #                           agent_name=self.agent_name,
            #                           save_freq=1,
            #                           file_path_for_pic=file_path_for_pic)
            rolling_scores_for_diff_runs.append(game_full_episodes_rolling_scores)
            mean_rolling_scores_for_diff_runs.append(game_full_episodes_mean_rolling_scores)
            cummulative_rolling_scores_for_diff_runs.append(game_full_episodes_cumtlv_rolling_scores)
            
        # logger.info(f"game_full_episodes_scores: {game_full_episodes_scores}")
        # logger.info(f"game_full_episodes_rolling_scores: {rolling_scores_for_diff_runs}")
        # logger.info(f"game_full_episodes_mean_rolling_scores: {mean_rolling_scores_for_diff_runs}")
        # logger.info(f"game_full_episodes_cummulative_rolling_scores: {cummulative_rolling_scores_for_diff_runs}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
        
        # Record final episode scores
        logger.info("logging rolling_scores")
        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'final_rolling_scores.jpg')
        visualize_overall_agent_results(rolling_scores_for_diff_runs[0], plot_title= 'Final scores', file_path_for_pic=file_path_for_pic)
        # visualize_overall_agent_results(agent_results=rolling_scores_for_diff_runs,
        #                                 agent_name=self.agent_name,
        #                                 show_mean_and_std_range=False,
        #                                 agent_to_color_dictionary=self.agent_to_color_dictionary,
        #                                 standard_deviation_results=1,
        #                                 file_path_for_pic=file_path_for_pic
        #                                 )
        
        # Record mean episode scores
        logger.info("logging mean_rolling_scores")
        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'mean_rolling_scores.jpg')
        visualize_overall_agent_results(mean_rolling_scores_for_diff_runs[0], plot_title= 'Mean scores', file_path_for_pic=file_path_for_pic)
        # visualize_overall_agent_results(agent_results=mean_rolling_scores_for_diff_runs,
        #                                 agent_name=self.agent_name,
        #                                 show_mean_and_std_range=False,
        #                                 agent_to_color_dictionary=self.agent_to_color_dictionary,
        #                                 standard_deviation_results=1,
        #                                 file_path_for_pic=file_path_for_pic
        #                                 )
        
        # Record cummulative episode scores
        logger.info("logging cummulative_rolling_scores")
        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'cummulative_rolling_scores.jpg')
        visualize_overall_agent_results(cummulative_rolling_scores_for_diff_runs[0], plot_title= 'Cummulative scores', file_path_for_pic=file_path_for_pic)
        # visualize_overall_agent_results(agent_results=cummulative_rolling_scores_for_diff_runs,
        #                                 agent_name=self.agent_name,
        #                                 show_mean_and_std_range=False,
        #                                 agent_to_color_dictionary=self.agent_to_color_dictionary,
        #                                 standard_deviation_results=1,
        #                                 file_path_for_pic=file_path_for_pic
        #                                 )
        logger.info("logging step_scores")
        file_path_for_pic = os.path.join(file_to_save_rolling_scores, 'step_scores.jpg')
        visualize_overall_agent_results(step_scores, plot_title= 'Step scores', file_path_for_pic=file_path_for_pic,step_score=True)
