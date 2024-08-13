# @author Metro
# @time 2021/11/03

import torch
import matplotlib.pyplot as plt
import numpy as np


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# -------- Plot --------
def visualize_overall_agent_results(scores, plot_title="Cummulative Episode Score", file_path_for_pic=None, step_score=False):
    # Example cumulative episode scores

    # Episode numbers
    episodes = np.arange(1, len(scores) + 1)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(episodes, scores, linestyle='-', color='b')
    plt.title(plot_title)
    if step_score:
        plt.xlabel('Steps)')
    else:
        plt.xlabel('Episode')
    plt.ylabel('Scores')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path_for_pic, bbox_inches='tight')

def visualize_overall_agent_results2(agent_results, agent_name, show_mean_and_std_range=True,
                                    agent_to_color_dictionary=None, standard_deviation_results=1,
                                    file_path_for_pic=None):
    """
    Visualize the results for one agent.

    :param file_path_for_pic:
    :param title:
    :param standard_deviation_results:
    :param agent_to_color_dictionary:
    :param agent_results: list of lists, each
    :param agent_name:
    :param show_mean_and_std_range:
    :return:
    """
    assert isinstance(agent_results, list), 'agent_results must be a list of lists.'
    assert isinstance(agent_results[0], list), 'agent_result must be a list of lists.'
    fig, ax = plt.subplots()
    color = agent_to_color_dictionary[agent_name]
    if show_mean_and_std_range:
        mean_minus_x_std, mean_results, mean_plus_x_std = get_mean_and_standard_deviation_difference(
            agent_results, standard_deviation_results)
        x_vals = list(range(len(mean_results)))
        ax.plot(x_vals, mean_results, label=agent_name, color=color)
        ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)  # TODO
        ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
        ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)
    else:
        color_idx = 0
        colors = ['red', 'blue', 'green', 'orange', 'yellow', 'purple']
        for ix, result in enumerate(agent_results):
            x_vals = list(range(len(agent_results[0])))
            ax.plot(x_vals, result, label=agent_name + '_{}'.format(ix + 1), color=color)
            color, color_idx = get_next_color(colors, color_idx)

    ax.set_facecolor('xkcd:white')
    # ax.legend(loc='upper right', shadow='Ture', facecolor='inherit')
    ax.set_title(label='Training', fontsize=15, fontweight='bold')
    ax.set_ylabel('Rolling Episode Scores')
    ax.set_xlabel('Episode Number')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.set_xlim([0, x_vals[-1]])

    y_limits = get_y_limits(agent_results)
    ax.set_ylim(y_limits)

    plt.tight_layout()
    plt.savefig(file_path_for_pic)


def get_mean_and_standard_deviation_difference(results, standard_deviation_results):
    """
    From a list of lists of specific agent results it extracts the mean result and the mean result plus or minus
    some multiple of standard deviation.

    :param standard_deviation_results:
    :param results:
    :return:
    """

    def get_results_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return results_at_a_time_step

    def get_std_at_a_time_step(results_, timestep):
        results_at_a_time_step = [result[timestep] for result in results_]
        return np.std(results_at_a_time_step)

    mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
    mean_minus_x_std = [mean_val - standard_deviation_results * get_std_at_a_time_step(results, timestep)
                        for timestep, mean_val in enumerate(mean_results)]
    mean_plus_x_std = [mean_val + standard_deviation_results * get_std_at_a_time_step(results, timestep)
                       for timestep, mean_val in enumerate(mean_results)]
    return mean_minus_x_std, mean_results, mean_plus_x_std


def get_next_color(colors=None, color_idx=None):
    """
    Gets the next color in list self.colors. If it gets to the end then it starts from beginning.

    :return:
    """

    color_idx += 1
    if color_idx >= len(colors):
        color_idx = 0

    return colors[color_idx], color_idx


def get_y_limits(results):
    """
    Extracts the minimum and maximum seen y_vals from a set of results.

    :param results:
    :return:
    """
    min_result = float('inf')
    max_result = float('-inf')
    for result in results:
        tem_max = np.max(result)
        tem_min = np.min(result)
        if tem_max > max_result:
            max_result = tem_max
        if tem_min < min_result:
            min_result = tem_min
    y_limits = [min_result, max_result]
    return y_limits


def visualize_results_per_run(agent_results, agent_name, save_freq, file_path_for_pic):
    """

    :param file_path_for_pic:
    :param save_freq:
    :param agent_name:
    :param agent_results:
    :return:
    """
    assert isinstance(agent_results, list), 'agent_results must be a list of lists.'
    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:white')
    ax.legend(loc='upper right', shadow='Ture', facecolor='inherit')
    ax.set_title(label='Episode Scores For One Specific Run', fontsize=15, fontweight='bold')
    ax.set_ylabel('Episode Scores')
    ax.set_xlabel('Episode Number')
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    x_vals = list(range(len(agent_results)))
    ax.set_xlim([0, x_vals[-1]])
    ax.set_ylim([min(agent_results), max(agent_results)])
    ax.plot(x_vals, agent_results, label=agent_name, color='blue')
    plt.tight_layout()

    Runtime = len(agent_results)
    if Runtime % save_freq == 0:
        plt.savefig(file_path_for_pic)


