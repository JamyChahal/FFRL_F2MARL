import ray
from ray import tune
from ray.air import Result
from ray.tune import ExperimentAnalysis


def analysis(experiment_path):
    print(f"Loading results from {experiment_path}")

    analysis = ExperimentAnalysis(experiment_path)
    result = analysis.get_best_checkpoint(analysis.trials[0], "episode_reward_mean", "max")
    print(result)



if __name__ == '__main__':
    ray.init()
    experiment_path = ""  # The path of your experiment

    analysis(experiment_path)

