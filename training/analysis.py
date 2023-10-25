import ray
from ray import tune
from ray.air import Result


def analysis(experiment_path):
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable='PPO')
    result_grid = restored_tuner.get_results()

    # Check if there have been errors
    if result_grid.errors:
        print("One of the trials failed!")
    else:
        print("No errors!")

    # Iterate over results
    for i, result in enumerate(result_grid):
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        print(
            f"Trial #{i} finished successfully with a mean accuracy metric of:",
            result.metrics["episode_reward_mean"]
        )

    # Get the result with the maximum test set `mean_accuracy`
    best_result = result_grid.get_best_result(metric='episode_reward_mean', mode='max')

    print("Best Checkpoint :")
    for b in best_result.best_checkpoints:
        print("Reward Mean :")
        print(b[1]['episode_reward_mean'])
        print("Checkpoint :")
        print(b[0])

    # Get the result with the maximum test set `mean_accuracy`
    best_result: Result = result_grid.get_best_result()

    print(best_result.config)


if __name__ == '__main__':
    ray.init()
    experiment_path = ""  # The path of your experiment
    analysis(experiment_path)

