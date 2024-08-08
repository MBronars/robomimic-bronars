import os
import json
import matplotlib.pyplot as plt
import robomimic

# Given JSON data
STAT_JSON_PATH = os.path.join(robomimic.__path__[0], "../safediffusion_d4rl/pointmaze/exps/diffusion_policy_guiding/single/stats.json")

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_figures(fig, file_path):
    fig.savefig(file_path)

if __name__ == "__main__":
    json_file_path = STAT_JSON_PATH

    data = load_json(json_file_path)

    dirname = os.path.dirname(json_file_path)

    seeds = [18, 19, 23, 24, 25, 26, 41, 45, 46, 47, 48, 49]

    data = {k:entry for k, entry in data.items() if int(k) in seeds}


    # Extracting horizons and interventions from the JSON data
    horizons = [entry['Horizon'] for k, entry in data.items()]
    interventions = [entry['Intervention']/entry["Horizon"] for k, entry in data.items()]

    # Calculating the success rate
    total_entries = len(data)
    success_count = sum(entry['Success'] for k, entry in data.items())
    success_rate = success_count / total_entries

    # Plotting histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for horizons
    axes[0].hist(horizons, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title('Histogram of Rollout Horizons')
    axes[0].set_xlabel('Horizon')
    axes[0].set_ylabel('Frequency')
    axes[0].set_ylim([0, 5])

    # Histogram for interventions
    axes[1].hist(interventions, bins=20, color='salmon', edgecolor='black')
    axes[1].set_title('Histogram of Interventions')
    axes[1].set_xlabel('Intervention')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 5])

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Save the figures
    save_figures(fig, os.path.join(dirname, 'histograms.png'))  # Replace with your desired file path

    # Print success rate
    print(f'Success Rate: {success_rate * 100:.2f}%')
