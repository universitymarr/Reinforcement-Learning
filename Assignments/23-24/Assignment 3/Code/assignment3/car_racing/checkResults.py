import json
import matplotlib.pyplot as plt
import numpy as np

# Initialize an empty list to store all the scores
all_scores = []

# List of JSON file names
json_files = ['results_checkpoint_da_0_a_1470.json', 'results_checkpoint_da_1470_a_3280.json',
               'results_checkpoint_da_3280_a_4400.json', 'results_checkpoint_da_4400_a_5870.json']

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        all_scores.extend(data['scores'])

# Calculate a moving average with a window size of your choice (e.g., 100)
window_size = 100
moving_avg = np.convolve(all_scores, np.ones(window_size)/window_size, mode='valid')

# Plot the original scores and the smoothed curve
plt.figure(figsize=(8,5))
plt.plot(all_scores, label='Original Scores', alpha=0.5)
plt.plot(np.arange(window_size-1, len(all_scores)), moving_avg, label=f'Moving Average (Window={window_size})', color='red')
plt.title('Scores | Training Mean reward: {:.2f}'.format(np.mean(all_scores)))
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend()
plt.show()