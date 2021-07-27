# Replicated: Visualization

## Preparation
Download the VisBERT models and save them to a folder, e.g., `./models`

Create a sample containing a question, the context and the answer. Save the sample to a file, e.g, `./samples/sample_squad.json`

## Usage
Run: `visualization.py`

For details about needed and accepted command line arguments, see below.

Example: `> python visualization.py -sample ./samples/sample_squad.json -model ./models/squad.bin`

Accepted command line arguments:
- `-sample`: Path to the sample file
- `-model`: Path to the model file
- `-type`: Type of the model (optional, default: bert-base-uncased)
- `-title`: Title of the experiment (optional)
- `-format`: File format for the output (png,pdf,svg,jpg) (optional)
- `--no-legend`: If set, no legend will be added to the plots

The results will be saved to a subfolder of `./outputs`, named like the experiment's title (set with `-title` option). If the title of the experiment is not set, the current date and time will be used instead.