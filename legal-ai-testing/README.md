# legal-ai-testing
## Table of Contents
1. [Setup](#Setup)
    - [Creating virtual environment](#Creating-virtual-environment)
    - [Installing the package](#Installing-the-package)
    - [OpenAI requirements](#OpenAI-requirements)
2. [Reproducing the results](#Reproducing-the-results)
## Setup
### Creating virtual environment
This project was developed using Python 3.8. It is recommended to install this repository in a virtual environment.
Make sure you have [Python 3.8](https://www.python.org/downloads/release/python-380/) installed on your machine. Then, initialize your virtual environment in this folder, for example via the command 
```bash
python3.8 -m venv .venv
```
You can activate the virtual environment via the command
```bash
source .venv/bin/activate
```
### Installing the package
The package can be installed via the command
```bash
pip install -e .
```

### OpenAI requirements
If you want to run the `experiments/create_paraphrased_facts.py` script, you need access to the OpenAI API. The code expects two environment variables to be set: `OPENAI_API_KEY` and `OPENAI_ORGANIZATION_ID`.


## Reproducing the results
### Running the experiments
If you want to simply test the performance of the trained model on the ECHR testset, run the following command.
```bash
python experiments/test_model_performance.py
```

In order to create paraphrases of legal facts stored in the ECHR format, run the following command.

```bash
python experiments/create_paraphrased_facts.py --mode create_paraphrases --subset_type logit_equal_distribution --subset_size 494 --num_attacks_per_case 10 --attack_most_important_fact
```

The paraphrased legal cases can be classified using the following command. The `file_path` argument should point to the file created in the previous step.
```bash
python experiments/classify_adversarial_paraphrases.py --file_path data/subsets/ECHR19_subset_logit_equal_distribution_size_494_with_attention_weights_extended_11_attacks_per_case.json
```

Finally, the plots can be created using the following command. The `result_path` argument should point to the file created in the previous step.
```bash
python experiments/plot_summary_plots.py --data_type paraphrasing_facts --experiment_type summary_plot_sensitivity --num_facts_per_case all --num_paraphrases_per_fact all --result_path data/subsets/ECHR19_subset_logit_equal_distribution_size_494_with_attention_weights_extended_11_attacks_per_case_results.json --save_plot
```