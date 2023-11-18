This code is for AnomyFL.

We use FISCO-BCOS 2.7 as our blockchain platform, so before running the code, we recommend that you familiarize yourself with FISCO-BCOS and its corresponding SDK code.
FISCO-BCOS repository: https://github.com/FISCO-BCOS/FISCO-BCOS
FISCO-BCOS python-sdk repository: https://github.com/FISCO-BCOS/python-sdk

The fisco_py in our code is the python-sdk folder for FISCO-BCOS, and besides that, we need to run the FISCO-BCOS blockchain.

The steps to run AnomyFL are as follows:

1. Download the FISCO-BCOS blockchain, configure it according to requirements, and run it.
2. Deploy smart contracts using the FISCO-BCOS console and save the contract address.
3. Install Python environment by using "pip install -r requirement.txt" and "pip install -r fisco_py/requirement.txt".
4. Our framework's configuration file is located in the config folder. We put the address after "address=" in each .ini file.
5. Modify other information in the configuration file if necessary

The key explanations for the keywords that may need to be modified in our configuration file are as follows:

1. dataset: Optional values are cifa10, cifa100, mnist, emnist.
2. base_path: Fill in the current working directory.
3. address: The address of the aggregated smart contract.
4. gpu: PyTorch's cuda option.
5. net: Which network to use - VGG for cifa10, resnet for cifa100, CNN for mnist and emnist.
6. dataset_distribution: Indicates whether the data is non-independent and identically distributed (IID).
7. exp_name, project_name: Corresponding contents of wandb (Weights & Biases), see main.py for details.
8. epoch: Number of local training rounds on each client.
9. num_clients: Total number of clients participating in training.
10. num_comm: Communication rounds.
11. mean_batch: Hyperparameter for E-AnomyFL and FedMix algorithms.
12. dataset_class_per_client：Number of classes per client's dataset
    13.frac：Proportion of clients sampled in each communication round
    14.lamb2：Ratio used to mix synthetic datasets
    15.decay_lamb2：Decay ratio for mixing synthetic datasets
    16.dead_lamb2：After how many rounds should synthetic datasets no longer participate in mixing
    17.dataset_balance：Whether the number of datasets per client is balanced or not
    18.iid：Whether it is independent and identically distributed (IID)

To run our code, you only need to do:
python main.py --config_path config/AnomyFL.ini

