# Exploring the Motivations behind Behavior: Cyberviolence Detection 

# Dataset
The experimental datasets where can be seen in dataset folder. Note that you can download the datasets only after an "Application to Use the Datasets for Cyberviolence Detection" has been submitted.

# Code

## Key Requirements

> python==3.8.10  
> torch==1.8.1+cu111  
> transformers==4.44.2 

# Preparation
## Step 1: Obtain the representations of posts/comments and linguistic context items (interaction information)

> cd M2EDCI/model/utils

Configure the dataset.

Of course, you can also prepare the BiLSTM_Graph model by your custom dataset.

## Step 2: Obtain the representations of nonlinguistic context items (user information)

```bash
python user_info_embedding.py
```

## Step 3: Construct the linguistic & nonlinguistic environment

```bash
python BiLSTM_Graph.py >> ./log/1.log
tail -f ./log/1.log
```
## Step 4: Prepare for the specific detectors

This step is for the preparation of the specific detectors. There are six base models in our paper, and the detailed can be seen in our paper.


# Training and Inferring

> cd M2EDCI
Configure the dataset and the parameters of the model

```bash
python xxx.main >> ./log/1.log
tail -f ./log/1.log
```
After that, the results and classification reports will be saved in saved_model/ and log/.

