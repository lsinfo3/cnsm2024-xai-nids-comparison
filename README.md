# Agree to Disagree: Exploring Consensus of XAI Methods for ML-based NIDS

## <dataset_name> folders

Firstly, these dataset folders contain *binary*/*multi* subfolders, since we only focused on binary analyses it only contains the *binary* folders here.

Secondly, these subfolders are then subdivided into the different feature selection strategies. For all of them, these folders are further subdivided into the number of chosen features; here: only 10. These contain the list of chosen features for each strategy. Since we mainly focused on the impurity-based approach, the *impurity* folder contains also the trained models, and evaluation metrics.

Note that when executing the *train_and_save_models.py* script, these folders will contain more data, i.e., the dataset splits and preprocessed features and labels (more infos below). We did not include them here because we did not want to simply reupload the datasets from the original authors. You can find them here:

CICIDS: https://www.unb.ca/cic/datasets/ids-2017.html

CIDDS: https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/

EdgeIIoT: https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications

## data

This folder contains the data needed to reproduce the plots and tables in the paper.

all_matrices_<dataset_name>.pkl contains the data for the three consensus analyses in the paper. So each file contains a 18x18x1000 matrix for each of the three analyses.

The rest of the the files are .csv that either contain the info for the local explanation for a specific sample, or the table for the global analysis.

## train_and_save_models.py

This script loads the data (see links above), saves the splits and preprocesses the data, and ultimately, trains various ML models. As mentioned above, this script will generate some additional data compared to what is contained in the repository. In detail, it will save into each <dataset_name> folder the raw data splits, and for each for the selection methods the preprocessed data.

## misc_helpers.py

This file contains some helper functions for the main script.

## explanation_comparison.py

This script loads the trained models and then executes the three consensus analyses and saves the data like shown in the *data* folder. Note that the explanation generation and the consensus analyses here happen more in an ad-hoc fashion, i.e., after generating the explanations the consensus is directly computed, without first saving *all* explanations in a more modular fashion. If we continue with our research in the future, we might make it more modular to make it easier to add more consensus analyses.

## plot_heatmaps.py

Simply takes the data from the *data* folder and plots the heatmaps shown in the paper.
