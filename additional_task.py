import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import argparse
import logging
import os
import sys

import pandas as pd



def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of predictions and answers should be the same")
        sys.exit(1)

    
    correct = sum(p == a for p, a in zip(predictions, answers))
    accuracy = round((correct / len(answers)) * 100, 2)

    
    true_positives = sum((p == 1 and a == 1) for p, a in zip(predictions, answers))
    false_positives = sum((p == 1 and a == 0) for p, a in zip(predictions, answers))
    false_negatives = sum((p == 0 and a == 1) for p, a in zip(predictions, answers))

    precision = round((true_positives / (true_positives + false_positives)) * 100, 2) if (true_positives + false_positives) > 0 else 0
    recall = round((true_positives / (true_positives + false_negatives)) * 100, 2) if (true_positives + false_negatives) > 0 else 0

    logging.info(f"Accuracy: {accuracy}%")
    logging.info(f"Precision: {precision}%")
    logging.info(f"Recall: {recall}%")



def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])
            tmp[8] = int(tmp[8])
            instances.append(tmp[1:-1])
            labels.append(tmp[-1])
    return instances, labels



def feature_engineering(instances, labels, top_k=2):
    """
    Automatically selects the top_k most informative features using mutual information.
    """
    instances = np.array(instances)
    labels = np.array(labels)
    
    
    mutual_info = mutual_info_classif(instances, labels, discrete_features=False)
    selected_indices = np.argsort(mutual_info)[-top_k:]  
    
    print(f"Selected Feature Indices: {selected_indices}")
    return instances[:, selected_indices], selected_indices

def training_automated(instances, labels):
    """
    Automates parameter calculation using Numpy and feature scaling.
    """
    dataset = {}
    parameters = {}
    
    for i in range(len(instances)):
        label = labels[i]
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(instances[i])

    for label, instances in dataset.items():
        instances = np.array(instances)
        mean = instances.mean(axis=0)
        stdev = instances.std(axis=0)
        parameters[label] = list(zip(mean, stdev))

    return parameters

def predict_automated(instance, parameters):
    """
    Automates probability calculations with Numpy's vectorization.
    """
    probabilities = {}
    for label, params in parameters.items():
        mean_stdev = np.array(params)
        mean, stdev = mean_stdev[:, 0], mean_stdev[:, 1]
        
        
        exponent = np.exp(-((instance - mean) ** 2) / (2 * stdev ** 2))
        probabilities[label] = np.prod((1 / (np.sqrt(2 * np.pi) * stdev)) * exponent)

    return max(probabilities, key=probabilities.get)





def run_automated(train_file, test_file):
    
    instances, labels = load_raw_data(train_file)
    
    
    processed_instances, selected_features = feature_engineering(instances, labels)
    
    
    parameters = training_automated(processed_instances, labels)
    
    
    test_instances, test_labels = load_raw_data(test_file)
    test_instances = np.array(test_instances)[:, selected_features]
    predictions = [predict_automated(instance, parameters) for instance in test_instances]
    
    
    report(predictions, test_labels)


def command_line_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--training", 
        required=True, 
        metavar="<file path to the training dataset>", 
        help="File path of the training dataset", 
        default="training.csv"
    )
    parser.add_argument(
        "-u", "--testing", 
        required=True, 
        metavar="<file path to the testing dataset>", 
        help="File path of the testing dataset", 
        default="testing.csv"
    )
    parser.add_argument(
        "-l", "--log", 
        help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", 
        type=str, 
        default="INFO"
    )
    args = parser.parse_args()
    return args

# Adjusted main
def main_automated():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error(f"Training dataset does not exist: {args.training}")
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error(f"Testing dataset does not exist: {args.testing}")
        sys.exit(1)

    run_automated(args.training, args.testing)

if __name__ == "__main__":
    main_automated()
