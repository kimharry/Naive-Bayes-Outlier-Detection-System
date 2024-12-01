import os
import sys
import argparse
import logging
from math import exp, pi, sqrt

def training(instances, labels):
    dataset = {}
    for i in range(len(instances)):
        label = labels[i]
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(instances[i])

    logging.debug("dataset: {}".format(dataset))

    parameters = {}
    for label, instances in dataset.items():
        parameters[label] = []
        for i in range(len(instances[0])):
            mean = sum([instance[i] for instance in instances]) / len(instances)
            stdev = sqrt(sum([(instance[i] - mean) ** 2 for instance in instances]) / len(instances))
            parameters[label].append((mean, stdev, len(instances)))

    logging.debug("parameters: {}".format(parameters))
    
    return parameters

def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def predict(instance, parameters):
    probabilities = {}
    for label, params in parameters.items():
        probabilities[label] = 1
        for i in range(len(params)):
            mean, stdev, _ = params[i]
            x = instance[i]
            logging.debug("mean: {}, stdev: {}, x: {}".format(mean, stdev, x))
            probabilities[label] *= calculate_probability(x, mean, stdev)
    
    best_label = None
    best_prob = -1
    for label, prob in probabilities.items():
        if prob > best_prob:
            best_label = label
            best_prob = prob

    logging.debug("probabilities: {}".format(probabilities))
    logging.debug("best_label: {}".format(best_label))

    return best_label

def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))

def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[0] = int(tmp[0].split("-")[1]) # month
            tmp[1] = int(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = float(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = int(tmp[7])
            tmp[8] = float(tmp[8])
            tmp[9] = int(tmp[9])
            instances.append([tmp[0], tmp[1], tmp[8]]) # month & is_holiday & power
            labels.append(tmp[-1])
    return instances, labels

def run(train_file, test_file):
    # training phase
    instances, labels = load_raw_data(train_file)
    logging.debug("instances: {}".format(instances))
    logging.debug("labels: {}".format(labels))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)
    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)
    
    # report
    report(predictions, labels)

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True, metavar="<file path to the training dataset>", help="File path of the training dataset", default="training.csv")
    parser.add_argument("-u", "--testing", required=True, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="testing.csv")
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")

    args = parser.parse_args()
    return args

def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)

if __name__ == "__main__":
    main()
