import os
import sys
import argparse
import logging
from math import exp, pi, sqrt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE 
import pandas as pd
import numpy as np

def preprocess_data(instances):
    df = pd.DataFrame(instances)
    
    # 날짜 특성 처리 (예: 'YYYY-MM-DD' 형식)
    for column in df.columns:
        if df[column].dtype == 'object' and df[column].str.contains('-').any():
            try:
                df[column] = pd.to_datetime(df[column]).astype(np.int64) // 10**9  # Unix 타임스탬프로 변환
            except:
                pass  # 변환이 안 되는 경우는 넘어감
    
    # 범주형 특성 처리 (예: 문자열)
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le  # 인코더를 저장하여 나중에 역변환에 사용할 수 있게 함
    
    return df.values  # numpy 배열로 반환

def feature_engineering(instances, n_components=3):
    # 데이터 표준화
    scaler = StandardScaler()
    instances = scaler.fit_transform(instances)

    # PCA 적용
    pca = PCA(n_components=n_components)
    processed_instances = pca.fit_transform(instances)
    
    return processed_instances

def training_automated(instances, labels):
    dataset = {}
    for i in range(len(instances)):
        label = labels[i]
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(instances[i])

    parameters = {}
    for label, instances in dataset.items():
        parameters[label] = []
        for i in range(len(instances[0])):
            mean = np.mean([instance[i] for instance in instances])
            stdev = np.std([instance[i] for instance in instances])
            parameters[label].append((mean, stdev, len(instances)))
    
    return parameters

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def predict_automated(instance, parameters):
    probabilities = {}
    for label, params in parameters.items():
        probabilities[label] = 1
        for i in range(len(params)):
            mean, stdev, _ = params[i]
            x = instance[i]
            probabilities[label] *= calculate_probability(x, mean, stdev)
    
    return max(probabilities, key=probabilities.get)

def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    accuracy = accuracy_score(answers, predictions) * 100

    precision, recall, f1, _ = precision_recall_fscore_support(answers, predictions, average='binary')

    logging.info("Accuracy: {:.2f}%".format(accuracy))
    logging.info("Precision: {:.2f}%".format(precision * 100))
    logging.info("Recall: {:.2f}%".format(recall * 100))

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

def run_automated(train_file, test_file):
    # 데이터 로드 및 전처리
    instances, labels = load_raw_data(train_file)
    
    instances = preprocess_data(instances)  # 훈련 데이터 전처리
    
    # SMOTE로 오버샘플링
    smote = SMOTE()
    instances, labels = smote.fit_resample(instances, labels)
    
    # 특성 공학 (PCA 적용)
    processed_instances = feature_engineering(instances)
    
    # 훈련
    parameters = training_automated(processed_instances, labels)
    
    # 테스트 데이터 처리 및 예측
    test_instances, test_labels = load_raw_data(test_file)
    
    test_instances = preprocess_data(test_instances)  # 테스트 데이터 전처리
    
    test_processed_instances = feature_engineering(test_instances)  # PCA 적용
    
    predictions = [predict_automated(instance, parameters) for instance in test_processed_instances]
    
    # 결과 보고
    report(predictions, test_labels)

def command_line_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--training", required=True,
                        metavar="<file path to the training dataset>",
                        help="File path of the training dataset",
                        default="training.csv")
                        
    parser.add_argument("-u", "--testing", required=True,
                        metavar="<file path to the testing dataset>",
                        help="File path of the testing dataset",
                        default="testing.csv")
                        
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
                        type=str, default="INFO")
    
    args = parser.parse_args()
    return args

# Main function to execute the script
def main():
    
    # Command line arguments parsing and logging configuration 
    args = command_line_args()
    
    logging.basicConfig(level=args.log)

    # Check existence of training and testing datasets 
    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    # Run automated training and testing 
    run_automated(args.training, args.testing)

if __name__ == "__main__":
   main()