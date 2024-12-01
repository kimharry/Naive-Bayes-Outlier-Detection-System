import os
import sys
import argparse
import logging
from math import exp, pi, sqrt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

def preprocess_data(instances):
    """
    숫자가 아닌 특성을 숫자형으로 변환하는 함수:
    - 날짜를 Unix 타임스탬프로 변환.
    - 범주형 데이터를 숫자로 인코딩.
    """
    # instances가 pandas DataFrame인 경우 변환
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

def feature_engineering(instances, labels):
    # 특성 선택
    mutual_info = mutual_info_classif(instances, labels, discrete_features=False)
    selected_features = np.argsort(mutual_info)[-3:]  # 상위 3개 특성 선택
    
    # 선택된 특성만 사용
    processed_instances = instances[:, selected_features]
    
    # 표준화
    scaler = StandardScaler()
    processed_instances = scaler.fit_transform(processed_instances)
    
    return processed_instances, selected_features

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
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
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
    precision = round(tp / (tp + fp), 2) * 100 if (tp + fp) > 0 else 0

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100 if (tp + fn) > 0 else 0

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
            instances.append(tmp[:-1])  # 마지막 열을 제외한 모든 열을 인스턴스로
            labels.append(int(tmp[-1]))  # 마지막 열을 레이블로
    return instances, labels

def run_automated(train_file, test_file):
    # 데이터 로드 및 전처리
    instances, labels = load_raw_data(train_file)
    instances = preprocess_data(instances)  # 훈련 데이터 전처리
    
    # 특성 공학
    processed_instances, selected_features = feature_engineering(instances, labels)
    
    # 훈련
    parameters = training_automated(processed_instances, labels)
    
    # 테스트
    test_instances, test_labels = load_raw_data(test_file)
    test_instances = preprocess_data(test_instances)  # 테스트 데이터 전처리
    test_instances = test_instances[:, selected_features]
    predictions = [predict_automated(instance, parameters) for instance in test_instances]
    
    # 결과 보고
    report(predictions, test_labels)

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

    run_automated(args.training, args.testing)

if __name__ == "__main__":
    main()