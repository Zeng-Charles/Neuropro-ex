import numpy as np
import wfdb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

import dataprocess
from FeatureExtractionSelection import process_all_data
from cal_features import calculate_features


if __name__ == "__main__":

    lowcut = 10.0
    highcut = 500.0
    cutoff_freq = 50.0

    label_maintenance = np.loadtxt('label_maintenance.txt', delimiter=',')
    pn_dir = 'hd-semg/1.0.0/pr_dataset/subject02_session1'

    data_list = []

    for i in [1, 3, 5, 7, 9]:
        sample_name = f"maintenance_raw_sample{i}"
        record = wfdb.rdrecord(sample_name, pn_dir=pn_dir)
        processed_data = dataprocess.dataprocess(record.p_signal, lowcut, highcut, cutoff_freq, record.fs)
        data_list.append(processed_data)  


    all_window = process_all_data(data_list)
    print("all_window.shape:", all_window.shape)

    selected_feature_names = [
        "waveform_length",
        "variance_central_frequency",
        "zero_crossings",
        "maximum_amplitude"
    ]

    all_windows_features = calculate_features(all_window, selected_feature_names)
    features_flattened = all_windows_features.reshape(all_windows_features.shape[0],-1)
    #需要后续简化label操作！
    label = label = np.array([1] * 162 + [2] * 162 + [3] * 162 + [4] * 162 + [5] * 162)

    train_data, test_data, train_label, test_label = train_test_split(features_flattened, label, train_size=0.7, test_size=0.3, random_state=42)
    scaler = StandardScaler()

    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    print(f"Please select a model (svm/lda/rf):")
    index = input().strip().lower() 

    if index == 'svm':
        #SVM
        model = SVC(kernel='rbf',random_state=42)
        model.fit(train_data, train_label)

        accuracy = model.score(test_data, test_label)
        print(f"accuracy of SVM model:{accuracy:.4f}" )

    elif index == 'lda':
        #LDA
        model = LinearDiscriminantAnalysis()
        model.fit(train_data, train_label)

        accuracy = model.score(test_data, test_label)
        print(f"accuracy of LDA model:{accuracy:.4f}" )

        X_train_2D = model.transform(train_data)
        plt.figure(figsize=(8, 6))
        for label in np.unique(train_label):
            plt.scatter(
                X_train_2D[train_label == label, 0],
                X_train_2D[train_label == label, 1],
                label=f"Class {label}"
            )
        plt.title("LDA: Training Data Transformed into 2D Space")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.legend()
        plt.grid()
        plt.show()
    
    elif index =='rf':
        #random forest
        model = RandomForestClassifier()
        model.fit(train_data, train_label)

        accuracy = model.score(test_data, test_label)
        print(f"accuracy of RandomForest model:{accuracy:.4f}" )

        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0], max_depth=3, filled=True, feature_names=None, class_names=[str(i) for i in np.unique(label)])
        plt.title("Decision Tree from Random Forest")
        plt.show()

    else:
        print("Invalid model selection! Please choose 'svm', 'lda', or 'randomforest'.")