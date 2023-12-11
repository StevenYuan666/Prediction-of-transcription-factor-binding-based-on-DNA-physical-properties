import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from dataloader import get_data_loader
from model import DNASequenceTransformer, ThreeLayerNN
from utils import set_seed


def preprocess(tf, verbose=False):
    bad_words = [">"]
    list_DNAprops = ["MGW", "HelT", "Roll", "ProT"]
    if verbose:
        print(list_DNAprops)
    # Create textfiles that will later be converted to pandas dataframe
    for prop in list_DNAprops:
        filein = f"Data/DNA_shape/{tf}/neg.{prop}"
        fileout = f"Data/DNA_shape/{tf}/neg.{prop}_processed"

        with open(filein) as oldfile, open(fileout, "w") as newfile:
            n = 20
            list_header = [prop] * n
            # print(list_header)
            for j in range(n):
                list_header[j] = list_header[j] + str(j + 1)
            header = ",".join(list_header)
            if verbose:
                print(header)
            newfile.write(header)
            newfile.write("\n")
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    newfile.write(line)
    files_neg = [f"Data/DNA_shape/{tf}/neg.{prop}_processed" for prop in list_DNAprops]
    if verbose:
        print(files_neg)
    for i in range(len(files_neg)):
        if i == 0:
            df_properties_ub = pd.read_csv(files_neg[i])

        else:
            df_property_ub = pd.read_csv(files_neg[i])
            df_properties_ub = pd.concat(
                [df_properties_ub, df_property_ub.reindex(df_properties_ub.index)],
                axis=1,
            )
    if verbose:
        print("Negative Data samples:", df_properties_ub.shape)
    df_properties_ub["label"] = "negative"
    if verbose:
        print("Negative Data samples:", df_properties_ub.shape)

    # Create textfiles that will later be converted to pandas dataframe
    for prop in list_DNAprops:
        filein = f"Data/DNA_shape/{tf}/pos.{prop}"
        fileout = f"Data/DNA_shape/{tf}/pos.{prop}_processed"

        with open(filein) as oldfile, open(fileout, "w") as newfile:
            n = 20
            list_header = [prop] * n
            # print(list_header)
            for j in range(n):
                list_header[j] = list_header[j] + str(j + 1)
            header = ",".join(list_header)
            if verbose:
                print(header)
            newfile.write(header)
            newfile.write("\n")
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    newfile.write(line)
    files_pos = [f"Data/DNA_shape/{tf}/pos.{prop}_processed" for prop in list_DNAprops]
    if verbose:
        print(files_pos)

    for i in range(len(files_pos)):
        if i == 0:
            df_properties_b = pd.read_csv(files_pos[i])
        else:
            df_property_b = pd.read_csv(files_pos[i])
            df_properties_b = pd.concat(
                [df_properties_b, df_property_b.reindex(df_properties_b.index)], axis=1
            )
    if verbose:
        print(df_properties_b.shape)
    df_properties_b["label"] = "bound"
    if verbose:
        print(df_properties_b.shape)

    # Combine both positive and negative dataframes
    df_properties_both = pd.concat([df_properties_ub, df_properties_b], axis=0)
    if verbose:
        print("Combined Data samples before balancing:", df_properties_both.shape)

    # Balance the dataset
    # Find the smaller of the two classes (negative or bound)
    min_class_size = min(df_properties_ub.shape[0], df_properties_b.shape[0])

    # Sample from the larger class to match the size of the smaller class
    df_properties_neg_balanced = df_properties_ub.sample(n=min_class_size)
    df_properties_bound_balanced = df_properties_b.sample(n=min_class_size)

    # Combine the balanced datasets
    df_properties_both_balanced = pd.concat(
        [df_properties_neg_balanced, df_properties_bound_balanced], axis=0
    )

    # Shuffle the combined dataset
    df_properties_both_balanced = df_properties_both_balanced.sample(
        frac=1
    ).reset_index(drop=True)

    if verbose:
        print("Balanced Data samples:", df_properties_both_balanced.shape)

    # Remove columns with all NaN values
    df_properties_both_balanced = df_properties_both_balanced.dropna(how="all", axis=1)
    if verbose:
        print(df_properties_both_balanced.shape)

    return df_properties_both_balanced


def train_KNN(data, if_draw=False, verbose=False):
    df_shuffled = sklearn.utils.shuffle(data)
    t_start = time.time()

    X = df_shuffled.iloc[:, :-1]
    y = df_shuffled.iloc[:, -1]

    n_list = [i for i in range(1, 100)]
    test_accuracy = []

    best_accuracy = 0
    best_n = 0

    for i in range(1, 100):
        n = i  # number of neighbours i.e. hyperparameter of KNN
        k = 5
        kf = KFold(n_splits=k)
        knn1 = KNeighborsClassifier(n_neighbors=n)  # change n_neighbors as necessary

        acc_score = []
        for train_index, test_index in kf.split(X):
            # print(train_index, test_index)

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Feature scaling
            s = StandardScaler()
            s.fit(X_train)
            X_train = s.transform(X_train)
            X_test = s.transform(X_test)

            # Fit the model
            knn1.fit(X_train, y_train)
            # Make predictions
            pred_values = knn1.predict(X_test)

            accuracy = accuracy_score(pred_values, y_test)
            acc_score.append(accuracy)

        avg_acc_score = sum(acc_score) / k
        test_accuracy.append(avg_acc_score)

        if avg_acc_score > best_accuracy:
            best_accuracy = avg_acc_score
            best_n = n

        if verbose:
            # print("Accuracy: each fold:", acc_score)
            print("Number of nearest neighbour:", n)
            print("Average accuracy:", avg_acc_score)
            print("best n:", best_n)
    t_end = time.time()
    t_total = t_end - t_start
    if verbose:
        print("Running time (s):", t_total)
        print("Number of neighbours with the highest accuracy:", best_n)
        print("Highest accuracy achieved:", best_accuracy)
    if if_draw:
        plt.title("TBP: Number of nearest neighbours vs accuracy")
        plt.xlabel("Number of nearest neighbours")
        plt.ylabel("Accuracy")
        plt.plot(n_list, test_accuracy)
        plt.show()

    return best_accuracy


def train_logistic_regression(data, verbose=False):
    df_shuffled = sklearn.utils.shuffle(data)
    t_start2 = time.time()

    X = df_shuffled.iloc[:, :-1]
    y = df_shuffled.iloc[:, -1]

    n_list = [i for i in range(1, 100)]
    test_accuracy = []

    k = 5
    kf = KFold(n_splits=k)
    classifier = LogisticRegression(max_iter=10000, solver="lbfgs")

    acc_score = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        classifier.fit(X_train, y_train)
        pred_values = classifier.predict(X_test)

        accuracy = accuracy_score(pred_values, y_test)
        acc_score.append(accuracy)

    avg_acc_score = sum(acc_score) / k
    test_accuracy.append(avg_acc_score)

    # print("Accuracy: each fold:", acc_score)
    if verbose:
        print("Average accuracy:", avg_acc_score)
    t_end2 = time.time()
    t_total2 = t_end2 - t_start2
    if verbose:
        print("Running time (s):", t_total2)

    pipe = Pipeline(
        [("classifier", LogisticRegression(max_iter=10000, solver="lbfgs"))]
    )
    param_grid = [
        {
            "classifier": [LogisticRegression(max_iter=10000, solver="lbfgs")],
            "classifier__penalty": ["l1", "l2"],  # L1 and L2 regularizations
            "classifier__C": np.logspace(-4, 4, 20),  # Strengths of regularization
            "classifier__solver": ["liblinear"],
        }
    ]

    # Create a grid search object
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_classifier = clf.fit(X_train, y_train)
    if verbose:
        print(best_classifier)

    # Make predictions and evaluate
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if verbose:
        print("Best Logistic Regression Accuracy:", accuracy)

    return accuracy


def train_SVM(data, verbose=False):
    df_shuffled = sklearn.utils.shuffle(data)
    X = df_shuffled.iloc[:, :-1]
    y = df_shuffled.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    svm_model = SVC(kernel="linear")  # Use linear kernel

    # Fit the model
    svm_model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if verbose:
        print("SVM Accuracy:", accuracy)

    return accuracy


def train_deep_model(
    model, data_loader, epochs, learning_rate, weight_decay, verbose=False
):
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_accuracy = 0
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for features, labels in data_loader:
            # Forward pass
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels.float())
            # print("Output:", outputs)
            # print("Labels:", labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate the model
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                predicted = (outputs.squeeze() > 0.5).long()
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

        accuracy = accuracy_score(y_true, y_pred)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    if verbose:
        print(f"Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


if __name__ == "__main__":
    # Run 5 runs of the code with different random seeds
    acc_dict = {
        "KNN": [],
        "Logistic Regression": [],
        "SVM": [],
        "DNN": [],
        "Transformer": [],
    }
    tf = "TBP"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(5):
        print("Seed:", i)
        df_properties_both = preprocess(tf, verbose=False)
        set_seed(i)
        knn_acc = train_KNN(df_properties_both, if_draw=False, verbose=False)
        log_acc = train_logistic_regression(df_properties_both, verbose=False)
        svm_acc = train_SVM(df_properties_both, verbose=False)
        # Train DNN on GPU
        data_loader = get_data_loader(df_properties_both, batch_size=32, shuffle=True)
        model = ThreeLayerNN(
            input_size=66, hidden_size1=16, hidden_size2=16, num_classes=1
        )
        model.to(device)
        mlp_acc = train_deep_model(
            model,
            data_loader,
            epochs=10,
            learning_rate=1e-2,
            weight_decay=0,
            verbose=False,
        )
        # Train Transformer on GPU
        model = DNASequenceTransformer(num_features=66, num_classes=1)
        model.to(device)
        tra_acc = train_deep_model(
            model,
            data_loader,
            epochs=100,
            learning_rate=1e-3,
            weight_decay=0,
            verbose=False,
        )
        acc_dict["KNN"].append(knn_acc)
        acc_dict["Logistic Regression"].append(log_acc)
        acc_dict["SVM"].append(svm_acc)
        acc_dict["DNN"].append(mlp_acc)
        acc_dict["Transformer"].append(tra_acc)

    # Print average and standard deviation of the accuracies
    print(acc_dict)
    for key in acc_dict:
        print(key)
        print("Mean:", np.mean(acc_dict[key]))
        print("Standard deviation:", np.std(acc_dict[key]))
        print("--------------------")
    # preprocess(tf, verbose=True)
