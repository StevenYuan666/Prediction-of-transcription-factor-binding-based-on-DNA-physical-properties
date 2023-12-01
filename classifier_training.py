import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import sklearn
from utils import set_seed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def preprocess(tf):
    bad_words = ['>']
    list_DNAprops = ["MGW", "HelT", "Roll", "ProT"]
    print(list_DNAprops)
    # Create textfiles that will later be converted to pandas dataframe
    for prop in list_DNAprops:
        filein = f"Data/DNA_shape/{tf}/neg.{prop}"
        fileout = f"Data/DNA_shape/{tf}/neg.{prop}_processed"

        with open(filein) as oldfile, open(fileout, 'w') as newfile:
            n = 20
            list_header = [prop] * n
            # print(list_header)
            for j in range(n):
                list_header[j] = list_header[j] + str(j + 1)
            header = ','.join(list_header)
            print(header)
            newfile.write(header)
            newfile.write("\n")
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    newfile.write(line)
    files_neg = [f"Data/DNA_shape/{tf}/neg.{prop}_processed" for prop in list_DNAprops]
    print(files_neg)
    for i in range(len(files_neg)):
        if i == 0:
            df_properties_ub = pd.read_csv(files_neg[i])

        else:
            df_property_ub = pd.read_csv(files_neg[i])
            df_properties_ub = pd.concat([df_properties_ub, df_property_ub.reindex(df_properties_ub.index)], axis=1)

    print("Negative Data samples:", df_properties_ub.shape)
    df_properties_ub["label"] = "negative"
    print("Negative Data samples:", df_properties_ub.shape)

    # Create textfiles that will later be converted to pandas dataframe
    for prop in list_DNAprops:
        filein = f"Data/DNA_shape/{tf}/pos.{prop}"
        fileout = f"Data/DNA_shape/{tf}/pos.{prop}_processed"

        with open(filein) as oldfile, open(fileout, 'w') as newfile:
            n = 20
            list_header = [prop] * n
            # print(list_header)
            for j in range(n):
                list_header[j] = list_header[j] + str(j + 1)
            header = ','.join(list_header)
            print(header)
            newfile.write(header)
            newfile.write("\n")
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    newfile.write(line)
    files_pos = [f"Data/DNA_shape/{tf}/pos.{prop}_processed" for prop in list_DNAprops]
    print(files_pos)

    for i in range(len(files_pos)):
        if i == 0:
            df_properties_b = pd.read_csv(files_pos[i])
        else:
            df_property_b = pd.read_csv(files_pos[i])
            df_properties_b = pd.concat([df_properties_b, df_property_b.reindex(df_properties_b.index)], axis=1)

    print(df_properties_b.shape)
    df_properties_b["label"] = "bound"
    print(df_properties_b.shape)

    # Combine both positive and negative dataframes
    df_properties_both = pd.concat([df_properties_ub, df_properties_b], axis=0)
    print(df_properties_both.shape)

    # Remove columns with all NaN values
    df_properties_both = df_properties_both.dropna(how='all', axis=1)
    print(df_properties_both.shape)
    return df_properties_both


def train_KNN(data):
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

        # print("Accuracy: each fold:", acc_score)
        print("Number of nearest neighbour:", n)
        print("Average accuracy:", avg_acc_score)
        print("best n:", best_n)
    t_end = time.time()
    t_total = t_end - t_start
    print("Running time (s):", t_total)
    print("Number of neighbours with the highest accuracy:", best_n)
    print("Highest accuracy achieved:", best_accuracy)
    plt.title("TBP: Number of nearest neighbours vs accuracy")
    plt.xlabel("Number of nearest neighbours")
    plt.ylabel("Accuracy")
    plt.plot(n_list, test_accuracy)
    plt.show()


def train_logistic_regression(data):
    df_shuffled = sklearn.utils.shuffle(data)
    t_start2 = time.time()

    X = df_shuffled.iloc[:, :-1]
    y = df_shuffled.iloc[:, -1]

    n_list = [i for i in range(1, 100)]
    test_accuracy = []

    k = 5
    kf = KFold(n_splits=k)
    classifier = LogisticRegression(max_iter=10000, solver='lbfgs')

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
    print("Average accuracy:", avg_acc_score)
    t_end2 = time.time()
    t_total2 = t_end2 - t_start2
    print("Running time (s):", t_total2)

    pipe = Pipeline([('classifier', LogisticRegression(max_iter=10000, solver='lbfgs'))])
    param_grid = [
        {'classifier': [LogisticRegression(max_iter=10000, solver='lbfgs')],
         'classifier__penalty': ['l1', 'l2'],  # L1 and L2 regularizations
         'classifier__C': np.logspace(-4, 4, 20),  # Strengths of regularization
         'classifier__solver': ['liblinear']
         }
    ]

    # Create a grid search object
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_classifier = clf.fit(X_train, y_train)
    print(best_classifier)


if __name__ == "__main__":
    set_seed(0)
    tf = "TBP"
    df_properties_both = preprocess(tf)
    train_KNN(df_properties_both)
    train_logistic_regression(df_properties_both)
