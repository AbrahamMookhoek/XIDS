# Import libraries
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, TimeDistributed, LSTM, LayerNormalization
from keras import regularizers
from keras.models import load_model
from scipy.stats import mode
shap.initjs()

class Simpleann():
    def __init__(self, csv_name, model_name):
        self.csv_name = csv_name
        self.model_name = model_name

    def load_dataset(self):
        df = pd.read_csv(self.csv_name)

        # Clean up dataset
        nan_value = float("NaN")

        df.replace("", nan_value, inplace=True)
        df.dropna(inplace=True)
        # df['Flag'].astype(object)
        df['Flag'].replace("R",0, inplace=True)
        df['Flag'].replace("T",1, inplace=True)

        for i in range(3, 11):
            df.iloc[:, i] = df.iloc[:, i].apply(int, base=16)

        df.iloc[:, 1] = df.iloc[:, 1].apply(int, base=16)

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        feature_cols = ['ID', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8']

        X = df.loc[:, feature_cols].values
        y = df['Flag'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.array(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=5)

        X_train = np.asarray(X_train).astype(np.float32)

        y_train = np.asarray(y_train).astype(np.float32)

        y_test = np.asarray(y_test).astype(np.float32)

        return X_train, X_test, y_train, y_test

    def ann(self, X_train, y_train, name):
        model_ann = Sequential()
        model_ann.add(Dense(16, input_dim=9, activation='relu', kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2(0.0001)))
        model_ann.add(BatchNormalization())
        model_ann.add(Dropout(0.2))
        model_ann.add(
            Dense(16, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        model_ann.add(BatchNormalization())
        model_ann.add(Dropout(0.2))
        model_ann.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
        model_ann.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

        model_ann.fit(X_train, y_train, epochs=10, batch_size=128)
        model_ann.save('simpleann/' + name)
        print("Done Training!")

    def detect(self, X_test, y_test):

        if self.model_name == 'dos':
            model_ann = load_model('simpleann/dos_model')
        elif self.model_name == 'fuzzy':
            model_ann = load_model('simpleann/fuzzy_model')
        elif self.model_name == 'gear':
            model_ann = load_model('simpleann/gear_model')
        elif self.model_name == 'rpm':
            model_ann = load_model('simpleann/rpm_model')
        else:
            model_ann = load_model('simpleann/normal_model')

        y_pred = model_ann.predict(X_test)

        y_pred = y_pred.astype(int)

        from sklearn import metrics
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print("Precision: ", metrics.precision_score(y_test, y_pred))
        print("Recall: ", metrics.recall_score(y_test, y_pred))
        print("Area Under Curve: ", metrics.roc_auc_score(y_test, y_pred))

        tp = metrics.confusion_matrix(y_test, y_pred)
        print(tp)

        self.t_TN = metrics.confusion_matrix(y_test, y_pred)[0][0]
        self.t_FP = metrics.confusion_matrix(y_test, y_pred)[0][1]
        self.t_FN = metrics.confusion_matrix(y_test, y_pred)[1][0]
        self.t_TP = metrics.confusion_matrix(y_test, y_pred)[1][1]
        #self.t_TP, self.t_FN, self.t_TP, self.t_TN = metrics.confusion_matrix()
        print("True Positive:",self.t_TP)
        print("False Positive:",self.t_FP)
        print("True Negative: ",self.t_TN)
        print("False Negative: ",self.t_FN)
        #fpr = round(self.t_FP / (self.t_FP + self.t_TN), 3)

        shap_X_test = X_test[:10]

        # BEGIN TESTING SHAP
        explainer = shap.Explainer(model_ann.predict, shap_X_test)
        shap_values = explainer(shap_X_test)
        
        # # Summary Plot: show importance of each feature overall
        # print("Summary Plot: ")
        # shap.summary_plot(shap_values, X_test[:20])
        
        # Dependence Plot: show dependence plot for "ID" and "Data8"
        print("Dependence Plot: ")
        shap.dependence_plot("Feature 0", shap_values, shap_X_test[:20], interaction_index="Feature 8")
        
        # force plots: contributions to either 0 or 1 classifications
        print("Force Plot: Class 0")
        shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], shap_X_test.iloc[0, :], matplotlib = True)
        print("Force Plot: Class 1")
        shap.plots.force(explainer.expected_value[1], shap_values[1][6, :], shap_X_test.iloc[6, :],matplotlib = True)
        
        # decision plots: impact of features on decisions (line is decision)
        print("Decision Plot Class 1:")
        shap.decision_plot(explainer.expected_value[1], shap_values[1], shap_X_test[:20].columns)
        print("Decision Plot Class 0:")
        shap.decision_plot(explainer.expected_value[0], shap_values[0], shap_X_test[:20].columns)

        return self.t_TP, self.t_FP, self.t_TN, self.t_FN



def main():
    normal_model = Simpleann('dos_cleaned.csv', 'whoops')
    X_train, X_test, y_train, y_test = normal_model.load_dataset()
    # normal_model.ann(X_train, y_train, 'normal_model')
    normal_model.detect(X_test, y_test)


if __name__ == '__main__':
    main()