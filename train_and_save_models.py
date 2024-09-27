import os
import pandas as pd
import numpy as np
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.inspection import permutation_importance
import joblib
from sklearn.metrics import f1_score
from misc_helpers import convert_suffix_to_int, MulticollinearityFilter, SimpleNN, DeepNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # train on GPU for the NNs

# function to preprocess the three datasets
def prepare_dataset(dataset_name):
    if dataset_name == "CICIDS":
        df = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv').fillna(-1)
        df.replace([-np.inf], -1, inplace=True)
        df.replace([np.inf], -1, inplace=True)
        
        columns_to_drop = [" Timestamp", "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port"]
        df = df.drop(columns=columns_to_drop, axis=1)
        
        X = df.drop(' Label', axis=1) # some columns have a space " " in front of their name, so this is intentional here
        y = df[' Label']
        
    elif dataset_name == "CIDDS":
        df = pd.read_csv('CIDDS-001-internal-week1.csv').fillna(-1)
        df.replace([-np.inf], -1, inplace=True)
        df.replace([np.inf], -1, inplace=True)
        df['Bytes'] = df['Bytes'].apply(convert_suffix_to_int).astype(int)
        
        columns_to_drop = ["Date first seen", "Src IP Addr", "Src Pt", "Dst IP Addr", "Dst Pt", "class", "attackID", "attackDescription"]
        df = df.drop(columns=columns_to_drop, axis=1)
        
        df = pd.get_dummies(df, columns=['Proto'], drop_first=True) # proto is a string
        
        # convert TCP flags, they are given like "A...R.."
        flag_types = ['U', 'A', 'P', 'R', 'S', 'F']
        for flag in flag_types:
            df[flag] = df['Flags'].apply(lambda x: 1 if flag in x else 0)
        df = df.drop('Flags', axis=1)
        
        X = df.drop('attackType', axis=1) # we use attackType as label here, since it contains the concrete attack names
        y = df['attackType']
    
    elif dataset_name == "EdgeIIoT":
        df = pd.read_csv('DNN-EdgeIIoT-dataset.csv').fillna(-1)
        df.replace([-np.inf], -1, inplace=True)
        df.replace([np.inf], -1, inplace=True)
        
        
        ### see preprocessing proposed by authors: https://www.kaggle.com/code/mohamedamineferrag/edge-iiotset-pre-processing
        # columns to drop as proposed by authors
        columns_to_drop = [
            "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4", 
            "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", 
            "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", 
            "tcp.dstport", "udp.port", "mqtt.msg", "Attack_label"
        ]
        df = df.drop(columns=columns_to_drop, axis=1)
        
        # columns to encode as proposed by authors
        df = pd.get_dummies(df, columns=[
            'http.request.method', 'http.referer', 'http.request.version', 
            'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'
        ], drop_first=True)
        
        X = df.drop('Attack_type', axis=1)
        y = df['Attack_type']

    return X, y

# called when complexty = binary
def binarize_labels(y, dataset_name):
    if dataset_name == "CICIDS":
        y_binary = np.where(y == 'BENIGN', 0, 1)
    elif dataset_name == "CIDDS":
        y_binary = np.where(y == '---', 0, 1)
    elif dataset_name == "EdgeIIoT":
        y_binary = np.where(y == 'Normal', 0, 1)
    return y_binary

# this method saves the *raw* datasplits for transparency (so without any feature selection etc., without any respect to multi or binary classification etc.)
def save_train_test_split(X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded, dataset_name):
    dataset_dir = os.path.join(dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # save the *raw* splits (i.e., withouth preprocessing) in the global dataset folder
    pd.DataFrame(X_train).to_csv(os.path.join(dataset_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(dataset_dir, "X_test.csv"), index=False)
    
    # save decoded (original) labels in the global dataset folder
    pd.Series(y_train).to_csv(os.path.join(dataset_dir, "y_train_decoded.csv"), index=False, header=["Label"])
    pd.Series(y_test).to_csv(os.path.join(dataset_dir, "y_test_decoded.csv"), index=False, header=["Label"])
    
    # save enecoded labels in the global dataset folder
    pd.Series(y_train_encoded).to_csv(os.path.join(dataset_dir, "y_train_encoded.csv"), index=False, header=["Label"])
    pd.Series(y_test_encoded).to_csv(os.path.join(dataset_dir, "y_test_encoded.csv"), index=False, header=["Label"])

# this saves labels for mult/binary again in the <dataset>/<complexcity> folder
# we reuse it for every selection method, so we can save it slightly more globally than the *preprocessed* training data
def save_labels(y_train, y_test, dataset_name, complexity):
    complexity_dir = os.path.join(dataset_name, complexity)
    os.makedirs(complexity_dir, exist_ok=True)
    pd.DataFrame(y_train).to_csv(os.path.join(complexity_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(complexity_dir, "y_test.csv"), index=False)

# function to load the dataset splits
def load_train_test_split(dataset_name):
    # Load X_train, X_test globally from the dataset folder
    dataset_dir = os.path.join(dataset_name)
    X_train = pd.read_csv(os.path.join(dataset_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(dataset_dir, "X_test.csv"))
    y_train_encoded = pd.read_csv(os.path.join(dataset_dir, "y_train_encoded.csv")).values.flatten()
    y_test_encoded = pd.read_csv(os.path.join(dataset_dir, "y_test_encoded.csv")).values.flatten()
    
    return X_train, X_test, y_train_encoded, y_test_encoded

# load *specific* labels for binary or multi classification
def load_labels(dataset_name, complexity):
    # Load y_train and y_test for the specified complexity
    complexity_dir = os.path.join(dataset_name, complexity)
    y_train = pd.read_csv(os.path.join(complexity_dir, "y_train.csv")).values.flatten()
    y_test = pd.read_csv(os.path.join(complexity_dir, "y_test.csv")).values.flatten()
    
    return y_train, y_test

# different selection methods
# in the paper we only use impurity
def feature_selection(X_train, X_test, y_train, selection_method, num_features_total):
    
    # filter 0 var features
    selector = VarianceThreshold(threshold=0)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    
    # we wanna save the actual feature names as they get lost when transforming the data etc.
    # so here we get the appropriate ones after filtering 0 var features
    selected_feature_names = np.array(X_train.columns)[selector.get_support(indices=True)]
    
    # scale between 0 and 1
    # doesn't filter out any features, so no need to chaange selected_feature_names
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train_sel)
    X_test_scale = scaler.transform(X_test_sel)
    
    if selection_method == "permutation":
        # this RF is only used for feature selection -> nothing to do with the other RF for the explainer/classification
        model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42, n_jobs=5)
        model.fit(X_train_scale, y_train)
        perm_importance = permutation_importance(model, X_train_scale, y_train, n_repeats=5, random_state=42, n_jobs=1)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1] # sort idx higest -> lowest
        top_features = sorted_idx[:num_features_total] # retrieve idx of top features (in the paper = 10)
        X_train_reduced = X_train_scale[:, top_features] # retrive top features in training data via the top idx
        X_test_reduced = X_test_scale[:, top_features]  # same for testing data
        selected_feature_names = selected_feature_names[top_features] # now filter out the selected feature names again; note that they are also now in the same order as columns in the data
    
    elif selection_method == "impurity":
        # workflow almost exactly like for permutation importance
        model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42, n_jobs=5)
        model.fit(X_train_scale, y_train)
        impurity_importance = model.feature_importances_ # this is the only difference
        sorted_idx = np.argsort(impurity_importance)[::-1]
        top_features = sorted_idx[:num_features_total]
        X_train_reduced = X_train_scale[:, top_features]
        X_test_reduced = X_test_scale[:, top_features]
        selected_feature_names = selected_feature_names[top_features]
    
    elif selection_method == "selectkbest":
        # importance via ANOVA
        selector = SelectKBest(f_classif, k=num_features_total)
        X_train_reduced = selector.fit_transform(X_train_scale, y_train)
        X_test_reduced = selector.transform(X_test_scale)
        top_features = selector.get_support(indices=True)
        selected_feature_names = selected_feature_names[top_features]
    
    elif selection_method == "correlation":
        # this is not really an importance measure, but it clusters features and selects on of each cluster -> removes correlation
        mc_filter = MulticollinearityFilter(n_clusters=num_features_total)
        X_train_reduced = mc_filter.fit_transform(X_train_scale).to_numpy()
        X_test_reduced = mc_filter.transform(X_test_scale).to_numpy()
        selected_feature_names = selected_feature_names[mc_filter.selected_features]
        
    elif selection_method == "all":
        # option to just keep all features; selected feature names already declared before the if statement
        X_train_reduced = X_train_scale
        X_test_reduced = X_test_scale
    
    return X_train_reduced, X_test_reduced, selected_feature_names

# saves the preprocessed data in the folders <dataset>/<complexity>/<num_features>/<selection_method>
def save_scaled_selected_data(X_train_reduced, X_test_reduced, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_reduced).to_csv(os.path.join(output_dir, "X_train_reduced.csv"), index=False)
    pd.DataFrame(X_test_reduced).to_csv(os.path.join(output_dir, "X_test_reduced.csv"), index=False)

# method to train the tree-based models
def train_tree_models(X_train_reduced, y_train):
    dt_model = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_model.fit(X_train_reduced, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=5)
    rf_model.fit(X_train_reduced, y_train)
    
    lgb_model = LGBMClassifier(max_depth=20, n_estimators=50, random_state=42, n_jobs=1)
    lgb_model.fit(X_train_reduced, y_train)
    
    return dt_model, rf_model, lgb_model

# method to train the NN-based models
def train_neural_network(X_train_reduced, y_train, num_epochs, model_type):
    input_dim = X_train_reduced.shape[1]
    hidden_dim = 64
    output_dim = len(np.unique(y_train))
    
    X_train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32).to(device) 
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device) 
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # if SLP = 1 layer, if MLP = 2 layers
    if model_type == 'simple':
        model = SimpleNN(input_dim, hidden_dim, output_dim)
    else:
        model = DeepNN(input_dim, hidden_dim, output_dim)
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    
    return model

# saves feature names for each selection method in <dataset>/<complexity>/<num_features>/<selection_method>
def save_features(selected_feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "selected_features.txt"), "w") as file:
        for feature in selected_feature_names:
            file.write(f"{feature}\n")

# saves models for each selection method in <dataset>/<complexity>/<num_features>/<selection_method>
def save_models(dt_model, rf_model, lgb_model, simple_nn_model, deep_nn_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(dt_model, os.path.join(output_dir, "dt_model.pkl"))
    joblib.dump(rf_model, os.path.join(output_dir, "rf_model.pkl"))
    joblib.dump(lgb_model, os.path.join(output_dir, "lgb_model.pkl"))
    torch.save(simple_nn_model.state_dict(), os.path.join(output_dir, "nn_model_simple.pth"))
    torch.save(deep_nn_model.state_dict(), os.path.join(output_dir, "nn_model_deep.pth"))    


def evaluate_and_save_metrics(dt_model, rf_model, lgb_model, simple_nn_model, deep_nn_model, X_test_reduced, y_test, output_dir):
    # tree predictions
    dt_pred = dt_model.predict(X_test_reduced)
    rf_pred = rf_model.predict(X_test_reduced)
    lgb_pred = lgb_model.predict(X_test_reduced)
    
    X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(device) 
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device) 
    
    # NN predictions
    simple_nn_model.eval()
    with torch.no_grad():
        simple_nn_outputs = simple_nn_model(X_test_tensor)
        _, simple_nn_pred = torch.max(simple_nn_outputs, 1)
    
    deep_nn_model.eval()
    with torch.no_grad():
        deep_nn_outputs = deep_nn_model(X_test_tensor)
        _, deep_nn_pred = torch.max(deep_nn_outputs, 1)
    
    # eval. F1-scores for all models and save them
    metrics = {"Model": [], "F1_Macro": [], "F1_Micro": []}
    
    for model_name, pred in zip(
        ["Decision Tree", "Random Forest", "LightGBM", "Simple Neural Network", "Deep Neural Network"], 
        [dt_pred, rf_pred, lgb_pred, simple_nn_pred.cpu().numpy(), deep_nn_pred.cpu().numpy()]
    ):
        metrics["Model"].append(model_name)
        metrics["F1_Macro"].append(f1_score(y_test, pred, average='macro'))
        metrics["F1_Micro"].append(f1_score(y_test, pred, average='micro'))
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

if __name__ == "__main__":  
    sel_meths_for_eval = ["impurity"] # we only need the models for impurity-based approach
    for dataset_name in ["EdgeIIoT"]: # or CIDDS, CICIDS
        X, y = prepare_dataset(dataset_name)
        
        # split data into training and testing sets before encoding
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(label_encoder_name_mapping)
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # save features, and both encoded and decoded labels (it is a bit redundant, but better save it now than be sorry later...)
        save_train_test_split(X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded, dataset_name)
        
        for multi in [False]: # we only do binary, so multi = False
            complexity = "multi" if multi else "binary"
            
            # binarize labels if multi is False (i.e., binary classification)
            y_train_binary = binarize_labels(y_train, dataset_name) if not multi else y_train_encoded
            y_test_binary = binarize_labels(y_test, dataset_name) if not multi else y_test_encoded
            
            # Save y_train and y_test for each complexity
            save_labels(y_train_binary, y_test_binary, dataset_name, complexity)
            
            for sel_meth in ["correlation", "permutation", "selectkbest", "impurity"]: # here we do all, cause we want the feature importances at least, but training is skipped for most
                for num_features in [10]:
                    # for reproducibility
                    # could be that for the NNs due to training on GPU we still have some variation
                    seed_value = 42
                    np.random.seed(seed_value)
                    random.seed(seed_value)
                    torch.manual_seed(seed_value)
                                       
                    epochs = 25
                    
                    X_train, X_test, _, _ = load_train_test_split(dataset_name) # load raw features (don't need the labels since we load the proper processed ones directly)
                    y_train2, y_test2 = load_labels(dataset_name, complexity) # load proper labels
                    
                    # preprocess data
                    X_train_reduced, X_test_reduced, selected_feature_names = feature_selection(X_train, X_test, y_train2, sel_meth, num_features)
                    
                    # and save preprocessed data
                    output_dir = os.path.join(dataset_name, complexity, sel_meth, str(num_features))
                    save_scaled_selected_data(X_train_reduced, X_test_reduced, output_dir)
                    
                    # also save feature names
                    save_features(selected_feature_names, output_dir)
                    if sel_meth in sel_meths_for_eval:

                        # train NNs
                        deep_nn_model = train_neural_network(X_train_reduced, y_train2, num_epochs=epochs, model_type='deep')
                        simple_nn_model = train_neural_network(X_train_reduced, y_train2, num_epochs=epochs, model_type='simple')
                        
                        # train trees
                        dt_model, rf_model, lgb_model = train_tree_models(X_train_reduced, y_train2)
    
                        # save models
                        save_models(dt_model, rf_model, lgb_model, simple_nn_model, deep_nn_model, output_dir)
                        
                        # eval. models
                        evaluate_and_save_metrics(dt_model, rf_model, lgb_model, simple_nn_model, deep_nn_model, X_test_reduced, y_test2, output_dir)
                    
                    
                    gc.collect() # collect garbage? :)