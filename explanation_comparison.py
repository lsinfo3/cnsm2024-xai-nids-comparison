import os
import pandas as pd
from sklearn.metrics import f1_score
import shap
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from treeinterpreter import treeinterpreter as ti
import joblib
import torch
import torch.nn as nn
import gc
import random
import torch.nn.functional as F
from captum.attr import DeepLift, IntegratedGradients, Saliency
import warnings
import copy
from misc_helpers import SimpleNN, DeepNN

# these two are some warnings by DLIFT/IG, they just inform the user that require_grad() is set for the input + that it sets hooks and then removes them again...
warnings.filterwarnings("ignore", category=UserWarning, message=".*Input Tensor 0 did not already require gradients.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Setting forward, backward hooks and attributes on non-linear.*")

# some performance warnings; feel free to improve :)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Usage of np.ndarray subset.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*To copy construct from a tensor, it is recommended to use.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*LightGBM binary classifier with TreeExplainer.*")

# Function to load trained models and datasets
def load_models_and_data(dataset_name, selection_method, complexity, num_features):
    
    # construct correct path
    dir_path = os.path.join(dataset_name, complexity, selection_method, str(num_features))
    
    # load already trained tree-based models
    dt_model = joblib.load(os.path.join(dir_path, "dt_model.pkl"))
    rf_model = joblib.load(os.path.join(dir_path, "rf_model.pkl"))
    lgb_model = joblib.load(os.path.join(dir_path, "lgb_model.pkl"))
    
    # load already trained NNs
    output_dim = len(np.unique(pd.read_csv(os.path.join(dir_path, "../../y_train.csv")).values.flatten()))
    
    simple_nn_model = SimpleNN(num_features, 64, output_dim)
    simple_nn_model.load_state_dict(torch.load(os.path.join(dir_path, "nn_model_simple.pth")))
    simple_nn_model.eval()

    deep_nn_model = DeepNN(num_features, 64, output_dim)
    deep_nn_model.load_state_dict(torch.load(os.path.join(dir_path, "nn_model_deep.pth")))
    deep_nn_model.eval()
    
    # load preprocessed data
    X_train_reduced = pd.read_csv(os.path.join(dir_path, "X_train_reduced.csv")).to_numpy()
    X_test_reduced = pd.read_csv(os.path.join(dir_path, "X_test_reduced.csv")).to_numpy()
    
    # load labels
    y_train = pd.read_csv(os.path.join(dir_path, "../../y_train.csv")).values.flatten()
    y_test = pd.read_csv(os.path.join(dir_path, "../../y_test.csv")).values.flatten()

    # load feature names -> they are in the same ordering as the columns in the preprocessed data
    # we need this for the tables in the paper, otherwise we only have the features as integers
    with open(os.path.join(dir_path, "selected_features.txt"), "r") as file:
        selected_feature_names = [line.strip() for line in file.readlines()]
    
    return dt_model, rf_model, lgb_model, simple_nn_model, deep_nn_model, X_train_reduced, X_test_reduced, y_train, y_test, selected_feature_names

def calc_sign_consensus(all_features_dict, num_samples, num_features_total):
    methods = list(all_features_dict.keys())
    num_methods = len(methods)
    detailed_agreement_matrix = np.zeros((num_methods, num_methods, num_samples))
    for sample_index in range(num_samples):        
        for i, method_i in enumerate(methods):
            for j, method_j in enumerate(methods):
                    signs_i = {feature[0]: feature[1] for feature in all_features_dict[method_i][sample_index]}
                    signs_j = {feature[0]: feature[1] for feature in all_features_dict[method_j][sample_index]}
                    # print(signs_i)
                    
                    # check that both methods have the features (should always be the case, cause we utilize all features for this analysis)
                    common_features = set(signs_i.keys()).intersection(set(signs_j.keys()))
                    # print(len(common_features))
                    if len(common_features) != num_features_total:
                        print("This should not happen.")
                    else:
                        agreement_count = sum(signs_i[feature] == signs_j[feature] for feature in common_features)
                        total_features = len(common_features)
                        agreement_percentage = (agreement_count / total_features) * 100
                        detailed_agreement_matrix[i, j, sample_index] = agreement_percentage
    return detailed_agreement_matrix

def calc_ranked_unranked_consensus(all_features_list, num_samples):
    num_methods = len(all_features_list)
    unordered_comparison_matrix = np.zeros((num_methods, num_methods, num_samples)) # init
    ordered_comparison_matrix = np.zeros((num_methods, num_methods, num_samples)) # init
    
    for index in range(num_samples):
        # "all_features" is basically just a list off 18 tuples with all 5 top features for each method
        all_features = [features[index] for _,features in all_features_list]
        # we then iterate through that list and compare its entries with each other
        for i, method1_features in enumerate(all_features):
            for j, method2_features in enumerate(all_features):
                # for unranked/unordered comparison we just look at the intersection
                unordered_comparison_matrix[i, j, index] = len(set(method1_features).intersection(set(method2_features)))
                match_count = 0
                # for ranked/ordered comparison we actually have to iterate manually through the list of features
                # but since they are already sorted, we simply increase the counter for each initial match
                # -> we zip it cause we are only interested in the *initial* X matches, so after one mismatch, we stop counting
                for feat1, feat2 in zip(method1_features, method2_features):
                    if feat1 == feat2:
                        match_count += 1 # if first features match+1, if first two match also+1 etc...
                    else:
                        break  # BUTTTT: we stop counting as soon as they don't match
                ordered_comparison_matrix[i, j, index] = match_count
    return ordered_comparison_matrix, unordered_comparison_matrix


# explainer function that takes a model and produces all explanations for the random samples for all valid explaination methods
def explain_model(model, model_name, X_test_reduced, y_pred, shap_explainer, lime_explainer, saliency, integrated_gradients, deeplift, selected_feature_names, random_indices, use_ti=True, selection_method="impurity", complexity="binary"):
    # the following are lists that will be filled with the feature names from most important -> least important
    # we use that for calculating the matrices/heatmaps of sign/ranked/unranked analyses; the 3 matrices are saved as a .pkl
    shap_top_features = []
    lime_top_features = []
    ti_top_features = []
    saliency_top_features = []
    deeplift_top_features = []
    integrated_gradients_top_features = []
    
    # here we save for *all* samples how much a feature impacted the decision; we save this in a seperate .csv
    global_importance_dict = {}

    print(random_indices[0:5]) # just to ensure consistency among the diff. ML methods... to check that they all explain the same samples
    
    # this is for the singular example which we investigate the explanation for; also saved as a seperate .csv
    feature_importance_data = []
    
    # just to see progress when running explainer script...
    counter=0
    
    for index in random_indices:
        
        print(counter)
        counter+=1
        
        # enable tracking of data if this is the first random sample
        if index == random_indices[0]:
            first_sample = True
        else:
            first_sample = False

        predicted_class = int(y_pred[index]) # convert to int cause the NN-based stuff doesn't like when it's numeric but not int...

        ################ SHAP start
        # we catch an error here, sometimes there are some rounding issues it seem with SHAP; none of them seemed significant
        # see also: https://github.com/shap/shap/issues/930 -> we looked at the tolerance and nothing really seemed out of line, just an issue with the default value to check
        try:
            shap_values_sample = shap_explainer.shap_values(X_test_reduced[index:index+1])
        except Exception as e:
            warnings.warn(f"SHAP additivity check failed: {e}")
            shap_values_sample = shap_explainer.shap_values(X_test_reduced[index:index+1], check_additivity=False)
        
        # this just sorts the feature names accordng to *absolute* impact value
        # we can generate explanations for alles classes, but we are here interested in the predicted class -> "shap_values_sample[predicted_class]"
        shap_features = sorted(zip(selected_feature_names, shap_values_sample[predicted_class].flatten()), key=lambda x: abs(x[1]), reverse=True)
        # store feature names + *signed* impact value (we need it for sign analysis; and can super simple convert it for the other analyses)
        shap_top_features.append([(f[0].strip(), np.sign(f[1])) for f in shap_features])  # Include feature sign
        
        for feature, importance in shap_features: # saving global impact
            key = ("SHAP", model_name, feature)
            if key in global_importance_dict:
                global_importance_dict[key] += abs(importance)
            else:
                global_importance_dict[key] = abs(importance)

        if first_sample: # save sample impact
            for rank, (feature, importance) in enumerate(shap_features, 1):
                feature_importance_data.append([model_name, "SHAP", feature, importance, abs(importance), rank])
        ################ SHAP end
        
        ################ LIME start
        # the NNs do not have the classical "predict_proba" from scikit_learn, so we simply convert the logits to a probability between 0 and 1 via softmax
        # LIME want specifically values between 0 and 1, but the other methods for NNs expect the raw values
        def model_predict_proba(input_data):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
            return probs.detach().numpy()

        # depending if we have an NN or not transform data into np.array
        data_row = X_test_reduced[index].numpy() if isinstance(X_test_reduced, torch.Tensor) else X_test_reduced[index]

        # here we don't have a check like for SHAP; just call the LIME explainer given to the function call
        lime_exp_instance = lime_explainer.explain_instance(
            data_row, 
            model.predict_proba if hasattr(model, 'predict_proba') else model_predict_proba, # see above for the "model_predict_proba" function
            labels=(predicted_class,), # we can generate explanations for alles classes, but we are here interested in the predicted class
            num_features=len(selected_feature_names) # LIME can choose how many features we want for the explanation, so we just choose all, since the other methods do as well
        )
        #print(lime_exp_instance.as_list(predicted_class))
        #print(lime_exp_instance.as_map()[predicted_class])
        # compared to SHAP, LIME is already in that "zipped" formated of "feature,value", so we don't call "zip" here
        #lime_features = sorted(lime_exp_instance.as_list(predicted_class), key=lambda x: abs(x[1]), reverse=True)
        lime_features_int = sorted(lime_exp_instance.as_map()[predicted_class], key=lambda x: abs(x[1]), reverse=True)
        lime_features = [(selected_feature_names[idx], value) for idx, value in lime_features_int]
        #print(lime_features)
        lime_top_features.append([(f[0].strip(), np.sign(f[1])) for f in lime_features])  # Include feature sign

        # same as for SHAP... saving stuff
        for feature, importance in lime_features:
            key = ("LIME", model_name, feature)
            if key in global_importance_dict:
                global_importance_dict[key] += abs(importance)
            else:
                global_importance_dict[key] = abs(importance)
        
        if first_sample:
            for rank, (feature, importance) in enumerate(lime_features, 1):
                feature_importance_data.append([model_name, "LIME", feature, importance, abs(importance), rank])
        ################ LIME end
        
        ################ TreeInterpreter start
        # here it is just a flag "use_ti" for DT and RF
        if use_ti:
            # this is the concept for TI; we are mostly interestend int the contributions
            _, _, contributions = ti.predict(model, X_test_reduced[[index]])
            # get it for the class we are interested in only...
            contributions_class = contributions[0][:, predicted_class].flatten() # it's for some reason as a list of lists of lists -> "contributions[0]"
            # then similar procedure as for SHAP with the sorting
            ti_features = sorted(zip(selected_feature_names, contributions_class), key=lambda x: abs(x[1]), reverse=True)
            ti_top_features.append([(f[0].strip(), np.sign(f[1])) for f in ti_features])  # Include feature sign
            
            # same same same as before
            for feature, importance in ti_features:
                key = ("TI", model_name, feature)
                if key in global_importance_dict:
                    global_importance_dict[key] += abs(importance)
                else:
                    global_importance_dict[key] = abs(importance)

            if first_sample:
                for rank, (feature, importance) in enumerate(ti_features, 1):
                    feature_importance_data.append([model_name, "TI", feature, importance, abs(importance), rank])
        ################ TreeInterpreter end
        
        # for NNs, explanations with captum pckg (DeepLift, IntegratedGradients, Saliency)
        if isinstance(model, nn.Module):
            
            input_sample = torch.tensor(X_test_reduced[index], dtype=torch.float32).unsqueeze(0) # back to tensor

            ################ Saliency start
            # overall super similar to SHAP
            # here we just explain the sample with our wanted target class... -> the class the model predicted
            saliency_attr = saliency.attribute(input_sample, target=predicted_class)
            # and then sort it according to absolute impact
            saliency_features = sorted(zip(selected_feature_names, saliency_attr[0].detach().numpy()), key=lambda x: abs(x[1]), reverse=True)
            saliency_top_features.append([(f[0].strip(), np.sign(f[1])) for f in saliency_features])  # Include feature sign

            # like before
            for feature, importance in saliency_features:
                key = ("Saliency", model_name, feature)
                if key in global_importance_dict:
                    global_importance_dict[key] += abs(importance)
                else:
                    global_importance_dict[key] = abs(importance)

            if first_sample:
                for rank, (feature, importance) in enumerate(saliency_features, 1):
                    feature_importance_data.append([model_name, "Saliency", feature, importance, abs(importance), rank])
            ################ Saliency end
            
            ################ DeepLift start
            # exactly like saliency, since it is the same package
            deeplift_attr = deeplift.attribute(input_sample, target=predicted_class)
            deeplift_features = sorted(zip(selected_feature_names, deeplift_attr[0].detach().numpy()), key=lambda x: abs(x[1]), reverse=True)
            deeplift_top_features.append([(f[0].strip(), np.sign(f[1])) for f in deeplift_features])  # Include feature sign
            
            # just like before
            for feature, importance in deeplift_features:
                key = ("DeepLIFT", model_name, feature)
                if key in global_importance_dict:
                    global_importance_dict[key] += abs(importance)
                else:
                    global_importance_dict[key] = abs(importance)

            if first_sample:
                for rank, (feature, importance) in enumerate(deeplift_features, 1):
                    feature_importance_data.append([model_name, "DeepLIFT", feature, importance, abs(importance), rank])
            ################ DeepLift end
            
            
            ################ Integrated Gradients start
            # exactly like saliency+IG
            ig_attr = integrated_gradients.attribute(input_sample, target=predicted_class)
            ig_features = sorted(zip(selected_feature_names, ig_attr[0].detach().numpy()), key=lambda x: abs(x[1]), reverse=True)
            integrated_gradients_top_features.append([(f[0].strip(), np.sign(f[1])) for f in ig_features])  # Include feature sign
            
            # just like before
            for feature, importance in ig_features:
                key = ("IG", model_name, feature)
                if key in global_importance_dict:
                    global_importance_dict[key] += abs(importance)
                else:
                    global_importance_dict[key] = abs(importance)

            if first_sample:
                for rank, (feature, importance) in enumerate(ig_features, 1):
                    feature_importance_data.append([model_name, "IG", feature, importance, abs(importance), rank])
            ################ Integrated Gradients end
            
    # convert to df for sample explanations
    df_feature_importance_sample = pd.DataFrame(feature_importance_data, columns=["ML Model", "XAI Method", "Feature Name", "Importance Value (Signed)", "Importance Value (Absolute)", "Feature Ranking"])
    
    # save the sample data
    # append the file since the function is called multiple times for every model
    file_path_sample = "feature_importance_sample_"+selection_method+"_"+dataset_name+"_"+complexity+".csv"
    file_path_sample = os.path.join("data", file_path_sample)
    if not os.path.isfile(file_path_sample):
        df_feature_importance_sample.to_csv(file_path_sample, index=False)
    else:
        df_feature_importance_sample.to_csv(file_path_sample, mode='a', header=False, index=False)
        
    # save the global data 
    # here we just save the different model+explainers into different files...
    global_importance_data = [{"XAI Method": key[0], "ML Model": key[1], "Feature Name": key[2], "Total Importance": value} for key, value in global_importance_dict.items()]
    df_global_importance = pd.DataFrame(global_importance_data)
    
    file_path_global = "global_importance_"+model_name+"_"+selection_method+"_"+dataset_name+"_"+complexity+".csv"
    file_path_global = os.path.join("data", file_path_global)
    
    df_global_importance.to_csv(file_path_global, index=False)

    return shap_top_features, lime_top_features, ti_top_features, saliency_top_features, deeplift_top_features, integrated_gradients_top_features


def compare_methods(dataset_name, selection_method, num_features_total, binary):
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    complexity = "binary" if binary else "multi"
    
   
    dt_model, rf_model, lgb_model, simple_nn_model, deep_nn_model, X_train_reduced, X_test_reduced, y_train, y_test, selected_feature_names = load_models_and_data(dataset_name, selection_method, complexity, num_features_total)
    
    # predictions -> we need this for the explanation eval, since we look at the explanations of the *predicted* class
    y_pred_dt = dt_model.predict(X_test_reduced)
    y_pred_rf = rf_model.predict(X_test_reduced)
    y_pred_lgb = lgb_model.predict(X_test_reduced)

    # convert for NN models
    X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32)
    
    with torch.no_grad():
        #print(simple_nn_model(X_test_tensor))
        simple_nn_pred = torch.argmax(simple_nn_model(X_test_tensor), dim=1).numpy()
        deep_nn_pred = torch.argmax(deep_nn_model(X_test_tensor), dim=1).numpy()

    # print some scores... (we already saved them after training too)
    for model_name, y_pred in zip(["DT", "RF", "LGBM", "SLP", "MLP"],[y_pred_dt, y_pred_rf, y_pred_lgb, simple_nn_pred, deep_nn_pred]):
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        print(f"{model_name} F1 Score (Macro): {f1_macro}")
        print(f"{model_name} F1 Score (Micro): {f1_micro}")

    # LIME explanations -> we only need one explainer; not dependant on model
    explainer_lime = LimeTabularExplainer(X_train_reduced, feature_names=selected_feature_names, class_names=[str(i) for i in np.unique(y_train)], mode='classification', random_state=42)

    # SHAP explanations for trees
    explainer_shap_dt = shap.TreeExplainer(dt_model)
    explainer_shap_rf = shap.TreeExplainer(rf_model)
    explainer_shap_lgb = shap.TreeExplainer(lgb_model)

    # SHAP explanations for NNs; diff. SHAP explainer
    background_size = 1000  # typically between 100 and 1000 according to documentation
    background_indices = np.random.choice(X_train_reduced.shape[0], background_size, replace=False)
    background_data = torch.tensor(X_train_reduced[background_indices], dtype=torch.float32)
    
    explainer_shap_simple_nn = shap.DeepExplainer(simple_nn_model, background_data)
    explainer_shap_deep_nn = shap.DeepExplainer(deep_nn_model, background_data)

    num_samples = 1000 # num of samples to evaluate
    random_indices = np.random.choice(X_test_reduced.shape[0], num_samples, replace=False)

    # NN specific explainers
    deeplift_simple_nn = DeepLift(simple_nn_model)
    deeplift_deep_nn = DeepLift(deep_nn_model)
    
    ig_simple_nn  = IntegratedGradients(simple_nn_model)
    ig_deep_nn  = IntegratedGradients(deep_nn_model)
    
    saliency_simple_nn  = Saliency(simple_nn_model)
    saliency_deep_nn  = Saliency(deep_nn_model)
    
    #################### Getting the explanations start
    ### generally: saliency etc. set to "None" for non-NNs and "use_ti" only enabled for RF and DT
    
    ################ DT explanations: disable saliency, IG, DeepLift, but enable TI
    shap_top_features_dt, lime_top_features_dt, ti_top_features_dt, _, _, _ = explain_model(
        dt_model, "DT", X_test_reduced, y_pred_dt, explainer_shap_dt, copy.deepcopy(explainer_lime), None, None, None, selected_feature_names, random_indices=random_indices, use_ti=True, selection_method=selection_method, complexity=complexity)
    
    ################ RF explanations: disable saliency, IG, DeepLift, but enable TI
    shap_top_features_rf, lime_top_features_rf, ti_top_features_rf, _, _, _ = explain_model(
        rf_model, "RF", X_test_reduced, y_pred_rf, explainer_shap_rf, copy.deepcopy(explainer_lime), None, None, None, selected_feature_names, random_indices=random_indices, use_ti=True, selection_method=selection_method, complexity=complexity)
    
    ################ LGBM explanations: disable saliency, IG, DeepLift, also disable TI
    shap_top_features_lgb, lime_top_features_lgb, _, _, _, _ = explain_model(
        lgb_model, "LGBM", X_test_reduced, y_pred_lgb, explainer_shap_lgb, copy.deepcopy(explainer_lime), None, None, None, selected_feature_names, random_indices=random_indices, use_ti=False, selection_method=selection_method, complexity=complexity)
    
    ################ SLP explanations: enable saliency, IG, DeepLift now, still disable TI
    shap_top_features_simple_nn, lime_top_features_simple_nn, _, saliency_top_features_simple_nn, deeplift_top_features_simple_nn, integrated_gradients_top_features_simple_nn = explain_model(
        simple_nn_model, "SLP", X_test_tensor, simple_nn_pred, explainer_shap_simple_nn, copy.deepcopy(explainer_lime), saliency_simple_nn, ig_simple_nn, deeplift_simple_nn, selected_feature_names, random_indices=random_indices,  use_ti=False, selection_method=selection_method, complexity=complexity)
    
    ################ MLP explanations: enable saliency, IG, DeepLift now, still disable TI
    shap_top_features_deep_nn, lime_top_features_deep_nn, _, saliency_top_features_deep_nn, deeplift_top_features_deep_nn, integrated_gradients_top_features_deep_nn = explain_model(
        deep_nn_model, "MLP", X_test_tensor, deep_nn_pred, explainer_shap_deep_nn, copy.deepcopy(explainer_lime), saliency_deep_nn, ig_deep_nn, deeplift_deep_nn, selected_feature_names, random_indices=random_indices, use_ti=False, selection_method=selection_method, complexity=complexity)
    #################### Getting explanations end
    
        
    # select top features from a list of lists
    # -> "all_features" is a list of lists, where each inner list contains tuples of (feature_name, sign)
    def select_top_features(all_features, top_n=5):
        return [[feature for feature, _ in features[:top_n]] for features in all_features]
        
        
    # for ranked+unranked consensus: filter top 5 features and don't need the signs
    all_features_list = [
        ('SHAP DT', select_top_features(shap_top_features_dt)), 
        ('LIME DT', select_top_features(lime_top_features_dt)), 
        ('TI DT', select_top_features(ti_top_features_dt)),
        ('SHAP RF', select_top_features(shap_top_features_rf)), 
        ('LIME RF', select_top_features(lime_top_features_rf)), 
        ('TI RF', select_top_features(ti_top_features_rf)),
        ('SHAP LGBM', select_top_features(shap_top_features_lgb)), 
        ('LIME LGBM', select_top_features(lime_top_features_lgb)),
        ('SHAP SLP', select_top_features(shap_top_features_simple_nn)), 
        ('LIME SLP', select_top_features(lime_top_features_simple_nn)),
        ('SHAP MLP', select_top_features(shap_top_features_deep_nn)), 
        ('LIME MLP', select_top_features(lime_top_features_deep_nn)),
        ('Sal. SLP', select_top_features(saliency_top_features_simple_nn)), 
        ('DLIFT SLP', select_top_features(deeplift_top_features_simple_nn)), 
        ('IG SLP', select_top_features(integrated_gradients_top_features_simple_nn)),
        ('Sal. MLP', select_top_features(saliency_top_features_deep_nn)), 
        ('DLIFT MLP', select_top_features(deeplift_top_features_deep_nn)), 
        ('IG MLP', select_top_features(integrated_gradients_top_features_deep_nn))
    ]
    
    
    # for sign consensus: no filtering; we use all here + need the signs
    all_features_dict = {
        'SHAP DT': shap_top_features_dt, 
        'LIME DT': lime_top_features_dt, 
        'TI DT': ti_top_features_dt,
        'SHAP RF': shap_top_features_rf, 
        'LIME RF': lime_top_features_rf, 
        'TI RF': ti_top_features_rf,
        'SHAP LGBM': shap_top_features_lgb, 
        'LIME LGBM': lime_top_features_lgb,
        'SHAP SLP': shap_top_features_simple_nn, 
        'LIME SLP': lime_top_features_simple_nn,
        'SHAP MLP': shap_top_features_deep_nn, 
        'LIME MLP': lime_top_features_deep_nn,
        'Sal. SLP': saliency_top_features_simple_nn, 
        'DLIFT SLP': deeplift_top_features_simple_nn, 
        'IG SLP': integrated_gradients_top_features_simple_nn,
        'Sal. MLP': saliency_top_features_deep_nn, 
        'DLIFT MLP': deeplift_top_features_deep_nn, 
        'IG MLP': integrated_gradients_top_features_deep_nn
    }

    # sign consensus
    detailed_agreement_matrix = calc_sign_consensus(all_features_dict, num_samples, num_features_total)

    # ranked+unranked consensus
    ordered_comparison_matrix, unordered_comparison_matrix = calc_ranked_unranked_consensus(all_features_list, num_samples)

    
    return unordered_comparison_matrix, ordered_comparison_matrix, detailed_agreement_matrix


if __name__ == "__main__":
    all_matrices = {}
    for sel_meth in ["impurity"]: # "permution", "correlation", "selectkbest" also possible
        for multi in [False]:
            for dataset_name in ["EdgeIIoT"]: # CICIDS, CIDDS also possible
                unordered_matrix, ordered_matrix, agreement_matrix = compare_methods(dataset_name, sel_meth, 10, not multi)
                all_matrices[(sel_meth, "unordered", multi)] = unordered_matrix
                all_matrices[(sel_meth, "ordered", multi)] = ordered_matrix
                all_matrices[(sel_meth, "sign", multi)] = agreement_matrix
                gc.collect() # dunno if this helps, but cannot hurt, i guess
                joblib.dump(all_matrices, os.path.join('data','all_matrices_'+ dataset_name +'.pkl'))