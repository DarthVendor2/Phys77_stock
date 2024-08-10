import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as math
import os
import shutil


def generate_test_regions(processed_data_file_path, num_tests = 10, max_days = 240, lower = None, upper = -1, uniform = False):
    processed_df = pd.read_csv(processed_data_file_path)[["Derivative_0"]]
    days = processed_df.index
    if lower is None:
        lower = max_days
    if uniform:
        start_days = [days[-1-max_days*n] for n in range(1,num_tests+1)]
        start_days = start_days[::-1]
    else:
        upper = days[upper]
        start_days = np.random.randint(lower, upper - max_days, num_tests)
    return start_days

#data_handling
def clear_path(ids = None):
    folder = "Model_storage"
    if ids:
        ids = [(f"{id}_{model_ids[id]}") for id in model_ids]
        folder = os.path.join(folder, *ids)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
   
def save_predictions(predictions, model_ids, start_days, singular = False, overwrite = False):
    if singular:
        predictions = np.array([[predictions]])
        start_days = np.array(start_days)
    root_folder = "Model_storage"
    ids = [(f"{id}_{model_ids[id]}") for id in model_ids]
    folder_path = os.path.join(root_folder, *ids)
    os.makedirs(folder_path, exist_ok = True)
    file_path = os.path.join(folder_path, "predictions.csv")
    if os.path.exists(file_path):
        if overwrite:
            os.remove(file_path)
            df = pd.DataFrame(np.transpose(predictions), columns = start_days)
        else:
            df = pd.read_csv(file_path)
            df[[start_days]] = pd.DataFrame(np.transpose(predictions))
    else:
        df = pd.DataFrame(np.transpose(predictions), columns = start_days)
    df.to_csv(file_path)
    
#to do: save_accuracies options, access files, fix workspace to match new structure
def access_predictions(model_ids, start_days = None, singular = False, return_days = False):    
    root_folder = "Model_storage"
    ids = [(f"{id}_{model_ids[id]}") for id in model_ids]
    folder_path = os.path.join(root_folder, *ids)
    file_path = os.path.join(folder_path, "predictions.csv")
    if singular:
        predictions = np.array(pd.read_csv(file_path, header = start_days))
        if return_days:
            days = np.array(predictions.index) + start_day
            return days, predictions
    else:
        if start_days:     
            predictions = np.array([pd.read_csv(file_path, header = start_day) for start_day in start_days])
        else:
            start_days = pd.read_csv(file_path).columns[1:]
            predictions = np.array([pd.read_csv(file_path)[start_day] for start_day in start_days])
            start_days = [int(math.floor(float(start_day))) for start_day in start_days] #corrects for some minor computer errors in the start_days
        return start_days, predictions
    return predictions

def save_accuracies(accuracies, errs, model_ids, accuracy_type  = "relative_change", overwrite = False):        
    root_folder = "Model_storage"
    ids = [(f"{id}_{model_ids[id]}") for id in model_ids]
    folder_path = os.path.join(root_folder, *ids)
    os.makedirs(folder_path, exist_ok = True)
    file_path = os.path.join(folder_path, "accuracies.csv" )
    data = {f"{accuracy_type}_accuracies": accuracies, f"{accuracy_type}_errs": errs}
    if os.path.exists(file_path):
        if overwrite:
            os.remove(file_path)
            df = pd.DataFrame(data, index = None)
        else:
            df = pd.read_csv(file_path)
            df[[f"{accuracy_type}_accuracies", f"{accuracy_type}_errs"]] = pd.DataFrame(data, index = None)
    else:
        df = pd.DataFrame(data, index = None)
    df.to_csv(file_path)

def access_accuracies(model_ids, accuracy_type = "relative_change", return_displacement = False):
    root_folder = "Model_storage"
    ids = [(f"{id}_{model_ids[id]}") for id in model_ids]
    folder_path = os.path.join(root_folder, *ids)
    file_path = os.path.join(folder_path, "accuracies.csv")
    accuracies = pd.read_csv(file_path)[f"{accuracy_type}_accuracies"]
    errs = pd.read_csv(file_path)[f"{accuracy_type}_errs"]
    if return_displacement:
        days_displacement = np.array(accuracies.index)
        return days_displacement, accuracies, errs
    return accuracies, errs

class analysis():
    def __init__(self, raw_data_file_path, predictions, starts, time_bound = None):
        self.raw_df = pd.read_csv(raw_data_file_path)["Open"]
        self.starts = np.array(starts)
        self.num_tests = len(starts)
        self.predictions = np.array(predictions)
        self.time_bound = time_bound
        self.analysis_range = len(self.predictions[0])
        if time_bound is not None:
            self.analysis_range = len(self.predictions[0,0:self.time_bound])

        
    def relative_change_accuracy(self):
        region_change_accuracies = np.empty((self.num_tests, self.analysis_range))
        for i,prediction in enumerate(self.predictions):
            start = self.starts[i]
            real = np.array(self.raw_df[start:(start+self.analysis_range)])
            real_frac_changes = (real - real[0]) / real[0]
            predicted_frac_changes = (prediction - prediction[0]) / prediction[0]
            difference_in_frac_change = np.absolute(real_frac_changes - predicted_frac_changes)
            region_change_accuracies[i, :] = difference_in_frac_change
        accuracies = np.array([np.sum(row) for row in np.transpose(region_change_accuracies)]) / self.num_tests
        errs = np.array([np.std(row) for row in np.transpose(region_change_accuracies)])
        return accuracies, errs

    def mean_relative_change_accuracy(self, single_value = False):
        accuracies, errs = self.relative_change_accuracy()
        if single_value:
            mean_accuracy = np.mean(accuracies)
            mean_err = np.mean(errs)
        else:
            locs = np.arange(1, len(accuracies)+1)
            mean_accuracy = np.cumsum(accuracies) / locs
            mean_err = np.cumsum(errs) / locs
        return mean_accuracy, mean_err
        
    def value_accuracy(self):
        region_value_accuracies = np.empty((self.num_tests, self.analysis_range))
        for i,prediction in enumerate(self.predictions):
            start = self.starts[i]
            real = np.array(self.raw_df[start:(start+self.analysis_range)])
            rel_accuracy = np.absolute(real - prediction) / real            
            region_value_accuracies[i, :] = rel_accuracy
        accuracies = np.array([np.sum(row) for row in np.transpose(region_value_accuracies)]) / self.num_tests
        errs = np.array([np.std(row) for row in np.transpose(region_value_accuracies)])
        return accuracies, errs

    def mean_value_accuracy(self, single_value = False):
        accuracies, errs = self.value_accuracy()
        if single_value:
            mean_accuracy = np.mean(accuracies)
            mean_err = np.mean(errs)
        else:
            locs = np.arange(1, len(accuracies)+1)
            mean_accuracy = np.cumsum(accuracies) / locs
            mean_err = np.cumsum(errs) / locs
        return mean_accuracy, mean_err

        
    def plot():
        pass
    


        
#data structure for prediction data
#data visualizations

#model updates?
#compute time?

#__________
#Accuracy Calculator (and criteria for accuracy) - Difference in Area, Difference at each Point/Step
#Compute time Calculator
#Data Visualization of Accuracies and Compute Times vs. M,N,K,T
#Data Visualization of Accuracies vs. Compute Times for the various inputs
#Criteria for Best Model in balancing Accuracy and Compute Time