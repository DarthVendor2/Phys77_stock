import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


def generate_test_regions(processed_data_file_path, num_tests = 10, max_days = 240, lower = None, upper = -1, uniform = False):
    processed_df = pd.read_csv(processed_data_file_path)[["Derivative_0"]]
    days = range(len(processed_df))
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
def clear_directory(Prediction = False, Accuracy = False, identifiers = None):
    if Prediction:
        folder = "Prediction_storage"
        if identifiers:
            ids = [f"{identifier}" for identifier in identifiers]
            folder = os.path.join(folder, *ids)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    if Accuracy:
        folder = "Accuracy_storage"
        if identifiers:
            ids = [f"{identifier}" for identifier in identifiers]
            folder = os.path.join(folder, *ids)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    
def save_prediction(prediction, model_identifiers, start_day = -1):
    root_folder = "Prediction_storage"
    identifiers = [f"{identifier}" for identifier in model_identifiers]
    folder_path = os.path.join(root_folder, *identifiers)
    os.makedirs(folder_path, exist_ok = True)
    file_path = os.path.join(folder_path, f"{start_day}.csv" )
    if os.path.exists(file_path):
                os.remove(file_path)
    df = pd.DataFrame(prediction)
    df.to_csv(file_path)

def access_prediction(model_identifiers, start_day = -1, return_days = False):    
    root_folder = "Prediction_storage"
    identifiers = [f"{identifier}" for identifier in model_identifiers]
    folder_path = os.path.join(root_folder, *identifiers)
    file_path = os.path.join(folder_path, f"{start_day}.csv")
    prediction = pd.read_csv(file_path, header=None)
    if return_days:
        days = range(start_day,start_day+len(prediction))
        return days, prediction
    else:
        return prediction

def save_accuracies(accuracies, errs, model_identifiers, accuracy_type  = "fractional_change_accuracy"):
    root_folder = "Accuracy_storage"
    identifiers = [f"{identifier}" for identifier in model_identifiers]
    folder_path = os.path.join(root_folder, *identifiers)
    os.makedirs(folder_path, exist_ok = True)
    file_path = os.path.join(folder_path, f"{accuracy_type}.csv" )
    if os.path.exists(file_path):
                os.remove(file_path)
    data = {"accuracies": accuracies, "errs": errs}
    df = pd.DataFrame(data)
    df.dropna(inplace=True)
    df.to_csv(file_path)

def access_accuracies(model_identifiers, accuracy_type = "relative_change_accuracy", return_displacement = False):
    root_folder = "Accuracy_storage"
    identifiers = [f"{identifier}" for identifier in model_identifiers]
    folder_path = os.path.join(root_folder, *identifiers)
    file = os.listdir(folder_path)
    file_path = os.path.join(folder_path, f"{accuracy_type}.csv")
    accuracies = pd.read_csv(file_path)["accuracies"]
    errs = pd.read_csv(file_path)["errs"]
    if return_displacement:
        days_displacement = range(len(accuracies))
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
            real_frac_changes = np.absolute(real - real[0]) / real[0]
            predicted_frac_changes = np.absolute(prediction - prediction[0]) / prediction[0]
            difference_in_frac_change = real_frac_changes - predicted_frac_changes
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
    
