class model_analysis:
    
    def __init__(self, raw_file_path, processed_file_path, start = 0, end = -1, interval_length = 1):
        
        self.raw_df = pd.read_csv(raw_file_path)[["Open"]]
        self.model_data_folder_path = "Model_data/"
        
    def test_region_generator(max_days = 240, number_of_tests = 10):
        days = range(len(self.raw_df))
        start_days = [days[:-region_length*n] for n in range(1,number_of_tests+1)]
        return region_starts

    def model_data_handler(self, region_starts):
        
        pass
    
    def mean_percent_accuracy(self, real, preds, start):
        real = np.array(real)
        preds = np.array(preds)
        region_differences = []
        for pred in preds:
            region_differences.append(directional_accuracy(real, pred)) #can just run on preds?
            
        mean_accuracy = np.sum(region_differences)/len(preds)
        
        return mean_accuracy

    def percent_accuracy(self, real, pred, start):   
        real_percent_change = (real - real[0]) / real[0]
        pred_percent_change = (pred - pred[0]) / pred[0]
        difference_in_percent_change = (real_percent_change - pred_percent_change) / real_percent_change   
        return difference_in_percent_change

    def value_accuracy():
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