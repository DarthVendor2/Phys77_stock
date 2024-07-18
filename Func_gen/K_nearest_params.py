#Will work on this next
#Will basically take N number of params (derivatives from our splice) and use k nearest to find M more number of derivatives to apply to our taylor
import os 

#Make sures folder file exists then gets name of files
data_folder_path= "Stock_data/Processed_data"

class get_data():
    def __init__(self, folder_path):
        self.ensure_folder(folder_path)

    def ensure_folder(self, folder_path):
        self.folder_path= folder_path
        os.makedirs(data_folder_path, exist_ok=True)
        self.files = os.listdir(data_folder_path)

    def print_files(self):
        print(self.files)

temp= get_data(data_folder_path)

temp.print_files()






