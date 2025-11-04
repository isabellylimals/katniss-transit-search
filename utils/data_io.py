
import pandas as pd

def load_csv(file_path):
        df = pd.read_csv(file_path, comment="#") 
        print(f"ok ok {file_path}")
        
        return df 
        

