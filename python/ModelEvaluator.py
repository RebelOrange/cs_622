

class ModelEvaluator:
    def __init__(self):
        pass
        
    def add_model(self, model_name: str, model):
        pass
        
    def evaluate_model(self, model_name: str, test_data, batch_size: int = 32):
        pass
    
    def calculate_metrics(self, true_labels: dict, predicted_labels: dict, class_names: dict):
        pass
    
    def compare_models(self):
        pass
    
    # will we need this?
    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        pass
    
    def plot_comparison(self, metric: str = "accuracy", save_path: str = None):
        pass

if __name__ == "__main__":
    pass