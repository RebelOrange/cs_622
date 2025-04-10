import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from DataManager import DataManager
from EfficientNet import EfficientNet

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.confusion_matrices = {}
        self.class_names = {}
        self.metrics = {}
        self.param_mapping = {
            'optimizer': 'optimizer_name',
            'lr': 'learning_rate',
            'wd': 'l2_weight_decay',
            'loss': 'loss_function',
            'dropout': 'dropout_rate'
        }
        
    def addModel(self, model_name: str, model):
        self.models[model_name] = model
        self.results[model_name] = {}
        print(f"Added model: {model_name}")
        
    def evaluateModel(self, model_name: str, test_data, batch_size: int = 32):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Add it first with addModel()")
        
        model = self.models[model_name]
        print(f"\nEvaluating model: {model_name}")
        
        true_labels = test_data["label"].tolist()
        
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data["image"])
            elif hasattr(model, 'predictBatch'):
                predictions = model.predictBatch(test_data["image"])
            elif hasattr(model, 'evaluate'):
                accuracy = model.evaluate(test_data, batch_size)
                print(f"Model returned accuracy: {accuracy:.2f}%")
                return accuracy
            else:
                raise AttributeError("Model does not have predict, predictBatch, or evaluate method")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0
            
        accuracy = accuracy_score(true_labels, predictions) * 100.0
        
        self.results[model_name] = {
            'predictions': predictions,
            'true_labels': true_labels,
            'accuracy': accuracy
        }
        
        all_classes = sorted(set(true_labels + predictions))
        self.class_names[model_name] = {i: cls for i, cls in enumerate(all_classes)}
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        true_idx = [class_to_idx[label] for label in true_labels]
        pred_idx = [class_to_idx.get(label, -1) for label in predictions]
        self.confusion_matrices[model_name] = confusion_matrix(true_idx, pred_idx, labels=range(len(all_classes)))
        
        self.metrics[model_name] = {'accuracy': accuracy}
        
        print(f"\nEvaluation completed for {model_name}: Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def compareModels(self):
        if not self.metrics:
            print("\nNo models evaluated yet. Use evaluateModel() first.")
            return None
            
        metrics_data = [{'Model': name, 'Accuracy (%)': metrics['accuracy']} for name, metrics in self.metrics.items()]
        comparison_df = pd.DataFrame(metrics_data)
        
        print("\nModel Comparison:")
        print(comparison_df)
        
        plt.figure(figsize=(15, 10))
        values = [self.metrics[model]['accuracy'] for model in self.metrics]
        bars = plt.bar(list(self.metrics.keys()), values, color='skyblue')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', va='bottom')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def plotConfusionMatrix(self, model_name: str, save_path: str = None):
        if (model_name not in self.confusion_matrices or model_name not in self.class_names):
            print(f"No confusion matrix available for model '{model_name}'. Run evaluateModel() first.")
            return
            
        cm = self.confusion_matrices[model_name]
        class_names = list(self.class_names[model_name].values())
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
            
        plt.show()
    
    def splitDataset(self, full_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        print(f"\nSplitting dataset with ratio - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")
        
        if test_ratio > 0:
            rest_data, test_data = train_test_split(
                full_dataset, test_size=test_ratio, random_state=seed, 
                stratify=full_dataset["label"] if "label" in full_dataset.columns else None
            )
            test_data = test_data.reset_index(drop=True)
        else:
            rest_data = full_dataset
            test_data = pd.DataFrame()
        
        if val_ratio > 0:
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            train_data, val_data = train_test_split(
                rest_data, test_size=val_ratio_adjusted, random_state=seed, 
                stratify=rest_data["label"] if "label" in rest_data.columns else None
            )
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
        else:
            train_data = rest_data.reset_index(drop=True)
            val_data = pd.DataFrame()
        
        print(f"Dataset split completed - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} samples")
        return train_data, val_data, test_data
    
    def evaluateModelVariants(self, model_class, model_name_prefix: str, training_data, 
                            num_classes: int = None, batch_size: int = 32, num_epochs: int = 5,
                            train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                            max_configs: int = 9, model_dir: str = "models", 
                            save_best: bool = False, load_best: bool = False):
        train_df, val_df, test_df = self.splitDataset(
            training_data, train_ratio, val_ratio, test_ratio
        )
        
        if num_classes is None and "label" in training_data.columns:
            num_classes = len(training_data["label"].unique())
            print(f"Inferred {num_classes} classes from dataset")
        
        base_model = model_class(num_classes=num_classes, model_dir=model_dir)
        print(f"\nGenerating configurations for {model_name_prefix}")
        print(f"\nFinding best configuration from up to {max_configs} configurations...")
        
        try:
            # this code kinda weird, if you passed validation_data then set validation_split to 0
            # if not passed, then set validation_split to value 
            if hasattr(base_model, 'findBestConfig'):
                best_config, configs = base_model.findBestConfig(
                    df=train_df,
                    validation_split=0,  # pass 0 in case we already split the data
                    epochs=2,
                    batch_size=batch_size,
                    max_configs=max_configs,
                    validation_data=val_df  # Pass the validation data directly if we already split
                )
            else:
                raise AttributeError(f"{model_name_prefix} does not have findBestConfig method")
        except Exception as e:
            print(f"\nError finding configurations: {e}")
            return pd.DataFrame()
        
        if not configs:
            print("\nNo configurations were generated")
            return pd.DataFrame()
            
        config_keys = list(configs[0].keys())
        print(f"\nEvaluating {len(configs)} configurations with parameters: {config_keys}")
        
        result_keys = [self.param_mapping.get(key, key) for key in config_keys]
        results = []
        
        for i, config in enumerate(configs):
            config_name = f"{model_name_prefix}_{i+1}"
            print(f"\n{'-'*60}\nEvaluating configuration {i+1}/{len(configs)}: {config}")
            
            try:
                mapped_config = {self.param_mapping.get(k, k): v for k, v in config.items()}
                
                model = model_class(num_classes=num_classes, model_dir=model_dir, **mapped_config)
                self.addModel(config_name, model)
                
                if hasattr(model, 'train'):
                    model.train(train_df, epochs=num_epochs, batch_size=batch_size, save_interval=num_epochs, save_best=save_best, load_best=load_best)
                else:
                    print(f"\nWarning: {config_name} does not have a train method, skipping training")
                
                # Evaluate test set !!! this is important -> use test and npot train
                accuracy = self.evaluateModel(config_name, test_df, batch_size)
                
                result_row = {'Model': config_name, 'Accuracy': accuracy}
                for orig_key, value in config.items():
                    mapped_key = self.param_mapping.get(orig_key, orig_key)
                    result_row[mapped_key] = value
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error with configuration {i+1}: {str(e)}")
        
        result_matrix = pd.DataFrame(results)
        if not result_matrix.empty:
            param_columns = []
            for key in config_keys:
                mapped_key = self.param_mapping.get(key, key)
                if mapped_key in result_matrix.columns:
                    param_columns.append(mapped_key)
                    
            cols = ['Model'] + param_columns + ['Accuracy']
            
            result_matrix = result_matrix[cols]
            
            self.plotConfigMatrix(result_matrix, model_name_prefix)
            
            if len(result_matrix) > 0:
                best_row = result_matrix.loc[result_matrix['Accuracy'].idxmax()]
                best_idx = int(best_row['Model'].split('_')[-1]) - 1
                print(f"\nBest configuration ({best_row['Model']}) with accuracy {best_row['Accuracy']:.2f}%:")
                for key in result_keys:
                    if key in best_row and pd.notna(best_row[key]):
                        print(f"  {key}: {best_row[key]}")
                
                if best_config:
                    print(f"\nOriginal best config from validation: {best_config}")
                    print(f"\nBest config from test data: {configs[best_idx] if best_idx < len(configs) else 'Unknown'}")
        
        return result_matrix
    
    def plotConfigMatrix(self, result_matrix, model_type, save_path=None):
        if result_matrix.empty:
            print("\nNo results to plot")
            return
        
        x_column = next((col for col in ['optimizer_name', 'optimizer'] if col in result_matrix.columns), None)
        y_column = next((col for col in ['learning_rate', 'lr'] if col in result_matrix.columns), None)
        
        if not x_column or not y_column:
            self.basePlot(result_matrix, model_type)
            return
            
        try:
            pivot = pd.pivot_table(
                result_matrix, values='Accuracy', 
                index=y_column, columns=x_column, aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                pivot, annot=True, fmt='.1f', cmap='crest',
                cbar_kws={'label': 'Accuracy (%)'}
            )
            
            plt.title(f'\n{model_type} Performance by Optimizer and Learning Rate')
            plt.tight_layout()
            
        except Exception as e:
            print(f"\nError creating pivot table: {e}")
            self.basePlot(result_matrix, model_type)
            return
            
        if save_path:
            plt.savefig(save_path)
            print(f"\nConfiguration matrix saved to {save_path}")
            
        plt.show()

    def basePlot(self, result_matrix, model_type="Model"):
        plt.figure(figsize=(10, 6))
        bars = plt.bar(result_matrix['Model'], result_matrix['Accuracy'], color='skyblue')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', va='bottom')
            
        plt.title(f'{model_type} Model Results')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dm = DataManager()
    current_folder = os.getcwd()
    
    print("Loading training data...")
    dm.LoadTrainingData(folderName=current_folder+"/../data/", csvFileName="Training_set.csv", numFiles=100)
    dm.RemoveMissingData()
    
    num_classes = len(dm.TrainingData["label"].unique())
    model_dir = os.path.join(current_folder, "../models")
    evaluator = ModelEvaluator()
    
    print("\nEvaluating EfficientNet model variants...")
    result_matrix = evaluator.evaluateModelVariants(
        model_class=EfficientNet,
        model_name_prefix="EfficientNet",
        training_data=dm.TrainingData,
        num_classes=num_classes,
        batch_size=8,
        num_epochs=3,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        max_configs=2,
        model_dir=model_dir,
        save_best=False, 
        load_best=False   
    )
    
    print("\nEvaluation Result Matrix:")
    print(result_matrix)