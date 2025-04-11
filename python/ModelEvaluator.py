import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
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
            raise ValueError(f"Model '{model_name}' not found. Add it with addModel() before run this")
        
        model = self.models[model_name]
        print(f"\nEvaluating model: {model_name}")
        
        true_labels = test_data["label"].tolist()
        
        predictions = None
        probabilities = None
        
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data["image"], return_proba=False)
                probabilities = model.predict(test_data["image"], return_proba=True)
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
            'accuracy': accuracy,
            'probabilities': probabilities
        }
        
        all_classes = sorted(set(true_labels + predictions))
        self.class_names[model_name] = {i: cls for i, cls in enumerate(all_classes)}
        # do not remove these or it will break code as we have custom dataframe
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        true_idx = [class_to_idx[label] for label in true_labels]
        pred_idx = [class_to_idx.get(label, -1) for label in predictions]

        # The issue might be here - Let's normalize and make sure classes are handled correctly
        self.confusion_matrices[model_name] = confusion_matrix(true_idx, pred_idx, labels=range(len(all_classes)))
        
        self.metrics[model_name] = {'accuracy': accuracy}
        
        print(f"\nEvaluation done for {model_name}: Accuracy: {accuracy:.2f}%")
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
    
    def plotConfusionMatrix(self, model_name: str, save_path: str = None, save_csv: bool = False):
        if (model_name not in self.confusion_matrices or model_name not in self.class_names):
            print(f"No confusion matrix available for model '{model_name}'. Run evaluateModel() first.")
            return
            
        cm = self.confusion_matrices[model_name]
        class_names = list(self.class_names[model_name].values())
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized) 
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix for {model_name} (counts)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            img_path = save_path
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = f"{save_path}.png"
            plt.savefig(img_path, bbox_inches='tight')
            print(f"Confusion matrix saved to {img_path}")
            
            norm_path = os.path.splitext(save_path)[0] + "_normalized.png"
            
            if save_csv:
                csv_path = os.path.splitext(save_path)[0] + "_cm.csv"
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                cm_df.to_csv(csv_path)
                print(f"Confusion matrix data saved to {csv_path}")
                
                norm_csv_path = os.path.splitext(save_path)[0] + "_cm_normalized.csv"
                cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
                cm_norm_df.to_csv(norm_csv_path)
                print(f"Normalized confusion matrix data saved to {norm_csv_path}")
        
        plt.show()
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Normalized Confusion Matrix for {model_name} (percentages)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            norm_path = os.path.splitext(save_path)[0] + "_normalized.png"
            plt.savefig(norm_path, bbox_inches='tight')
            print(f"Normalized confusion matrix saved to {norm_path}")
            
        plt.show()
        
        return cm, cm_normalized
    
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
    
    """
        Train and evaluate multiple configurations of a model.
        This should handles the full pipeline: splitting data, finding best configurations,
        training models, and evaluating their performance.
    """
    def evaluateModelVariants(self, model_class, model_name_prefix: str, training_data, 
                            num_classes: int = None, batch_size: int = 32, num_epochs: int = 5,
                            train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                            max_configs: int = 9, model_dir: str = "models", 
                            save_best: bool = False, load_best: bool = False,
                            save_csv: bool = False, save_path: str = None):
        
        train_df, val_df, test_df = self.splitDataset(
            training_data, train_ratio, val_ratio, test_ratio
        )
        
        if num_classes is None and "label" in training_data.columns:
            num_classes = len(training_data["label"].unique())
            print(f"Detected {num_classes} classes in dataset")
        
        base_model = model_class(num_classes=num_classes, model_dir=model_dir)
        print(f"\nFinding up to {max_configs} configurations for {model_name_prefix}...")
        
        try:
            if not hasattr(base_model, 'findBestConfig'):
                raise AttributeError(f"{model_name_prefix} does not have findBestConfig method")
            
            # Get candidate configurations -> this code kinda weird as we need to set 0 for valid split 
            # if we have val_df and set non-0 if we have no val_df
            # so careful when use this
            best_config, configs = base_model.findBestConfig(
                df=train_df,
                validation_split=0,
                epochs=3, 
                batch_size=batch_size,
                max_configs=max_configs,
                validation_data=val_df
            )
        except Exception as e:
            print(f"\nError finding configurations: {e}")
            return pd.DataFrame()
        
        if not configs:
            print("\nNo configurations were generated")
            return pd.DataFrame()
            
        config_keys = list(configs[0].keys())
        result_keys = [self.param_mapping.get(key, key) for key in config_keys]
        results = []
        
        print(f"\nTraining and evaluating {len(configs)} configurations with parameters: {config_keys}")
        
        # Train + evaluate each configuration
        for i, config in enumerate(configs):
            config_name = f"{model_name_prefix}_{i+1}"
            print(f"\n{'-'*60}\nConfiguration {i+1}/{len(configs)}: {config}")
            
            try:
                mapped_config = {self.param_mapping.get(k, k): v for k, v in config.items()}
                
                model = model_class(num_classes=num_classes, model_dir=model_dir, **mapped_config)
                self.addModel(config_name, model)
                
                # Train 
                if hasattr(model, 'train'):
                    print(f"\nTraining {config_name}...")
                    model.train(train_df, epochs=num_epochs, batch_size=batch_size, 
                                save_interval=num_epochs, save_best=save_best, load_best=load_best)
                else:
                    print(f"\nWarning: {config_name} does not have a train method, using pre-trained model")
                
                # Evaluate
                print(f"\nEvaluating {config_name} on test data...")
                accuracy = self.evaluateModel(config_name, test_df, batch_size)
                
                # Append results
                result_row = {'Model': config_name, 'Accuracy': accuracy}
                for orig_key, value in config.items():
                    mapped_key = self.param_mapping.get(orig_key, orig_key)
                    result_row[mapped_key] = value
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error with configuration {i+1}: {str(e)}")
        
        # resul + matrix
        result_matrix = pd.DataFrame(results)
        if not result_matrix.empty:
            param_columns = [self.param_mapping.get(key, key) for key in config_keys 
                            if self.param_mapping.get(key, key) in result_matrix.columns]
            cols = ['Model'] + param_columns + ['Accuracy']
            result_matrix = result_matrix[cols]
            
            self.plotConfigMatrix(result_matrix, model_name_prefix, save_path, save_csv)
            
            if len(result_matrix) > 0:
                best_row = result_matrix.loc[result_matrix['Accuracy'].idxmax()]
                best_idx = int(best_row['Model'].split('_')[-1]) - 1
                
                print(f"\n Best configuration ({best_row['Model']}) achieved {best_row['Accuracy']:.2f}% accuracy:")
                for key in result_keys:
                    if key in best_row and pd.notna(best_row[key]):
                        print(f"  {key}: {best_row[key]}")
                
                if best_config:
                    print(f"\nBest validation config: {best_config}")
                    print(f"Best test config: {configs[best_idx] if best_idx < len(configs) else 'Unknown'}")
        
        return result_matrix
    
    def plotConfigMatrix(self, result_matrix, model_type, save_path=None, save_csv=False):
        if result_matrix.empty:
            print("\nNo results to plot")
            return
        
        x_column = next((col for col in ['optimizer_name', 'optimizer'] if col in result_matrix.columns), None)
        y_column = next((col for col in ['learning_rate', 'lr'] if col in result_matrix.columns), None)
        
        if not x_column or not y_column:
            self.basePlot(result_matrix, model_type, save_path, save_csv)
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
            self.basePlot(result_matrix, model_type, save_path, save_csv)
            return
            
        if save_path:
            img_path = save_path
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = f"{save_path}.png"
            plt.savefig(img_path, bbox_inches='tight')
            print(f"\nConfiguration matrix saved to {img_path}")
            
            if save_csv:
                csv_path = os.path.splitext(save_path)[0] + "_matrix.csv"
                pivot.to_csv(csv_path)
                print(f"Configuration matrix data saved to {csv_path}")
                
                # Also save the full result matrix with all parameters
                full_csv_path = os.path.splitext(save_path)[0] + "_full_results.csv"
                result_matrix.to_csv(full_csv_path, index=False)
                print(f"Full configuration results saved to {full_csv_path}")
            
        plt.show()

    def basePlot(self, result_matrix, model_type="Model", save_path=None, save_csv=False):
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
        
        if save_path:
            img_path = save_path
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = f"{save_path}.png"
            plt.savefig(img_path, bbox_inches='tight')
            print(f"\nConfiguration results plot saved to {img_path}")
            
            if save_csv:
                csv_path = os.path.splitext(save_path)[0] + "_results.csv"
                result_matrix.to_csv(csv_path, index=False)
                print(f"Configuration results data saved to {csv_path}")
                
        plt.show()
    
    def plotROCCurve(self, model_name: str, save_path: str = None, save_csv: bool = False):
        if model_name not in self.results or self.results[model_name].get('probabilities') is None:
            print(f"\nNo probability data available for model '{model_name}'")
            return
            
        probs = self.results[model_name]['probabilities']
        y_true = self.results[model_name]['true_labels']
        classes = list(self.class_names[model_name].values())
        
        # dont rm this as it is important
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        y_true_idx = np.array([class_to_idx[label] for label in y_true])
        y_true_bin = label_binarize(y_true_idx, classes=range(len(classes)))
        
        # Handle shape mismatch -> dont rm
        n_classes = min(len(classes), probs.shape[1])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        roc_data = {'class': [], 'fpr': [], 'tpr': [], 'auc': []}
        
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
                
                roc_data['class'].append(classes[i])
                roc_data['fpr'].append(fpr)
                roc_data['tpr'].append(tpr)
                roc_data['auc'].append(roc_auc)
                
            except Exception as e:
                print(f"\nSkipping class {classes[i]}: {e}")
        
        try:
            y_bin_trim = y_true_bin[:, :n_classes]
            probs_trim = probs[:, :n_classes]
            
            macro_auc = roc_auc_score(y_bin_trim, probs_trim, multi_class='ovr', average='macro')
            micro_auc = roc_auc_score(y_bin_trim, probs_trim, multi_class='ovr', average='micro')
            
            self.metrics[model_name].update({'macro_auc': macro_auc, 'micro_auc': micro_auc})
            print(f"\nMacro-AUC: {macro_auc:.4f}, Micro-AUC: {micro_auc:.4f}")
        except Exception as e:
            print(f"\nError computing average metrics: {e}")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlim=[0, 1], ylim=[0, 1.05], title=f'ROC Curves for {model_name}',
                xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend(loc="best", fontsize='small', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        
        if save_path:
            img_path = save_path
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = f"{save_path}.png"
            plt.savefig(img_path, bbox_inches='tight')
            print(f"ROC curve saved to {img_path}")
            
            if save_csv:
                csv_path = os.path.splitext(save_path)[0] + "_roc.csv"
                
                csv_data = []
                for i, cls in enumerate(roc_data['class']):
                    fpr = roc_data['fpr'][i]
                    tpr = roc_data['tpr'][i]
                    auc_value = roc_data['auc'][i]
                    
                    df = pd.DataFrame({
                        'Class': cls,
                        'FPR': fpr,
                        'TPR': tpr,
                        'AUC': auc_value
                    })
                    csv_data.append(df)
                
                if csv_data:
                    final_df = pd.concat(csv_data, ignore_index=True)
                    final_df.to_csv(csv_path, index=False)
                    print(f"ROC data saved to {csv_path}")
        
        plt.show()
        
        return roc_data
    
    def computeModelROC(self, model_name):
        probs = self.results[model_name]['probabilities']
        true_labels = self.results[model_name]['true_labels']
        classes = list(self.class_names[model_name].values())
        
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        # 1= correct, 0=incorrect
        binary_correct = []
        confidence_scores = []
        
        for i, (prob, true) in enumerate(zip(probs, true_labels)):
            if i < len(true_labels) and true in class_to_idx:
                pred_idx = np.argmax(prob)
                true_idx = class_to_idx[true]
                binary_correct.append(1 if pred_idx == true_idx else 0)
                confidence_scores.append(np.max(prob))
        
        fpr, tpr, _ = roc_curve(binary_correct, confidence_scores)
        roc_auc = auc(fpr, tpr)
        
        return fpr, tpr, roc_auc
    
    def compareModelTypeVariants(self, model_type, save_path=None, save_csv=False):
        variants = [m for m in self.results.keys() 
                        if m.startswith(f"{model_type}_") and 
                        self.results[m].get('probabilities') is not None]
        
        if not variants:
            print(f"No {model_type} variants with probability data found.")
            return []
        
        plt.figure(figsize=(10, 8))
        
        compare_data = {'model': [], 'fpr': [], 'tpr': [], 'auc': []}
        
        for model in variants:
            try:
                fpr, tpr, roc_auc = self.computeModelROC(model)
                plt.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.2f})')
                
                compare_data['model'].append(model)
                compare_data['fpr'].append(fpr)
                compare_data['tpr'].append(tpr)
                compare_data['auc'].append(roc_auc)
                
            except Exception as e:
                print(f"Error with {model}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Comparison of {model_type} Variants')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            img_path = save_path
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = f"{save_path}.png"
            plt.savefig(img_path, bbox_inches='tight')
            print(f"ROC comparison saved to {img_path}")
            
            if save_csv and compare_data['model']:
                csv_path = os.path.splitext(save_path)[0] + "_compare_roc.csv"
                
                csv_data = []
                for i, model in enumerate(compare_data['model']):
                    fpr = compare_data['fpr'][i]
                    tpr = compare_data['tpr'][i]
                    auc_value = compare_data['auc'][i]
                    
                    df = pd.DataFrame({
                        'Model': model,
                        'FPR': fpr,
                        'TPR': tpr,
                        'AUC': auc_value
                    })
                    csv_data.append(df)
                
                if csv_data:
                    final_df = pd.concat(csv_data, ignore_index=True)
                    final_df.to_csv(csv_path, index=False)
                    print(f"ROC comparison data saved to {csv_path}")
                    
                summary_csv = os.path.splitext(save_path)[0] + "_summary.csv"
                summary_df = pd.DataFrame({
                    'Model': compare_data['model'],
                    'AUC': compare_data['auc']
                })
                summary_df.to_csv(summary_csv, index=False)
                print(f"Summary statistics saved to {summary_csv}")
        
        plt.show()
        
        return variants
    
    def compareBestAcrossModelTypes(self, model_types=None, save_path=None, save_csv=False):
        if not model_types:
            model_types = set()
            for model in self.results.keys():
                if '_' in model and self.results[model].get('probabilities') is not None:
                    model_types.add(model.split('_')[0])
        
        if not model_types:
            print("No model types detected with probability data.")
            return []
        
        # We will plot best variants of each model type only
        # maybe we can change? if need
        best_models = []
        for model_type in model_types:
            variants = [m for m in self.results.keys() 
                            if m.startswith(f"{model_type}_") and 
                            self.results[m].get('probabilities') is not None]
            
            if variants:
                best = max(variants, key=lambda v: self.metrics.get(v, {}).get('accuracy', -1))
                best_models.append(best)
                print(f"Best {model_type}: {best} (Accuracy: {self.metrics.get(best, {}).get('accuracy', -1):.2f}%)")
        
        if not best_models:
            return []
        
        plt.figure(figsize=(10, 8))
        
        best_data = {'model': [], 'fpr': [], 'tpr': [], 'auc': [], 'accuracy': []}
        
        for model in best_models:
            try:
                fpr, tpr, roc_auc = self.computeModelROC(model)
                plt.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.2f})')
                
                # Store data for CSV export
                best_data['model'].append(model)
                best_data['fpr'].append(fpr)
                best_data['tpr'].append(tpr)
                best_data['auc'].append(roc_auc)
                best_data['accuracy'].append(self.metrics.get(model, {}).get('accuracy', -1))
                
            except Exception as e:
                print(f"Error with {model}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Comparison of Best Model Variants')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if save_path:
            img_path = save_path
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = f"{save_path}.png"
            plt.savefig(img_path, bbox_inches='tight')
            print(f"Best models ROC comparison saved to {img_path}")
            
            if save_csv and best_data['model']:
                csv_path = os.path.splitext(save_path)[0] + "_best_models_roc.csv"
                
                csv_data = []
                for i, model in enumerate(best_data['model']):
                    fpr = best_data['fpr'][i]
                    tpr = best_data['tpr'][i]
                    auc_value = best_data['auc'][i]
                    
                    df = pd.DataFrame({
                        'Model': model,
                        'FPR': fpr,
                        'TPR': tpr,
                        'AUC': auc_value
                    })
                    csv_data.append(df)
                
                if csv_data:
                    final_df = pd.concat(csv_data, ignore_index=True)
                    final_df.to_csv(csv_path, index=False)
                    print(f"Best models ROC data saved to {csv_path}")
                
                summary_csv = os.path.splitext(save_path)[0] + "_best_models_summary.csv"
                summary_df = pd.DataFrame({
                    'Model': best_data['model'],
                    'AUC': best_data['auc'],
                    'Accuracy(%)': best_data['accuracy']
                })
                summary_df.to_csv(summary_csv, index=False)
                print(f"Best models summary statistics saved to {summary_csv}")
        
        plt.show()
        
        return best_models

    def compareBestModelVariants(self, save_path=None, save_csv=False):
        return self.compareBestAcrossModelTypes(save_path=save_path, save_csv=save_csv)

    def exportAllConfusionMatrices(self, output_dir, save_csv=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        if not self.confusion_matrices:
            print("No confusion matrices available to export. Run evaluateModel() first.")
            return
            
        for model_name in self.confusion_matrices.keys():
            if model_name in self.class_names:
                save_path = os.path.join(output_dir, f"{model_name}_confusion_matrix")
                self.plotConfusionMatrix(model_name, save_path=save_path, save_csv=save_csv)
                
        print(f"All confusion matrices exported to {output_dir}")
        
    def exportTestResults(self, output_path):
        if not self.metrics:
            print("No model metrics available. Run evaluateModel() first.")
            return
            
        metrics_data = []
        for model_name, metric_dict in self.metrics.items():
            row = {'Model': model_name}
            row.update(metric_dict)
            metrics_data.append(row)
            
        result_df = pd.DataFrame(metrics_data)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        result_df.to_csv(output_path, index=False)
        print(f"Test results exported to {output_path}")
        return result_df

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Test 1: ModelEvaluator")
    print("="*70)
    
    dm = DataManager()
    current_folder = os.getcwd()
    data_folder = os.path.join(current_folder, "../data/")
    model_dir = os.path.join(current_folder, "../models")
    os.makedirs(model_dir, exist_ok=True)
    # dict for csv
    output_dir = os.path.join(current_folder, "../output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading training data...")
    dm.LoadTrainingData(folderName=data_folder, csvFileName="Training_set.csv", numFiles=100)
    dm.RemoveMissingData()
    
    num_classes = len(dm.TrainingData["label"].unique())
    print(f"Loaded {len(dm.TrainingData)} samples with {num_classes} classes")
    
    evaluator = ModelEvaluator()
    
    print("\n" + "="*70)
    print("Test 2: Evaluating EfficientNet model variants")
    print("="*70)
    
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
        load_best=False,
        save_path=output_dir,
        save_csv=True,
    )
    
    print("\nTest 3: Evaluation Result Matrix:")
    print(result_matrix)
    
    print("\n" + "="*70)
    print("Test 4: Analyzing model performance")
    print("="*70)
    
    # ROC curves for the best model we chose
    if result_matrix is not None and not result_matrix.empty:
        best_model = result_matrix.loc[result_matrix['Accuracy'].idxmax(), 'Model']
        print(f"\nPlotting ROC curve for best model: {best_model}")
        evaluator.plotROCCurve(best_model, save_path=output_dir, save_csv=True)
        
        # Plot confusion matrix for the best model
        print(f"\nPlotting confusion matrix for best model: {best_model}")
        evaluator.plotConfusionMatrix(best_model, save_path=output_dir, save_csv=True)
    # Compare all model variants using ROC curves
    if len(evaluator.models) > 1:
        # Compare all variants of EfficientNet
        print("\nComparing all EfficientNet variants:")
        evaluator.compareModelTypeVariants("EfficientNet", save_path=output_dir, save_csv=True)
        
        # Automatic detection of model types for comparison
        print("\nComparing best variant of each detected model type:")
        evaluator.compareBestAcrossModelTypes(save_path=output_dir, save_csv=True)
        
        # compare ResNet variants (todo)
        
        # compare best variants across different model types (todo)
        
    
    # test all export 
    print("\n" + "="*70)
    print("Test 5: Exporting confusion matrices and test results")
    print("="*70)
    evaluator.exportAllConfusionMatrices(output_dir, save_csv=True)
    evaluator.exportTestResults(os.path.join(output_dir, "test_results.csv"))

    print("\n" + "="*70)
    print("ModelEvaluator tests completed")
    print("="*70)