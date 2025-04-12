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
    
    ################# ADD MODEL #################################################
    def addModel(self, name, model):
        self.models[name] = model
        self.results[name] = {}
        print(f"\nAdded model: {name}")
        
    ################ EVAL ################################################################
    def evaluateModel(self, name, test_data, batch_size=32):
        if name not in self.models:
            raise ValueError(f"\nModel '{name}' not found")
        
        model = self.models[name]
        print(f"\nEvaluating model: {name}")
        true_labels = test_data["label"].tolist()
        
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data["image"], return_proba=False)
                probabilities = model.predict(test_data["image"], return_proba=True)
            elif hasattr(model, 'predictBatch'):
                predictions = model.predictBatch(test_data["image"])
                probabilities = model.predictBatch(test_data["image"], return_proba=True)
            elif hasattr(model, 'evaluate'):
                accuracy = model.evaluate(test_data, batch_size)
                print(f"\nModel returned accuracy: {accuracy:.2f}%")
                return accuracy
            else:
                raise AttributeError("\nModel has no prediction method")
                
        except Exception as e:
            print(f"\nError during prediction: {e}")
            return 0
            
        accuracy = accuracy_score(true_labels, predictions) * 100.0
        
        self.results[name] = {
            'predictions': predictions,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'probabilities': probabilities
        }
        
        all_classes = sorted(set(true_labels + predictions))
        self.class_names[name] = {i: cls for i, cls in enumerate(all_classes)}
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        true_idx = [class_to_idx[label] for label in true_labels]
        pred_idx = [class_to_idx.get(label, -1) for label in predictions]

        self.confusion_matrices[name] = confusion_matrix(true_idx, pred_idx, labels=range(len(all_classes)))
        self.metrics[name] = {'accuracy': accuracy}
        
        if probabilities is not None and len(all_classes) > 1:
            try:
                y_true_idx = np.array([class_to_idx[label] for label in true_labels])
                y_true_bin = label_binarize(y_true_idx, classes=range(len(all_classes)))
                
                n_classes = min(len(all_classes), probabilities.shape[1])
                y_bin = y_true_bin[:, :n_classes]
                probs = probabilities[:, :n_classes]
                
                macro_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
                micro_auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='micro')
                
                self.metrics[name].update({
                    'macro_auc': macro_auc,
                    'micro_auc': micro_auc
                })
            except Exception as e:
                self.metrics[name].update({
                    'macro_auc': None,
                    'micro_auc': None
                })
        
        print(f"\nAccuracy: {accuracy:.2f}%")
        return accuracy
    
    def evaluateModelVariants(self, model_class, prefix, training_data, 
                            num_classes=None, batch_size=32, num_epochs=5,
                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                            max_configs=9, model_dir="models", 
                            save_best=False, load_best=False,
                            save_path=None, save_csv=False):
        
        train_df, val_df, test_df = self.splitDataset(
            training_data, train_ratio, val_ratio, test_ratio
        )
        
        if num_classes is None and "label" in training_data.columns:
            num_classes = len(training_data["label"].unique())
        
        base_model = model_class(num_classes=num_classes, model_dir=model_dir)
        
        try:
            if not hasattr(base_model, 'findBestConfig'):
                raise AttributeError(f"{prefix} missing findBestConfig method")
            
            best_config, configs = base_model.findBestConfig(
                df=train_df, validation_split=0, epochs=3, 
                batch_size=batch_size, max_configs=max_configs,
                validation_data=val_df
            )
        except Exception as e:
            print(f"\nConfig error: {e}")
            return pd.DataFrame()
        
        if not configs:
            return pd.DataFrame()
            
        results = []
        
        for i, config in enumerate(configs):
            config_name = f"{prefix}_{i+1}"
            
            try:
                mapped_config = {self.param_mapping.get(k, k): v for k, v in config.items()}
                
                model = model_class(num_classes=num_classes, model_dir=model_dir, **mapped_config)
                self.addModel(config_name, model)
                
                if hasattr(model, 'train'):
                    model.train(train_df, epochs=num_epochs, batch_size=batch_size, 
                                save_interval=num_epochs, save_best=save_best, load_best=load_best)
                
                accuracy = self.evaluateModel(config_name, test_df, batch_size)
                
                result_row = {'Model': config_name, 'Accuracy': accuracy}
                for orig_key, value in config.items():
                    mapped_key = self.param_mapping.get(orig_key, orig_key)
                    result_row[mapped_key] = value
                
                results.append(result_row)
                
            except Exception as e:
                print(f"\nError with config {i+1}: {str(e)}")
        
        result_matrix = pd.DataFrame(results)
        if not result_matrix.empty:
            self.plotConfigMatrix(result_matrix, prefix, save_path, save_csv)
        
        return result_matrix
    
    ################ UTILITIES #################################################
    def splitDataset(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        if test_ratio > 0:
            rest_data, test_data = train_test_split(
                dataset, test_size=test_ratio, random_state=seed, 
                stratify=dataset["label"] if "label" in dataset.columns else None
            )
            test_data = test_data.reset_index(drop=True)
        else:
            rest_data = dataset
            test_data = pd.DataFrame()
        
        if val_ratio > 0:
            val_ratio_adj = val_ratio / (train_ratio + val_ratio)
            train_data, val_data = train_test_split(
                rest_data, test_size=val_ratio_adj, random_state=seed, 
                stratify=rest_data["label"] if "label" in rest_data.columns else None
            )
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
        else:
            train_data = rest_data.reset_index(drop=True)
            val_data = pd.DataFrame()
        
        print(f"\nDataset split: Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    ############## PLOT + SAVE ################################################################
    def savePlot(self, fig, path, suffix=""):
        if not path:
            return
            
        img_path = path
        if suffix:
            base, ext = os.path.splitext(path)
            img_path = f"{base}{suffix}{ext}" if ext else f"{base}{suffix}.png"
        elif not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = f"{path}.png"
            
        fig.savefig(img_path, bbox_inches='tight')
        return img_path
    
    def saveDfToCSv(self, df, path, suffix="", index=False):
        if not path or df.empty:
            return
            
        base, ext = os.path.splitext(path)
        csv_path = f"{base}{suffix}.csv"
        
        df.to_csv(csv_path, index=index)
        return csv_path
    
    ################### CONFUSION MATRIX #################################################
    def plotConfusionMatrix(self, name, path=None, save_csv=False):
        if name not in self.confusion_matrices or name not in self.class_names:
            return
            
        cm = self.confusion_matrices[name]
        class_names = list(self.class_names[name].values())
        
        row_sums = cm.sum(axis=1)
        safe_row_sums = np.where(row_sums > 0, row_sums, 1)
        cm_norm = cm.astype('float') / safe_row_sums[:, np.newaxis]
        
        cm_norm = np.nan_to_num(cm_norm)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Normalized Confusion Matrix for {name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        
        if path:
            self.savePlot(fig, path)
            
            if save_csv:
                norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
                self.saveDfToCSv(norm_df, path, "_normalized", index=True)
        
        plt.show()
        return cm, cm_norm
    
    ################### CONFIG MATRIX #################################################
    def plotConfigMatrix(self, matrix, model_type, path=None, save_csv=False):
        if matrix.empty:
            return
        
        x_column = next((col for col in ['optimizer_name', 'optimizer'] if col in matrix.columns), None)
        y_column = next((col for col in ['learning_rate', 'lr'] if col in matrix.columns), None)
        
        if x_column and y_column:
            try:
                pivot = pd.pivot_table(
                    matrix, values='Accuracy', 
                    index=y_column, columns=x_column, aggfunc='mean'
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    pivot, annot=True, fmt='.1f', cmap='crest',
                    cbar_kws={'label': 'Accuracy (%)'}, ax=ax
                )
                
                ax.set_title(f'{model_type} Performance by Optimizer and Learning Rate')
                plt.tight_layout()
                
                if path:
                    self.savePlot(fig, path)
                    if save_csv:
                        self.saveDfToCSv(matrix, path, "_full_results")
                
                plt.show()
                return
            except:
                pass
        
        self.basePlot(matrix, model_type, path, save_csv)
    
    def basePlot(self, matrix, model_type="Model", path=None, save_csv=False):
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(matrix['Model'], matrix['Accuracy'], color='skyblue')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.2f}%', ha='center', va='bottom')
            
        ax.set_title(f'{model_type} Model Results')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if path:
            self.savePlot(fig, path)
            if save_csv:
                self.saveDfToCSv(matrix, path, "_results")
                
        plt.show()
    
    ################## ROC CURVE #################################################
    def plotROCCurve(self, name, path=None, save_csv=False):
        if name not in self.results or self.results[name].get('probabilities') is None:
            return
            
        probs = self.results[name]['probabilities']
        y_true = self.results[name]['true_labels']
        classes = list(self.class_names[name].values())
        
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        y_true_idx = np.array([class_to_idx[label] for label in y_true])
        y_true_bin = label_binarize(y_true_idx, classes=range(len(classes)))
        
        n_classes = min(len(classes), probs.shape[1])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        roc_data = {'class': [], 'fpr': [], 'tpr': [], 'auc': []}
        
        for i in range(n_classes):
            try:
                if np.sum(y_true_bin[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=1.5, alpha=0.8, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
                    
                    roc_data['class'].append(classes[i])
                    roc_data['fpr'].append(fpr)
                    roc_data['tpr'].append(tpr)
                    roc_data['auc'].append(roc_auc)
                else:
                    print(f"Skipping ROC curve for class {classes[i]} - no positive samples")
            except:
                continue
        
        try:
            y_bin = y_true_bin[:, :n_classes]
            probs_trim = probs[:, :n_classes]
            
            valid_classes = []
            for i in range(n_classes):
                if np.sum(y_bin[:, i]) > 0:
                    valid_classes.append(i)
            
            if valid_classes:
                valid_y_bin = y_bin[:, valid_classes]
                valid_probs = probs_trim[:, valid_classes]
                
                macro_auc = roc_auc_score(valid_y_bin, valid_probs, multi_class='ovr', average='macro')
                micro_auc = roc_auc_score(valid_y_bin, valid_probs, multi_class='ovr', average='micro')
                
                self.metrics[name].update({'macro_auc': macro_auc, 'micro_auc': micro_auc})
        except Exception as e:
            print(f"Error calculating AUC metrics: {e}")
            pass
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlim=[0, 1], ylim=[0, 1.05], title=f'ROC Curves for {name}', xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend(loc="best", fontsize='small', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        
        if path and roc_data['class']:
            self.savePlot(fig, path)
            
            if save_csv:
                all_roc_data = []
                for i, cls in enumerate(roc_data['class']):
                    df = pd.DataFrame({
                        'Class': cls, 'FPR': roc_data['fpr'][i],
                        'TPR': roc_data['tpr'][i], 'AUC': roc_data['auc'][i]
                    })
                    all_roc_data.append(df)
                
                if all_roc_data:
                    final_df = pd.concat(all_roc_data, ignore_index=True)
                    self.saveDfToCSv(final_df, path, "_roc")
        
        plt.show()
        return roc_data
    
    def computeModelROC(self, name):
        probs = self.results[name]['probabilities']
        true_labels = self.results[name]['true_labels']
        classes = list(self.class_names[name].values())
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        binary_correct = []
        confidence_scores = []
        
        valid_samples = 0
        for prob, true in zip(probs, true_labels):
            if true in class_to_idx:
                pred_idx = np.argmax(prob)
                true_idx = class_to_idx[true]
                binary_correct.append(1 if pred_idx == true_idx else 0)
                confidence_scores.append(np.max(prob))
                valid_samples += 1
        
        if valid_samples == 0 or np.sum(binary_correct) == 0:
            print(f"Warning: No valid samples or all predictions incorrect for model {name}")
            # dummy in case the classes not positive in ROC
            return [0, 1], [0, 0], 0.5
        
        # ROC sageguard for empty or all-zero confidence scores
        try:
            fpr, tpr, _ = roc_curve(binary_correct, confidence_scores)
            roc_auc = auc(fpr, tpr)
            return fpr, tpr, roc_auc
        except Exception as e:
            print(f"Error calculating ROC for {name}: {e}")
            return [0, 1], [0, 1], 0.5
    
    def plotModelCompareROC(self, models, title, path=None, save_csv=False):
        fig, ax = plt.subplots(figsize=(10, 8))
        compare_data = {'model': [], 'fpr': [], 'tpr': [], 'auc': [], 'accuracy': []}
        
        for model in models:
            try:
                fpr, tpr, roc_auc = self.computeModelROC(model)
                ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.2f})')
                
                compare_data['model'].append(model)
                compare_data['fpr'].append(fpr)
                compare_data['tpr'].append(tpr)
                compare_data['auc'].append(roc_auc)
                compare_data['accuracy'].append(self.metrics.get(model, {}).get('accuracy', -1))
            except:
                continue
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        plt.tight_layout()
        
        if path and compare_data['model']:
            self.savePlot(fig, path)
            
            if save_csv:
                all_data = []
                for i, model in enumerate(compare_data['model']):
                    df = pd.DataFrame({
                        'Model': model, 'FPR': compare_data['fpr'][i],
                        'TPR': compare_data['tpr'][i], 'AUC': compare_data['auc'][i]
                    })
                    all_data.append(df)
                
                if all_data:
                    full_df = pd.concat(all_data, ignore_index=True)
                    self.saveDfToCSv(full_df, path, "_roc")
                
                summary_df = pd.DataFrame({
                    'Model': compare_data['model'],
                    'AUC': compare_data['auc'],
                    'Accuracy(%)': compare_data['accuracy']
                })
                self.saveDfToCSv(summary_df, path, "_summary")
        
        plt.show()
        return compare_data
    
    ############# MODEL TYPE COMPARISON #################################################
    def compareModels(self):
        if not self.metrics:
            return None
            
        metrics_data = [{'Model': n, 'Accuracy (%)': m['accuracy']} for n, m in self.metrics.items()]
        df = pd.DataFrame(metrics_data)
        
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
        
        return df
    
    def compareModelTypeVariants(self, model_type, path=None, save_csv=False):
        variants = [m for m in self.results.keys() 
                    if m.startswith(f"{model_type}_") and 
                    self.results[m].get('probabilities') is not None]
        
        if not variants:
            return []
        
        title = f'ROC Comparison of {model_type} Variants'
        return self.plotModelCompareROC(variants, title, path, save_csv)
    
    def compareBestAcrossModelTypes(self, model_types=None, path=None, save_csv=False):
        if not model_types:
            model_types = set()
            for model in self.results.keys():
                if '_' in model and self.results[model].get('probabilities') is not None:
                    model_types.add(model.split('_')[0])
        
        if not model_types:
            return []
        
        best_models = []
        for model_type in model_types:
            variants = [m for m in self.results.keys() 
                        if m.startswith(f"{model_type}_") and 
                        self.results[m].get('probabilities') is not None]
            
            if variants:
                best = max(variants, key=lambda v: self.metrics.get(v, {}).get('accuracy', -1))
                best_models.append(best)
        
        if not best_models:
            return []
        
        title = 'Comparison of Best Model Variants'
        return self.plotModelCompareROC(best_models, title, path, save_csv)

    def compareBestModelVariants(self, path=None, save_csv=False):
        return self.compareBestAcrossModelTypes(path=path, save_csv=save_csv)

    ############# DATA EXPORT #################################################
    def exportAllConfusionMatrices(self, output_dir, save_csv=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not self.confusion_matrices:
            return
            
        for name in self.confusion_matrices.keys():
            if name in self.class_names:
                path = os.path.join(output_dir, f"{name}_confusion_matrix")
                self.plotConfusionMatrix(name, path=path, save_csv=save_csv)
    
    def exportTestResults(self, output_path, save_csv=True):
        if not self.metrics or not save_csv:
            return
            
        metrics_data = []
        for name, metric_dict in self.metrics.items():
            row = {'Model': name}
            
            standard_metrics = ['accuracy', 'macro_auc', 'micro_auc']
            for metric in standard_metrics:
                if metric in metric_dict and metric_dict[metric] is not None:
                    if metric == 'accuracy':
                        row[metric] = f"{metric_dict[metric]:.2f}"
                    else:
                        row[metric] = f"{metric_dict[metric]:.4f}"
            
            for k, v in metric_dict.items():
                if k not in standard_metrics and v is not None:
                    row[k] = v
                    
            metrics_data.append(row)
            
        result_df = pd.DataFrame(metrics_data)
        result_df = result_df.dropna(axis=1, how='all')
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        result_df.to_csv(output_path, index=False)
        return result_df

################### MAIN #################################################
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
        prefix="EfficientNet",
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
        save_path=os.path.join(output_dir, "efficientnet_variants"),  
        save_csv=True, 
    )
    
    print("\nTest 3: Evaluation Result Matrix:")
    print(result_matrix)
    
    print("\n" + "="*70)
    print("Test 4: Analyzing model performance")
    print("="*70)
    
    # ROC curves for the best model with optional saving
    if result_matrix is not None and not result_matrix.empty:
        best_model = result_matrix.loc[result_matrix['Accuracy'].idxmax(), 'Model']
        print(f"\nPlotting ROC curve for best model: {best_model}")
        evaluator.plotROCCurve(best_model, path=os.path.join(output_dir, f"{best_model}_roc"), save_csv=True)
        
        print(f"\nPlotting confusion matrix for best model: {best_model}")
        evaluator.plotConfusionMatrix(best_model, path=os.path.join(output_dir, f"{best_model}_cm"), save_csv=True)  
    
    if len(evaluator.models) > 1:
        print("\nComparing all EfficientNet variants:")
        evaluator.compareModelTypeVariants("EfficientNet", path=os.path.join(output_dir, "efficientnet_variants_roc"), save_csv=True)
        
        # Automatic detection of model types for comparison
        print("\nComparing best variant of each detected model type:")
        evaluator.compareBestAcrossModelTypes(path=os.path.join(output_dir, "best_models_comparison"), save_csv=True)
        
        # compare ResNet variants (todo)
        
        # compare best variants across different model types (todo)
        
    
    # test all export with optional CSV saving
    print("\n" + "="*70)
    print("Test 5: Exporting confusion matrices and test results")
    print("="*70)
    evaluator.exportAllConfusionMatrices(output_dir, save_csv=True) 
    evaluator.exportTestResults(os.path.join(output_dir, "test_results.csv"), save_csv=True) 

    print("\n" + "="*70)
    print("ModelEvaluator tests completed")
    print("="*70)