import torch 
import os 
import glob

class ModelManager:
    def __init__(self, model, optimizer, device, model_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_dir = model_dir

    def save(self, epoch, loss, model_name="model"):
        os.makedirs(self.model_dir, exist_ok=True)

        filename = f"{model_name}_epoch{epoch}_loss{loss:.4f}.pth"
        save_path = os.path.join(self.model_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
        
        return save_path
        
    def save_best(self, epoch, loss, model_name="model"):
        os.makedirs(self.model_dir, exist_ok=True)
        
        best_model_path = os.path.join(self.model_dir, f"{model_name}_best.pth")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        torch.save(checkpoint, best_model_path)
        
        info_path = os.path.join(self.model_dir, f"{model_name}_epoch{epoch}_loss{loss:.4f}.pth")
        torch.save(checkpoint, info_path)
        
        for old_file in glob.glob(os.path.join(self.model_dir, f"{model_name}_epoch*_loss*.pth")):
            if old_file != info_path:  
                try:
                    os.remove(old_file)
                    print(f"Removed older model: {os.path.basename(old_file)}")
                except Exception as e:
                    print(f"Warning: Could not remove file {old_file}: {e}")
        
        print(f"Best model saved to {best_model_path}")
        return best_model_path

    def load(self, best_only=True, model_name="model"):
        try:
            if best_only:
                model_path = os.path.join(self.model_dir, f'{model_name}_best.pth')
                if not os.path.exists(model_path):
                    print(f"No best model found at {model_path}")
                    return self.load(best_only=False, model_name=model_name)
            else:
                model_files = glob.glob(os.path.join(self.model_dir, f'{model_name}_epoch*_loss*.pth'))
                if not model_files:
                    model_path = os.path.join(self.model_dir, f'{model_name}.pth')
                    if not os.path.exists(model_path):
                        print(f"No model file found at {model_path}")
                        return 0, float('inf')
                else:
                    model_files.sort(key=lambda x: float(x.split('_loss')[-1].split('.pth')[0]))
                    model_path = model_files[0] 
            
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'], checkpoint['loss']
        except Exception as e:
            print(f"Error loading model: {e}")
            return 0, float('inf')