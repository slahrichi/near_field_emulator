import os
import shutil
import yaml
import sys
from pdf2image import convert_from_path

sys.path.append("../")
from conf.schema import load_config

def save_file(filepath, save_directory):
    """Save different file types to a specified directory"""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if filepath.endswith('.pdf'):
        try:
            images = convert_from_path(filepath)
            for i, img in enumerate(images):
                img.save(os.path.join(save_directory, f"{os.path.basename(filepath)}_page_{i}.png"))
        except Exception as e:
            print(f"Could not save PDF: {filepath}")
            print(f"Error: {e}")

    elif filepath.endswith(('.png', '.jpg', '.jpeg', '.gif', '.txt')):
        shutil.copy(filepath, save_directory)

def save_plot_directories(model_dir, save_base_dir):
    """Save plot directories for a specific model directory"""
    plot_dirs = ['loss_plots', 'dft_plots', 'flipbooks', 'performance_metrics']
    model_name = os.path.basename(model_dir)
    model_save_dir = os.path.join(save_base_dir, model_name)
    
    print(f"\nProcessing model: {model_name}")
    
    for plot_dir in plot_dirs:
        source_dir = os.path.join(model_dir, plot_dir)
        if os.path.exists(source_dir):
            save_dir = os.path.join(model_save_dir, plot_dir)
            print(f"Saving {plot_dir} from {model_name}")
            save_all_in_directory(source_dir, save_dir)
        else:
            print(f"Directory not found: {source_dir}")

def save_all_in_directory(directory, save_directory):
    """Save all supported files in a directory to another directory"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    files = os.listdir(directory)
    if not files:
        print(f"No files found in {directory}")
        return

    print(f"Found {len(files)} files in {directory}")
    for filename in sorted(files):
        if filename.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.txt')):
            print(f"Saving {filename}")
            filepath = os.path.join(directory, filename)
            save_file(filepath, save_directory)

def process_model_type(config_path):
    """Process all model directories for model type specified in config"""
    # Load config
    config = load_config(config_path)
    model_type = config.model.arch
    
    base_path = '/develop/results/meep_meep'
    model_type_path = os.path.join(base_path, model_type)
    save_base_path = f'/develop/saved_results/{model_type}'
    
    if not os.path.exists(model_type_path):
        print(f"Model type directory not found: {model_type_path}")
        return
    
    model_dirs = [d for d in os.listdir(model_type_path) 
                 if os.path.isdir(os.path.join(model_type_path, d))]
    
    if not model_dirs:
        print(f"No model directories found in {model_type_path}")
        return
    
    print(f"\nFound {len(model_dirs)} model directories for {model_type}")
    
    for model_dir in model_dirs:
        full_model_path = os.path.join(model_type_path, model_dir)
        save_plot_directories(full_model_path, save_base_path)

if __name__ == "__main__":
    process_model_type('/develop/code/near_field_emulator/config.yaml')
    
    # then,
    # kubectl cp save-eval-plots:/develop/saved_results /develop/results