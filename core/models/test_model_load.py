import torch
import yaml
#from omegaconf import OmegaConf
from WaveMLP import WaveMLP
from near_field_emulator import conf
from near_field_emulator.conf.schema import load_config
from near_field_emulator.conf.schema import ModelConfig
import warnings
           
torch.serialization.add_safe_globals([ModelConfig])

# checkpoint_path = "/develop/code/model-v2.ckpt"
# config_path = "/develop/code/params.yaml"

checkpoint_path = "/develop/results/meep_meep/cvnn/model_forward-v2/model.ckpt"
config_path = "/develop/results/meep_meep/cvnn/model_forward-v2/params.yaml"
forward_conf = load_config(config_path)
model = WaveMLP(model_config=forward_conf.model)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
model.eval()

inp = torch.randn(1, 9)
#inp = torch.randn(1, 2, 166, 166)
output = model(inp)
print(output)
# 1. rebuild the conf file from the yaml file
# 2. instantiate the forward model (WaveMLP object) using conf
# 3. load_state_dict from the saved checkpoint

# forward_conf = load_config(config_path)
# # Initialize the model
# model = WaveMLP(model_config=forward_conf)  # Ensure `your_model_config` is defined
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
# # Load the state dictionary
# model.load_state_dict(checkpoint['state_dict'])

# def load_forward_model(forward_ckpt_path: str, forward_config_path: str):
#     """Load a trained forward model from checkpoint and config."""
#     # Load forward model config
#     forward_conf = load_config(forward_config_path)
#     print(forward_conf.model)
#     # Instantiate forward model
#     forward_model = WaveMLP(forward_conf)
#     # Load checkpoint (extract state_dict, ignore Lightning wrapper)
#     checkpoint = torch.load(forward_ckpt_path, map_location='cpu', weights_only=False)
#     print("##############")
#     print(checkpoint.keys()) 
#     #forward_model.load_state_dict(checkpoint['state_dict'])
#     #print(checkpoint['state_dict'])
#     forward_model.eval()
#     return forward_model


#fwd = load_forward_model(checkpoint_path, config_path)

# with open(config_path, "r") as f:
#     conf_dict = yaml.safe_load(f)
# conf = OmegaConf.create(conf_dict)  
# print(conf.keys())
# # model = WaveMLP(model_config=conf)  # Match your __init__ signature

# # # Load checkpoint weights directly
# # checkpoint_path = "/develop/code/model-v2.ckpt"
# # checkpoint = torch.load(checkpoint_path, map_location="cpu")

# # # Load state_dict only (avoids dependency on original conf module)
# # model.load_state_dict(checkpoint["state_dict"])
# # model.eval()
# # print("Model loaded successfully!")

# import yaml
# import torch

# # Import your model class and any config conversion functions.
# # (Adjust the import paths as necessary.)
# from WaveMLP import WaveMLP  
# from near_field_emulator.conf.schema import load_config   # If you have a function to create a config object
# from near_field_emulator.conf.schema import MainConfig

# from typing import Any, Type
# from pydantic import BaseModel

# def from_plain_dict(plain: Any, cls: Type[Any] = None) -> Any:
#     """
#     Recursively convert plain dictionaries and lists back into objects.
    
#     If a target class (typically a subclass of BaseModel) is provided,
#     the plain dict is converted via that class’s parse_obj method.
    
#     Parameters:
#         plain: The plain dictionary (or list) to convert.
#         cls:   Optional target class. For example, MainConfig.
        
#     Returns:
#         An instance of cls if provided and appropriate, or a recursively
#         reconstructed structure otherwise.
#     """
#     if cls is not None and isinstance(plain, dict) and issubclass(cls, BaseModel):
#         # Use Pydantic’s parse_obj to reconstruct the model.
#         return cls.parse_obj(plain)
#     elif isinstance(plain, dict):
#         # Recursively process each value.
#         return {k: from_plain_dict(v) for k, v in plain.items()}
#     elif isinstance(plain, list):
#         return [from_plain_dict(item) for item in plain]
#     else:
#         return plain



# # --- Step 1. Load the YAML configuration ---
# with open('/develop/code/params.yaml', 'r') as f:
#     plain_config = yaml.safe_load(f)

# # Rebuild the MainConfig instance:
# config = from_plain_dict(plain_config, MainConfig)

# # In our code below we assume that the section for the model is stored under 'model'
# model_config = config['model']

# # --- Step 2. Load the checkpoint using Lightning's load_from_checkpoint ---
# # (Assuming that the training code used self.save_hyperparameters(), which it does.)
# checkpoint_path = "/develop/code/model-v2.ckpt"

# # Use Lightning's built-in method.
# # You can also pass any additional arguments that the __init__ requires.
# model = WaveMLP.load_from_checkpoint(checkpoint_path, model_config=model_config, fold_idx=None)

# # Set the model to evaluation mode
# model.eval()

# # --- Step 3. Evaluate the model on new data ---
# # For the 'inverse' mode used here, the forward() expects an input tensor 
# # representing near_fields of shape (batch_size, 2, 166, 166)
# # (The two channels correspond to the real and imaginary parts.)
# dummy_near_fields = torch.randn(1, 2, 166, 166)  # Change shape as appropriate
# output = model(dummy_near_fields)

# print("Model output:", output)
