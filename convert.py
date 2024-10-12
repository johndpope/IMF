import re
import os
import sys
import argparse

def convert_var_name(name):
    # Convert from snake_case to camelCase
    name = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), name)
    
    # Common PyTorch to TensorFlow naming conventions
    conversions = {
        'conv': 'conv',
        'bn': 'batchNorm',
        'relu': 'relu',
        'leakyrelu': 'leakyRelu',
        'maxpool': 'maxPool',
        'avgpool': 'avgPool',
        'linear': 'dense',
        'dropout': 'dropout',
        'upsample': 'upSampling',
        'flatten': 'flatten',
        'lstm': 'lstm',
        'gru': 'gru',
    }
    
    for pt_name, tf_name in conversions.items():
        name = re.sub(fr'\b{pt_name}\b', tf_name, name, flags=re.IGNORECASE)
    
    return name

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Convert variable names
    converted_content = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', lambda m: convert_var_name(m.group(1)), content)
    
    # Convert torch functions to tf equivalents (this is a basic example and needs expansion)
    torch_to_tf = {
        'torch.': 'tf.',
        'nn.': 'tf.keras.layers.',
        'F.': 'tf.nn.',
        '.cuda()': '',  # Remove .cuda() calls
    }
    for torch_func, tf_func in torch_to_tf.items():
        converted_content = converted_content.replace(torch_func, tf_func)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(converted_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model file to TensorFlow naming conventions")
    parser.add_argument("input_file", help="Input PyTorch model file")
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = "./tf"
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    process_file(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")