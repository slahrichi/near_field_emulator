import os
import argparse
import yaml

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Flipbook Comparison</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .comparison-row {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 40px;
        }}
        .model-card {{
            background-color: #f7f7f7;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 45%; /* Allows two cards per row */
        }}
        .model-header {{
            margin-bottom: 10px;
            padding: 10px 0;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }}
        .gif-container {{
            display: flex; /* Makes ground truth and prediction appear side-by-side */
            justify-content: space-around; /* Adds even spacing between GIFs */
            gap: 10px; /* Adds small spacing between the two GIFs */
        }}
        .gif-box {{
            text-align: center;
            margin-bottom: 20px;
        }}
        img {{
            width: 100%;
            max-width: 250px;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        img:hover {{
            transform: scale(1.05);
        }}
        h1, h2 {{
            text-align: center;
        }}
        @media (max-width: 768px) {{
            .model-card {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <h1>Model Flipbook Comparisons</h1>
    {content}
</body>
</html>
"""

def get_model_id(params_file_path):
    """
    Extracts the `model_id` from the params.yaml file if it exists.
    """
    if os.path.exists(params_file_path):
        with open(params_file_path, 'r') as f:
            params = yaml.safe_load(f)
            model_dict = params.get("model", {})
            return model_dict.get("model_id", "Unknown Model ID")
    return "Unknown Model ID"

def generate_html_content(base_path, model_type, io_mode, spacing_mode, channel):
    html_content = ""

    # If model_type is None, include all available model types
    model_types = [model_type] if model_type else [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    io_modes = [io_mode] if io_mode else ["one_to_many", "many_to_many"]
    spacing_modes = [spacing_mode] if spacing_mode else ["distributed", "sequential"]

    for model_type in model_types:
        model_dir_path = os.path.join(base_path, model_type)
        if not os.path.exists(model_dir_path):
            print(f"Model type directory '{model_dir_path}' not found.")
            continue

        for io in io_modes:
            for spacing in spacing_modes:
                subdir_path = os.path.join(model_dir_path, io, spacing)
                if not os.path.exists(subdir_path):
                    print(f"Subdirectory '{subdir_path}' not found.")
                    continue

            html_content += f"<h2>{model_type.upper()} - {io.replace('_', ' ').title()} - {spacing.title()}</h2>"
            html_content += "<div class='comparison-row'>"

            # Iterate over each model directory
            for model_name in os.listdir(subdir_path):
                model_path = os.path.join(subdir_path, model_name, "flipbooks")
                params_path = os.path.join(subdir_path, model_name, "params.yaml")
                model_id = get_model_id(params_path)

                if not os.path.exists(model_path):
                    print(f"Model path '{model_path}' not found.")
                    continue

                # Gather the ground truth and predicted GIF paths
                ground_truth_file = os.path.join(model_path, f"sample_0_{channel}_groundtruth_valid.gif")
                predicted_file = os.path.join(model_path, f"sample_0_{channel}_prediction_valid.gif")

                if os.path.exists(ground_truth_file) and os.path.exists(predicted_file):
                    html_content += f"<div class='model-card'>"
                    html_content += f"<div class='model-header'>Model ID: {model_id}</div>"
                    html_content += """
                    <div class="gif-container">
                    <div class="gif-box">
                        <h3>Ground Truth</h3>
                        <img src="{ground_truth}" alt="Ground Truth">
                    </div>
                    <div class="gif-box">
                        <h3>Prediction</h3>
                        <img src="{predicted}" alt="Prediction">
                    </div>
                    </div>
                    </div>
                    """.format(ground_truth=ground_truth_file, predicted=predicted_file)

            html_content += "</div>"

    return html_content

def main():
    parser = argparse.ArgumentParser(description="Generate an HTML file to compare GIFs for different model configurations.")
    parser.add_argument("--base_path", type=str, default="/develop/results/meep_meep/", help="Base directory path containing model results.")
    parser.add_argument("--model_type", type=str, help="Type of model (e.g., lstm, convlstm). If not specified, include all models.")
    parser.add_argument("--io_mode", type=str, choices=["one_to_many", "many_to_many"], help="Input-output mode.")
    parser.add_argument("--spacing_mode", type=str, choices=["distributed", "sequential"], help="Spacing mode.")
    parser.add_argument("--channel", type=str, required=True, choices=["mag", "phase"], help="Channel to use (mag or phase).")
    parser.add_argument("--output_file", type=str, default="comparison.html", help="Output HTML file name.")

    args = parser.parse_args()

    # Generate HTML content
    html_content = generate_html_content(args.base_path, args.model_type, args.io_mode, args.spacing_mode, args.channel)

    # Write the final HTML file
    with open(args.output_file, "w") as f:
        f.write(HTML_TEMPLATE.format(content=html_content))

    print(f"HTML file generated: {args.output_file}")

if __name__ == "__main__":
    main()