import pandas as pd

def create_js_file_from_csv(csv_path, js_path):
    # Load data from CSV
    data = pd.read_csv(csv_path)

    # Start the JavaScript array
    js_content = "const imageData = [\n"
    
    # Iterate over rows in the DataFrame to create JavaScript objects
    for index, row in data.iterrows():
        js_object = "    {\n"
        for col in data.columns:
            # Handle different data types appropriately
            if isinstance(row[col], str):
                reformatted = row[col].replace('"', '\"')
                js_object += f'        {col}: "{reformatted}",\n'  # Escape double quotes in strings
            else:
                js_object += f'        {col}: {row[col]},\n'
        js_object += "    },\n"
        js_content += js_object
    
    # Close the JavaScript array
    js_content += "];\n"

    # Write to a .js file
    with open(js_path, 'w') as file:
        file.write(js_content)

# Example usage
csv_file_path = 'merged_compositions.csv'  # Specify your CSV file path here
js_file_path = 'image_data_two_models.js'  # Specify your desired output JS file path
create_js_file_from_csv(csv_file_path, js_file_path)
