import os

"""
This is a utility script for extracting the whole code into a single markdown file. Nothing important for the main functionality
"""

def write_python_scripts_to_markdown(directory, output_file):
    with open(output_file, 'w') as md_file:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    md_file.write(f'## {file}\n\n')
                    md_file.write('```python\n')
                    md_file.write(f'# {file}\n\n')
                    with open(file_path, 'r') as py_file:
                        md_file.write(py_file.read())
                    md_file.write('\n```\n\n')

if __name__ == "__main__":
    directory_to_scan = './'  # Replace with your directory path
    output_markdown_file = 'output.md'  # Replace with your desired output file name
    write_python_scripts_to_markdown(directory_to_scan, output_markdown_file)
    print(f'All Python scripts have been written to {output_markdown_file}')
