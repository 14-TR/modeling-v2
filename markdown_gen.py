import os

# Function to write directory contents to a Markdown file
def write_directory_to_md(dir_path, output_file):
    with open(output_file, 'w') as md_file:
        for root, dirs, files in os.walk(dir_path):
            level = root.replace(dir_path, '').count(os.sep)
            indent = ' ' * 4 * level
            md_file.write(f'{indent}- {os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                md_file.write(f'{subindent}- {f}\n')
                # Check if the file is a Python script
                if f.endswith('.py'):
                    # Open the Python script and read its contents
                    with open(os.path.join(root, f), 'r') as script_file:
                        script_contents = script_file.read()
                    # Write the script contents to the markdown file
                    md_file.write(f'```python\n{script_contents}\n```\n')

# Specify the directory path and output Markdown file name
directory_path = directory_path = r'C:\Users\TR\Desktop\Spring 2024 Courses\z\GIT\modeling-v2\modeling-v2'  # Current directory. Change it if necessary  # Current directory. Change it if necessary
output_md = 'directory_contents.md'

# Call the function
write_directory_to_md(directory_path, output_md)
