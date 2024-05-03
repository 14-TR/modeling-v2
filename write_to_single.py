import os

# Specify the directory you want to combine scripts from
directory = r'C:\Users\tingram\Desktop\Captains Log\UWYO\GIT\modeling-v2'

# Specify the output file
output_file = 'combined_script.py'

# Get a list of all Python files in the directory
python_files = [f for f in os.listdir(directory) if f.endswith('.py')]

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    for fname in python_files:
        # Open each Python file in read mode
        with open(os.path.join(directory, fname)) as infile:
            # Write the contents of the Python file to the output file
            outfile.write(infile.read())
            # Write a newline character to separate scripts
            outfile.write('\n')

print(f"Combined scripts saved to {output_file}")