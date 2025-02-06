import os

# Define the base directory to search
base_dir = "/volumes1/vlm-cl/normal_cls/lang"

# Flags to track the presence of 'model.ph' and 'checkpoint_100.pth'
model_ph_exists = False
checkpoint_100_exists = False

# Traverse the directory to check for the existence of 'model.ph' and 'checkpoint_100.pth'
for root, dirs, files in os.walk(base_dir):
    if "model.ph" in files:
        model_ph_exists = True
    if "checkpoint_100.pth" in files:
        checkpoint_100_exists = True

# Traverse the directory again to delete files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(root, file)
        # Check if the file ends with '.pth' or '.ph'
        if file.endswith(".pth") or file.endswith(".ph"):
            # Do not delete 'model.ph' or 'checkpoint_100.pth' if conditions are met
            if file == "model.ph":
                continue
            if file == "checkpoint_100.pth" and not model_ph_exists:
                continue
            # Delete the file
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
