from screenshot import GameScreenshot
from pathlib import Path 
import json 
import os 
 
def marquee_tool(): 
    screenshot = GameScreenshot() 
    root_dir = Path(__file__).parent.parent.parent
    # Directory containing images 
    image_dir = os.path.join(root_dir, 'marquee_tool', 'competitive')

    # List all image files in the directory 
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]


    if not image_files: 
        print("No image files found in the directory.")
        return

    # Display the images with an index
    print("Available images:") 
    for idx, image_file in enumerate(image_files, start=1):
        print(f"{idx}. {image_file}") 

    # Prompt the user to select an image by index
    while True:
        try:
            selected_index = int(input("Enter the index of the image to process: ")) - 1
            if 0 <= selected_index < len(image_files):
                break
            else:
                print("Invalid index. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get the selected image 
    selected_image = image_files[selected_index] 

    Snapshot = os.path.splitext(selected_image)[0]  # Remove file extension

    output_dir = os.path.join(root_dir, 'marquee_tool', Snapshot)  # Set output directory name

    # Create or clear the output directory
    if os.path.exists(output_dir):
        # Clear the output folder if it exists
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove the sub-directory (only works if empty)
            except Exception as e: 
                print(f"Failed to delete {file_path}. Reason: {e}")
    else: 
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
 
    # Process the selected image 
    image_path = os.path.join(image_dir, selected_image)
    regions = screenshot.interactive_labeling(image_path=image_path, output_dir=output_dir, Snapshot=Snapshot)

    # Write the regions dictionary to a text file 
    output_file = os.path.join(output_dir, Snapshot + "_regions.json")
    try: 
        with open(output_file, 'w', encoding='utf-8') as file: 
            # Format the dictionary as JSON for readability 
            json.dump(regions, file, indent=4, ensure_ascii=False)
        print(f"Regions successfully saved to {output_file}")
    except Exception as e: 
        print(f"An error occurred while writing to file: {e}")
 
marquee_tool() 
