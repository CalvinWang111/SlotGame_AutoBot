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

    # Extract base name before the first underscore
    Snapshot = os.path.splitext(selected_image)[0]  # Remove file extension
    Snapshot = Snapshot.split('_')[0]  # Extract base name
    print(Snapshot)

    output_dir = os.path.join(root_dir, 'marquee_tool', Snapshot)  # Set output directory name

    # Create or use the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process the selected image
    image_path = os.path.join(image_dir, selected_image)
    original_image_size = screenshot.get_image_size(image_path)  # Record original image size
    regions = screenshot.interactive_labeling(image_path=image_path, output_dir=output_dir, Snapshot=Snapshot)

    # Add original image size to regions dictionary
    regions_with_size = {
        "original_image_size": original_image_size,
        "regions": regions
    }

    # Check if JSON file already exists and merge content if necessary
    output_file = os.path.join(output_dir, Snapshot + "_regions.json")
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
            # Merge regions into existing data
            if "regions" in existing_data:
                existing_data["regions"].extend(regions_with_size["regions"])
            else:
                existing_data["regions"] = regions_with_size["regions"]

            # Ensure no duplicates in regions
            unique_regions = {json.dumps(region, sort_keys=True) for region in existing_data["regions"]}
            existing_data["regions"] = [json.loads(region) for region in unique_regions]

            regions_with_size = existing_data
        except Exception as e:
            print(f"An error occurred while reading the existing JSON file: {e}")

    # Write the updated regions dictionary to the JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(regions_with_size, file, indent=4, ensure_ascii=False)
        print(f"Regions successfully saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while writing to file: {e}")

marquee_tool()