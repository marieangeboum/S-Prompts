import os
import imageio
import numpy as np

target_domain_expe = "/scratcht/FLAIR_1/train/D091_2021/"
root_flair_custom_mab = "/scratcht/FLAIR_1_Custom_912021/"
domaines_custom = os.listdir(target_domain_expe)
modes = ["train", "val"]
binary_classes = ['0_no_building', "1_building"]

for domain in domaines_custom:
    for mode in modes:
        for classe in binary_classes:
            # Input directory containing TIFF files
            input_directory = f"/scratcht/FLAIR_1_Custom_912021/{domain}/{mode}/{classe}/"
            # Output directory where PNG files will be saved
            output_directory = f"/scratcht/FLAIR_1_Custom_912021_png/{domain}/{mode}/{classe}/"

            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # List all TIFF files in the input directory
            tiff_files = [f for f in os.listdir(input_directory) if f.endswith(".tif")]

            for tiff_file in tiff_files:
                input_path = os.path.join(input_directory, tiff_file)
                output_file = os.path.splitext(tiff_file)[0] + ".png"
                output_path = os.path.join(output_directory, output_file)

                # Read the TIFF image
                try:
                    image = imageio.imread(input_path)
                    # If the image has 5 channels, select the first 4 channels and convert it to RGBA
                    if image.shape[2] == 5:
                        image = image[:, :, :3]
                    else:
                        # Handle other cases, e.g., 3 channels, differently if needed
                        pass

                    # Save the 4D (RGBA) image as PNG using imageio
                    imageio.imsave(output_path, image, format="png")
                    print(f"Converted {tiff_file} to {output_file}")
                except Exception as e:
                    print(f"Error converting {tiff_file}: {e}")

print("Conversion completed.")
