#
# This script is intended to generate the exo_no_mesh.xml file using
# the XMLModel class from the mujoco_simulation package.
# This makes sure that the xml files are the same in sim and mjpc.
#

import os
from mujoco_simulation.model.robot_model import XMLModel

def generate_exo_xml():
    # Create instance of the XMLModel class
    xml_model = XMLModel()

    # Set whether the humanoid is included in the xml file
    xml_model.humanoid = False

    # Get the complete XML string
    xml_string = xml_model.get_xml_string()

    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full path for the output file
    output_file = os.path.join(current_dir, "exo_no_mesh.xml")
    # Create the build directory path using relative pathing because the build directory is quite a bit away from the source directory
    build_dir = os.path.join(current_dir, "..", "..", "..", "build", "mjpc", "tasks", "exo")
    
    # Create the full path for the build output file
    build_output_file = os.path.join(build_dir, "exo_no_mesh.xml")
    
    # Create directories if they don't exist so it doesnt crash if MJPC is not built yet
    os.makedirs(build_dir, exist_ok=True)
    
    # Write the XML content to the build file
    with open(build_output_file, "w") as file:
        file.write(xml_string)
        print(f"Generated XML file at: {build_output_file}")

    # Write the XML content to the file
    with open(output_file, "w") as file:
        file.write(xml_string)
        print(f"Generated XML file at: {output_file}")

if __name__ == "__main__":
    generate_exo_xml()
    

