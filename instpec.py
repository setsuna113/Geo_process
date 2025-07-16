import fiona
import geopandas as gpd

# Define the path to your GeoPackage file
gpkg_path = '/home/jason/gpkg_data/00/00/Acacia oerfota.gpkg'

# --- Step 1: List all layers within the GeoPackage ---
# A single GeoPackage file can contain multiple layers.
# We use fiona to get a list of their names.
try:
    layer_names = fiona.listlayers(gpkg_path)
    print(f"Layers found in '{gpkg_path}':")
    for name in layer_names:
        print(f"- {name}")
except fiona.errors.DriverError as e:
    print(f"Error accessing the GeoPackage file: {e}")
    exit()

# --- Step 2: Choose a layer and inspect its data dimensions (columns) ---
# Let's assume your plant polygon layer is the first one,
# or you know its name from the list above.
if layer_names:
    # Replace 'layer_names[0]' with the actual name of your layer if needed
    target_layer_name = layer_names[0] 
    
    print(f"\nInspecting layer: '{target_layer_name}'")

    # Read the specific layer into a GeoDataFrame
    # A GeoDataFrame is like a spreadsheet with a special 'geometry' column.
    gdf = gpd.read_file(gpkg_path, layer=target_layer_name)

    # The data dimensions are the columns of the GeoDataFrame.
    print("Data dimensions (columns) found in this layer:")
    for column in gdf.columns:
        print(f"- {column}")

    # You can also get a quick preview of the data
    print("\nFirst 5 rows of data:")
    print(gdf.head())
else:
    print("No layers found in the GeoPackage.")