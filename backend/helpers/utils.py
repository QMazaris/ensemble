import os

def create_directories(directories, summary=False):
    """Create all necessary output directories if they don't exist.
    
    Args:
        directories (list): List of directory paths to create
        summary (bool): Whether to print status messages
    """
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            if summary:
                print(f"✓ Created or found: {directory}")
        except Exception as e:
            if summary:
                print(f"❌ Error creating {directory}: {str(e)}") 