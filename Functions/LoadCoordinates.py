import pandas as pd

def load_tile_coordinates(csv_path: str) -> pd.DataFrame:
    """
    Load tile coordinates from a CSV and return as a clean DataFrame.

    Returns:
        DataFrame with columns: ['tile_name', 'x', 'y']
    """
    df = pd.read_csv(csv_path)
    df['tile_name'] = df['Tile'].str.strip()
    df['x'] = df['X'].astype(int)
    df['y'] = df['Y'].astype(int)
    return df[['tile_name', 'x', 'y']]

