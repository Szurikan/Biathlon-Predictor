import os
import pandas as pd

def load_data(file_path=None):
    """
    Įkelia sportininkių duomenis iš CSV failo.
    
    Args:
        file_path (str, optional): Kelias iki CSV failo. Jei nenurodyta, 
                                   naudojamas numatytasis kelias.
    
    Returns:
        pandas.DataFrame: Įkelti duomenys
    """
    if file_path is None:
        # Numatytasis kelias į duomenų failą
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "data", "female_athletes_2425_full_stats_with_ranks.csv")
    
    # Tikriname, ar failas egzistuoja
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Duomenų failas nerastas: {file_path}")
    
    # Įkeliame duomenis
    df = pd.read_csv(file_path)
    
    return df