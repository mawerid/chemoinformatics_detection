import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import FunctionTransformer

def mol_dsc_calc_SI(mols, descriptors):
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k, 
                         f in descriptors.items()} for m in mols)


def predict(data: str) -> [float, float, float]:
    df = pd.DataFrame(data, columns = 'SMILES')
    descriptors = {
                "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
                "MR": Descriptors.MolMR,
                "NumValenceElectrons": Descriptors.NumValenceElectrons,
                "NOCount": Descriptors.NOCount,
                'fr_pyridine':Descriptors.fr_pyridine,
                "NumRotatableBonds": Descriptors.NumRotatableBonds,
                'fr_ether':Descriptors.fr_ether
            }
    descriptors_transformer = FunctionTransformer(mol_dsc_calc_SI, descriptors)
    X_des = descriptors_transformer.transform(df['SMILES'])
    df_des = pd.DataFrame(X_des, columns = 'fr_pyridine')
    df_des = df_des.dropna(subset=['fr_pyridine'])
    reg = xgb.XGBRegressor()
    reg.load_model("weights\SI_XGB_model_after_GridSrch.json")
    si = reg.predict(df_des) 
    return  si
    



