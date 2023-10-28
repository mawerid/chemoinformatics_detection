import onnxmltools
import pandas as pd
import xgboost as xgb
from onnxconverter_common import FloatTensorType
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import FunctionTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem

descriptors_SI = {
    "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
    "MR": Descriptors.MolMR,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NOCount": Descriptors.NOCount,
    'fr_pyridine': Descriptors.fr_pyridine,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    'fr_ether': Descriptors.fr_ether
}
descriptors_IC = {
    "MR": Descriptors.MolMR,
    'fr_bicyclic': Descriptors.fr_bicyclic,
    'fr_NH1': Descriptors.fr_NH1,
    'fr_methoxy': Descriptors.fr_methoxy,
    "NHOHCount": Descriptors.NHOHCount
}

#url_wights = "https://raw.githubusercontent.com/mawerid/chemoinformatics_detection/main/weights/"
url_wights = "../weights/"

si_reg = xgb.XGBRegressor()
si_reg.load_model(url_wights + "SI_XGB_model_after_GridSrch.json")

ic_reg = xgb.XGBRegressor()
ic_reg.load_model(url_wights + "IC50_XGB_model_cut_descriptors.json")

initial_types = [(
'float_input', FloatTensorType([None, 8])
)]
si_onnx_model = onnxmltools.convert_xgboost(si_reg, initial_types=initial_types)
ic_onnx_model = onnxmltools.convert_xgboost(ic_reg, initial_types=initial_types)

onnxmltools.utils.save_model(si_onnx_model, '../models/xgboost_SI.onnx')
onnxmltools.utils.save_model(ic_onnx_model, '../models/xgboost_IC.onnx')

def mol_dsc_calc_SI(mols):
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k,
    f in descriptors_SI.items()} for m in mols)


def mol_dsc_calc_IC(mols):
    return pd.DataFrame({k: f(Chem.MolFromSmiles(m)) for k, f in descriptors_IC.items()} for m in mols)


def predict_SI(data):
    descriptors_transformer = FunctionTransformer(mol_dsc_calc_SI)
    X_des = descriptors_transformer.transform(pd.Series(data))
    X_des = X_des.dropna(subset=['fr_pyridine'])
    X_des = pd.DataFrame(X_des)
    df = pd.DataFrame([0])
    df_des = pd.concat([df, X_des], axis=1)
    df_des = df_des.dropna(subset=['fr_pyridine'])
    print(df_des)
    si = si_reg.predict(df_des)
    return abs(si[0])


def predict_IC(data):
    descriptors_transformer = FunctionTransformer(mol_dsc_calc_IC)
    X_des = descriptors_transformer.transform(pd.Series(data))
    X_des = X_des.dropna(subset=['fr_bicyclic'])
    X_des = pd.DataFrame(X_des)
    df1 = pd.DataFrame([0])
    df2 = pd.DataFrame([0])
    df3 = pd.DataFrame([0])
    df_des = pd.concat([df1, X_des, df2, df3], axis=1)
    df_des = df_des.dropna(subset=['fr_methoxy'])
    print(df_des)
    ic = ic_reg.predict(df_des)
    return abs(ic[0])


def predict(data: str) -> [float, float, float]:
    if data is None:
        return [0, 0, 0]
    data = data.strip()
    if len(data) == 0:
        return [0, 0, 0]
    try:
        si = predict_SI(data)
        ic = predict_IC(data)
        return [ic, ic * si, si]
    except:
        return [0, 0, 0]


print(predict("CC"))
