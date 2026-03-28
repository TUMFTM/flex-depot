
import yaml
from flex_dep_opt.io.flexibility_io import (
    read_flexibility_bounds_csv,
)

def load_settings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def generate_fcr_availability_df():
    # load flexibility bounds csv, todo dont do this here
    cfg = load_settings("src/flex_dep_opt/config/settings_example.yaml")
    opt_cfg = cfg["optimization"]
    flex_cfg = opt_cfg.get("flexibility", {})
    flexibility_bounds_full = read_flexibility_bounds_csv(flex_cfg["bounds_file"])

    symmetric_limit = flexibility_bounds_full.copy()
    
    # for fcr, we need the symmetric limit, which is the minimum of the upper and lower capacity bounds
    symmetric_limit['inst_symmetric_limit'] = symmetric_limit[['Capacity_upper_kWh', 'Capacity_lower_kWh']].abs().min(axis=1)

    # resample to 4h blocks by taking the minimum symmetric limit within each block, represents capacity for 4h auctions
    fcr_grouped = symmetric_limit['inst_symmetric_limit'].resample('4h', label='left', closed='left').min()

    return symmetric_limit, fcr_grouped