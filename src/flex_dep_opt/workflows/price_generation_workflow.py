from flex_dep_opt.market.prices_generator import (write_example_prices_DA_csv, write_from_epex_DA_csv, write_example_prices_ID_csv)

def run_generate_prices_DA(out_path: str):
    result = write_example_prices_DA_csv(out_path)
    print(f"Generated example price series → {result}")

def run_import_epex_DA(src: str, out: str):
    result = write_from_epex_DA_csv(src_path=src, dst_path=out)
    print(f"Imported and cleaned EPEX day-ahead data → {result}")

def run_generate_prices_ID(out_path: str):
    result = write_example_prices_ID_csv(out_path)
    print(f"Generated example price series → {result}")