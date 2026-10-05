"""Shared helpers: paths, Excel input reader and solution writer."""
from pathlib import Path

import pandas as pd

# Repository layout: code/phase2_multi_source/<this file> -> repo root is two levels up
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "phase2_input_unknown_source.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
SOLUTION_FILE = RESULTS_DIR / "phase2_solution.txt"


def read_excel_data(filename, sheet_name):
    """Read one sheet of the input workbook.

    A single row or column is returned as a list; a matrix is returned as a
    dict {(i, j): value}.
    """
    data = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    values = data.values
    if min(values.shape) == 1:
        if values.shape[0] == 1:
            values = values.tolist()
        else:
            values = values.transpose()
            values = values.tolist()
        return values[0]
    else:
        data_dict = {}
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                data_dict[(i, j)] = values[i][j]
        return data_dict


def write_solution(built_sol, incoming_heat_sol, outcoming_heat_sol, source_id, filename=SOLUTION_FILE):
    """Write the solution of one MILP run in a format visualization.py can read back."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"source_id = {source_id}\n\n")

        f.write("built = {\n")
        for i in sorted(built_sol.keys()):
            f.write(f"    {i}: {built_sol[i]},\n")
        f.write("}\n\n")

        f.write("incoming_heat = {\n")
        for i in sorted(incoming_heat_sol.keys()):
            f.write(f"    {i}: {incoming_heat_sol[i]},\n")
        f.write("}\n\n")

        f.write("outcoming_heat = {\n")
        for i in sorted(outcoming_heat_sol.keys()):
            f.write(f"    {i}: {outcoming_heat_sol[i]},\n")
        f.write("}\n")
