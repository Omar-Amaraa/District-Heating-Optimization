"""MILP that designs the heat network for one candidate source.

Given a source, its installed power and an estimate of the gas share, the
model chooses which pipes to build and how much heat flows through each one,
at minimum total cost.
"""
from pulp import *
from utils import DATA_FILE, read_excel_data, write_solution


#  Load the input data (once, at import time)

_path = DATA_FILE

number_of_nodes  = read_excel_data(_path, "Nodes")[0]
cords            = read_excel_data(_path, "NodesCord")
cFix             = read_excel_data(_path, "FixedUnitCost")[0]
cVar             = read_excel_data(_path, "cvar(cijvar)")
cCom             = read_excel_data(_path, "com(cijom)")
cRev             = read_excel_data(_path, "crev(cijrev)")[0]
thetaFix         = read_excel_data(_path, "vfix(thetaijfix)")
thetaVar         = read_excel_data(_path, "vvar(thetaijvar)")
D                = read_excel_data(_path, "EdgesDemandPeak(i)")
cMax             = read_excel_data(_path, "Cmax(cijmax)")
alpha            = read_excel_data(_path, "Alpha")[0]
betta            = read_excel_data(_path, "Betta")[0]
lambda_          = read_excel_data(_path, "Lambda")[0]
surface          = read_excel_data(_path, "Surface(i)")
surface_cons     = read_excel_data(_path, "SurfaceConsumption(i)")
cGaz             = read_excel_data(_path, "cgaz")[0]
ENR_sourceMaxCap = read_excel_data(_path, "ENR_SourceMaxCap")[0]
CinvestENR       = read_excel_data(_path, "CinvestENR")[0]
CinvestGaz       = read_excel_data(_path, "CinvestGaz")[0]

demand = {}
for j in range(number_of_nodes + 10):
    demand[j] = surface[j] * surface_cons[j]

distances = {}
for i in range(number_of_nodes + 10):
    for j in range(number_of_nodes + 10):
        distances[(i, j)] = (
            (cords[(i, 0)] - cords[(j, 0)]) ** 2 +
            (cords[(i, 1)] - cords[(j, 1)]) ** 2
        ) ** 0.5


#  Main function

def optimize_for_given_source(source_id, P_installed, gaz_part_i):
    """
    Solve the MILP for a given source and installed power.
    source_id  : int (0-9), index of the source among the 10 candidate sources
    P_installed: float, installed power (kW)
    gaz_part_i : float (0-1), estimated gas share

    Returns a dict: status, total_cost, built, incoming_heat,
                    outcoming_heat, Q_deliverd, Power_renewable, Power_gaz
    """
    source_node = number_of_nodes + source_id
    nodes = list(range(number_of_nodes)) + [source_node]

    built         = LpVariable.dicts("built",         (nodes, nodes), cat="Binary")
    incoming_heat = LpVariable.dicts("incoming_heat", (nodes, nodes), lowBound=0, cat="Continuous")
    outcoming_heat= LpVariable.dicts("outcoming_heat",(nodes, nodes), lowBound=0, cat="Continuous")

    Q_deliverd      = LpVariable("Q_deliverd", lowBound=0, cat="Continuous")
    Power_renewable = min(P_installed, ENR_sourceMaxCap)
    Power_gaz       = LpVariable("Power_gaz",  lowBound=0, cat="Continuous")

    problem = LpProblem("MILP_heat_network", LpMinimize)

    # Total heat delivered over one year
    problem += Q_deliverd == lpSum(
        incoming_heat[source_node][j]
        for j in range(number_of_nodes)
    ) * 24 * 365.25

    # Gas power = total installed power - renewable power
    problem += Power_gaz == P_installed - Power_renewable

    # Subsidy: -50% on the investment if the gas share is <= 50%
    subvention = True if gaz_part_i <= 0.5 else False

    #  Objective function
    problem += (
        #  fixed cost + maintenance of the pipes
        lpSum(built[i][j] * distances[(i, j)] * (cCom[(i, j)] + alpha * cFix)
              for i in nodes for j in nodes if i != j)
        #  variable transport cost
        + lpSum(distances[(i, j)] * alpha * cVar[(i, j)] * incoming_heat[i][j]
                for i in nodes for j in nodes if i != j)
        #  revenue
        - lpSum(cRev * demand[j] * lambda_ * built[i][j]
                for i in nodes for j in nodes)
        #  investment cost (with the subsidy when it applies)
        + (Power_gaz * CinvestGaz + Power_renewable * CinvestENR) * (1 - 0.5 * subvention)
        #  cost of the gas energy
        + Q_deliverd * gaz_part_i * cGaz
    )

    #  Constraints

    # 1  tree structure: at most one parent per node
    for i in nodes:
        problem += lpSum(built[j][i] for j in nodes) <= 1

    # 2  a single flow direction between two nodes
    for i in nodes:
        for j in nodes:
            if i != j:
                problem += built[i][j] + built[j][i] <= 1

    # 3  demand satisfaction (heat balance on each pipe)
    for i in nodes:
        for j in nodes:
            problem += (
                (1 - thetaVar[(i, j)] * distances[(i, j)]) * incoming_heat[i][j]
                - (D[j] * betta * lambda_ + thetaFix[(i, j)] * distances[(i, j)]) * built[i][j]
                - outcoming_heat[i][j] == 0
            )

    # 4  energy conservation (at every node except the source)
    for j in range(number_of_nodes):
        problem += (
            lpSum(outcoming_heat[i][j] for i in nodes if i != j)
            - lpSum(incoming_heat[j][k] for k in nodes if k != j) == 0
        )

    # 5  capacity of each pipe
    for i in nodes:
        for j in nodes:
            if i != j:
                problem += incoming_heat[i][j] <= cMax[(i, j)] * built[i][j]

    # 6  no flow into the source
    problem += lpSum(
        built[j][source_node]
        for j in range(number_of_nodes)
    ) == 0

    # 7  equity constraint (minimum connected surface)
    gamma = 0.4
    total_surface = sum(surface[i] for i in range(number_of_nodes))
    problem += lpSum(
        surface[j] * built[i][j]
        for i in nodes for j in nodes if i != j
    ) >= gamma * total_surface

    # 8  no node connected to itself
    for i in nodes:
        problem += built[i][i] == 0
        problem += incoming_heat[i][i] == 0
        problem += outcoming_heat[i][i] == 0

    # 9  installed power of the source
    problem += lpSum(
        incoming_heat[source_node][j]
        for j in range(number_of_nodes)
    ) <= P_installed

    #  Solve (time limit of 120 s per call)
    problem.solve(PULP_CBC_CMD(timeLimit=120))

    status     = LpStatus[problem.status]
    total_cost = value(problem.objective)

    built_sol          = {}
    incoming_heat_sol  = {}
    outcoming_heat_sol = {}

    for i in range(number_of_nodes):
        built_sol[i]          = {}
        incoming_heat_sol[i]  = {}
        outcoming_heat_sol[i] = {}
        for j in range(number_of_nodes):
            built_sol[i][j]          = int(built[i][j].varValue)          if built[i][j].varValue          is not None else 0
            incoming_heat_sol[i][j]  = float(incoming_heat[i][j].varValue) if incoming_heat[i][j].varValue  is not None else 0.0
            outcoming_heat_sol[i][j] = float(outcoming_heat[i][j].varValue)if outcoming_heat[i][j].varValue is not None else 0.0

    # Also include the pipes leaving the source
    built_sol[source_node]          = {}
    incoming_heat_sol[source_node]  = {}
    outcoming_heat_sol[source_node] = {}
    for j in range(number_of_nodes):
        built_sol[source_node][j]          = int(built[source_node][j].varValue)           if built[source_node][j].varValue          is not None else 0
        incoming_heat_sol[source_node][j]  = float(incoming_heat[source_node][j].varValue)  if incoming_heat[source_node][j].varValue  is not None else 0.0
        outcoming_heat_sol[source_node][j] = float(outcoming_heat[source_node][j].varValue) if outcoming_heat[source_node][j].varValue is not None else 0.0

    write_solution(built_sol, incoming_heat_sol, outcoming_heat_sol, source_id)

    return {
        "status"          : status,
        "total_cost"      : total_cost,
        "built"           : built_sol,
        "incoming_heat"   : incoming_heat_sol,
        "outcoming_heat"  : outcoming_heat_sol,
        "Q_deliverd"      : Q_deliverd.varValue,
        "Power_renewable" : Power_renewable,
        "Power_gaz"       : value(Power_gaz),
    }
