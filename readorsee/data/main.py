from readorsee.data import facade

# Preprocess and stratify all data
# sf = facade.StratifyFacade("local_search")
# sf.stratify(days=[60, 212, 365])
# sf.save()

# Load and check data
mf = facade.DataFacade()
dfs = mf.get_participants_dataframes()
data = facade.StratifyFacade("").load_stratified_data()
