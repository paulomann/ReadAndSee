from readorsee.data import facade

# Preprocess and stratify all data
# sf = facade.StratifyFacade("local_search")
# sf.stratify(days=[60, 212, 365])
# sf.save()

# Load and check data
# dl = facade.DataLoader()
# user_dfs = dl.get_participants_dataframes()
# raw_data = dl.load_raw_data()
# posts_dfs = dl.get_posts_dataframes()

ppf = facade.PreProcessFacade("twitter_external", save=False)
ppf.process_pipeline()
