from readorsee.data import facade

sf = facade.StratifyFacade()
sf.stratify(days=[60, 212, 365])
saved_data = sf.save()
