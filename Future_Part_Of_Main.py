import glob
from Drought_Warbler_Combiner import DroughtAndWarblerCombiner
from Drought_Data_Combiner import DroughtDataCombiner
from Warbler_Data_Combiner import WarblerDataCombiner

path_warblers = r"Data/eBird_Warbler_Files_CSV"
all_files_warblers = glob.glob(path_warblers + "/*.csv")
path_drought = r"Data/Drought_Data_CSV"
all_files_drought = glob.glob(path_drought + "/*.csv")

WarblerDataCombiner(all_files_warblers)
DroughtDataCombiner(all_files_drought)

drought_file = "Data/Combined_Drought_CSV/02_to_19_Drought_Data.csv"
warbler_file = "Data/Combined_Warbler_CSV/02_to_19_Warbler_Data.csv"
DroughtAndWarblerCombiner(drought_file, warbler_file)