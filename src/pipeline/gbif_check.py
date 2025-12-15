

import pandas as pd
from pygbif import species


def verif(names_list) :
    correct_occurences = {}
    misspelled_ocrurrences = []
    wrong_occurences = []
    # We initialize one dictionnary for the correct specie names with
    # the gbif link, and two other lists for variant spellings and
    # false occurences


    # On itère sur toute la liste de noms et on vérifie le status de
    # la recherche pour décider dans quelle liste ajouter le nom
    for name in names_list : 
        res = species.name_backbone(scientificName=name)
        if res["diagnostics"]["matchType"] == "EXACT" :
            correct_occurences[res["usage"]["canonicalName"]] = f"GBIF taxonomy page - https://www.gbif.org/species/{res["usage"]["key"]}"
        elif res["diagnostics"]["matchType"] == "VARIANT" :
            misspelled_ocrurrences.append([name, res["usage"]["canonicalName"], f"GBIF taxonomy page - https://www.gbif.org/species/{res["usage"]["key"]}"])
        else : 
            wrong_occurences.append(name)

    return(correct_occurences, misspelled_ocrurrences, wrong_occurences)



def result_csv(csv_path) :
    csv = pd.read_csv(csv_path)
    csv["Accepted Names"] = "couldn't find"
    csv["Misspelled Names"] = ""
    csv["Unrecognised Names"] = ""
    length = len(csv)
    for i in range(length) : 
        if csv.at(i, "Taxon") != "couldn't find" : 
            a, b, c = verif(csv.at(i, "Taxon"))
            csv.at[i, "Accepted Names"] = a
            csv.ati[i, "Misspelled Names"] = b
            csv.ati[i, "Unrecognised Names"] = c
    csv.to_csv("result_checked.csv")