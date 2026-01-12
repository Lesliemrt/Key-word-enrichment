import pandas as pd
from pygbif import species
import time

def verif(names_list) :
    correct_occurences = {}
    misspelled_occurences = []
    wrong_occurences = []
    
    for name in names_list :
        # print(f"Testing {name}")
        time.sleep(0.1)
        
        try :
            res = species.name_backbone(scientificName=name)
            
            if "diagnostics" in res and "matchType" in res["diagnostics"] :
                if res["diagnostics"]["matchType"] == "EXACT" :
                    correct_occurences[res["usage"]["canonicalName"]] = f"https://www.gbif.org/species/{res['usage']['key']}"
                elif res["diagnostics"]["matchType"] == "VARIANT" :
                    misspelled_occurences.append([name, res["usage"]["canonicalName"], f"https://www.gbif.org/species/{res['usage']['key']}"])
                else :
                    wrong_occurences.append(name)
            else :
                wrong_occurences.append(name)
                
        except Exception as e :
            print(f"Error with {name}: {e}")
            wrong_occurences.append(name)
    
    return correct_occurences, misspelled_occurences, wrong_occurences

def result_csv(df) :
    # csv = pd.read_csv(csv_path)
    csv = df
    csv["Accepted Names"] = ""
    csv["Misspelled Names"] = ""
    csv["Unrecognised Names"] = ""
    
    length = len(csv)
    print(f"Processing {length} records")
    
    for i in range(length) :
        extracted_value = csv.at[i, "Extracted"]
        # Check if the value is not NaN, empty, or "None"
        if pd.notna(extracted_value) and str(extracted_value).strip() != "" and str(extracted_value) != "None" :
            strings = [x.strip() for x in str(extracted_value).split(",")]
            a, b, c = verif(strings)
            csv.at[i, "Accepted Names"] = str(a)
            csv.at[i, "Misspelled Names"] = str(b)
            csv.at[i, "Unrecognised Names"] = str(c)
        
        if (i + 1) % 10 == 0 :
            print(f"Progress: {i + 1}/{length}")
    
    accepted_count = sum(1 for i in range(length) if csv.at[i, "Accepted Names"] != "")
    misspelled_count = sum(1 for i in range(length) if csv.at[i, "Misspelled Names"] != "[]")
    unrecognised_count = sum(1 for i in range(length) if csv.at[i, "Unrecognised Names"] != "[]")
    
    print(f"Accepted species: {accepted_count}")
    print(f"Misspelled species: {misspelled_count}")
    print(f"Unrecognised species: {unrecognised_count}")
    return csv
    

if __name__ == "__main__" :
    try :
        result_csv("species_extraction_results.csv")
    except Exception as e :
        print(f"Error: {e}")
        print("Make sure species_extraction_results.csv exists")