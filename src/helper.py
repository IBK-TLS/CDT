def convert_label_RCIS(label):
    dict_label = {
                    "100.0" : "H",
                    "75.0"  : "MH",
                    "50.0"  : "ML",
                    "25.0"  : "L",
		    "0"     : "0"	
            }
    # ex label = ["PN_100.0_50.0"]
    list_lab = label.split("_") # ["PN", "100.0", "50.0"]
    
    #new_list_lab = [list_lab[0], dict_label[list_lab[1]], dict_label[list_lab[2]] ]
    new_list_lab = [list_lab[0]]
    
    if list_lab[0] in ["PN", "SCN", "VN"]:
        new_list_lab.append("-"+dict_label[list_lab[1]])
    else:
        new_list_lab.append(dict_label[list_lab[1]])

    if list_lab[0] in ["PN", "ECN", "VN"]:
        new_list_lab.append("-"+dict_label[list_lab[2]])
    else:
        new_list_lab.append(dict_label[list_lab[2]])
    
    return new_list_lab[0]+"("+new_list_lab[1]+","+new_list_lab[2]+")"

labels = ["PN_100.0_50.0"] # -> PN_-H_-ML
labels += ["SCN_100.0_0"] # -> SCN_-H_0
labels += ["ECN_0_25.0"] # -> ECN_0_-L
labels += ["VN_50.0_25.0"] # -> VN_-ML_-L
labels += ["VN_50.0_25.0"] # -> VN_-ML_-L
labels += ["VN_50.0_25.0"] # -> VN_-ML_-L

test = [convert_label_RCIS(label) for label in labels]

print(test)


    
