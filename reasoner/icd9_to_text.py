from icd9cms.icd9 import search

def get_description(icd9):
    icd9 = search(icd9)
    if icd9:
        return icd9.short_desc
    else:
        return None