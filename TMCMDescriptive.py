import pandas as pd

# This function performs the descriptive calculations according to the components of mode A (separated by positive and negative loadings) in the real case study.
def descriptive():
    df = pd.read_csv(r"MatrixABE.csv", 
                     sep=";", index_col=0, decimal=".")
    comps = ["DC1-A","DC2-A","DC3-A","DC4-A","DC5-A"]
    results = {}
    for age_group in [1, 2, 3]:
        for comp in comps:
            positives = ((df["age"] == age_group) & (df[comp] > 0)).sum()
            negatives = ((df["age"] == age_group) & (df[comp] < 0)).sum()
            results[f"Age{age_group}_{comp}_Positives"] = positives
            results[f"Age{age_group}_{comp}_Negatives"] = negatives
    for gender_group in [1, 2]:
        for comp in comps:
            positives = ((df["gender"] == gender_group) & (df[comp] > 0)).sum()
            negatives = ((df["gender"] == gender_group) & (df[comp] < 0)).sum()
            results[f"Gender{gender_group}_{comp}_Positives"] = positives
            results[f"Gender{gender_group}_{comp}_Negatives"] = negatives
    for level in [1, 2, 3, 4, 5]:
        for comp in comps:
            positives = ((df["academic_level"] == level) & (df[comp] > 0)).sum()
            negatives = ((df["academic_level"] == level) & (df[comp] < 0)).sum()
            results[f"Academic{level}_{comp}_Positives"] = positives
            results[f"Academic{level}_{comp}_Negatives"] = negatives
    for comp in comps:
        positives = (df[comp] > 0).sum()
        negatives = (df[comp] < 0).sum()
        zeros = (df[comp] == 0).sum()
        results[f"{comp}_Total_Positives"] = positives
        results[f"{comp}_Total_Negatives"] = negatives
        results[f"{comp}_Total_Zeros"] = zeros
    return results

results = descriptive()
for key, value in results.items():
    print(key, ":", value)
