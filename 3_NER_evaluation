import json
import pandas as pd
import numpy as np

gold_path = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Research Desk/2024MedNER/model_lab/Result_strict_partial_calculator/data/validate_result/true.json'
pred_path = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Research Desk/2024MedNER/model_lab/Result_strict_partial_calculator/data/validate_result/pred_3.json'

def combine_and_replace_entities(df, suffixes):
    new_data = []
    i = 0
    while i < len(df):
        if any(df.iloc[i]['entity_type'].endswith(suffix) for suffix in suffixes):
            entity_suffix = df.iloc[i]['entity_type'][-3:]  # Get the suffix (e.g., TIM, SUB)
            combined_string = df.iloc[i]['surface_string']
            j = i + 1
            while j < len(df) and df.iloc[j]['entity_type'].endswith(entity_suffix):
                combined_string += ' ' + df.iloc[j]['surface_string']
                j += 1
            new_data.append([entity_suffix, combined_string])
            # Add NaN rows for the I- entities
            for k in range(i + 1, j):
                new_data.append([np.nan, np.nan])
            i = j
        else:
            if df.iloc[i]['entity_type'].startswith('I-'):
                new_data.append([np.nan, np.nan])
            else:
                new_data.append([df.iloc[i]['entity_type'], df.iloc[i]['surface_string']])
            i += 1
    return pd.DataFrame(new_data, columns=["entity_type", "surface_string"])

def load_json(file_path):
    # Load the JSON data
    with open(str(file_path), 'r') as file:
        data = json.load(file)
    return data

def custom_algorithm(row):
    if pd.isna(row['entity_type_gold']) and pd.isna(row['surface_string_gold']) and pd.isna(row['entity_type_pred']) and pd.isna(row['surface_string_pred']):
        row['strict'] = np.nan
        row['exact'] = np.nan
        row['partial'] = np.nan
        row['type'] = np.nan
    elif pd.isna(row['entity_type_gold']) and pd.isna(row['surface_string_gold']):
        row['strict'] = 'SPU'
        row['exact'] = 'SPU'
        row['partial'] = 'SPU'
        row['type'] = 'SPU'
    elif pd.isna(row['entity_type_pred']) and pd.isna(row['surface_string_pred']):
        row['strict'] = 'MIS'
        row['exact'] = 'MIS'
        row['partial'] = 'MIS'
        row['type'] = 'MIS'
    elif row['entity_type_gold'] == row['entity_type_pred'] and row['surface_string_gold'] == row['surface_string_pred']:
        row['strict'] = 'COR'
        row['exact'] = 'COR'
        row['partial'] = 'COR'
        row['type'] = 'COR'
    elif row['entity_type_gold'] == row['entity_type_pred']:
        row['strict'] = 'INC'
        row['exact'] = 'INC'
        row['type'] = 'COR'
        if isinstance(row['surface_string_gold'], str) and isinstance(row['surface_string_pred'], str) and (row['surface_string_gold'] in row['surface_string_pred'] or row['surface_string_pred'] in row['surface_string_gold']):
            row['partial'] = 'PAR'
        else:
            row['partial'] = 'INC'
    elif row['surface_string_gold'] == row['surface_string_pred']:
        row['strict'] = 'INC'
        row['partial'] = 'COR'
        row['exact'] = 'COR'
        if row['entity_type_gold'].lower() == row['entity_type_pred'].lower():
            row['type'] = 'COR'
        else:
            row['type'] = 'INC'
    else:
        row['strict'] = 'INC'
        row['exact'] = 'INC'
        row['type'] = 'INC'
        if isinstance(row['surface_string_gold'], str) and isinstance(row['surface_string_pred'], str) and (row['surface_string_gold'] in row['surface_string_pred'] or row['surface_string_pred'] in row['surface_string_gold']):
            row['partial'] = 'PAR'
        else: 
            row['partial'] = 'INC'
    
    return row

# Load JSON data
gold_json = load_json(gold_path)
pred_json = load_json(pred_path)

suffixes = ['TIM', 'SUB', 'ROU', 'UNI', 'MEAS', 'DEC', 'RTIM']
combined_results = []

# Iterate over all indexes
for index in range(len(gold_json)):
    df_gold = pd.DataFrame(gold_json[index], columns=["entity_type", "surface_string"])
    df_pred = pd.DataFrame(pred_json[index], columns=["entity_type", "surface_string"])

    gold_combined = combine_and_replace_entities(df_gold, suffixes)
    pred_combined = combine_and_replace_entities(df_pred, suffixes)

    result = pd.merge(gold_combined, pred_combined, how='left', left_index=True, right_index=True, suffixes=('_gold', '_pred'))
    result['type'] = None
    result['partial'] = None
    result['exact'] = None
    result['strict'] = None

    result = result.apply(custom_algorithm, axis=1)
    combined_results.append(result)

# Concatenate all results into a single DataFrame
final_result = pd.concat(combined_results, ignore_index=True)

# Display the final DataFrame
print(final_result.head(20))

# Save the final DataFrame to a file
#final_result.to_csv('/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Research Desk/2024MedNER/model_lab/Result_strict_partial_calculator/data/final_result.csv', index=False)

def count_measures_by_entity(df):
    measures = ['COR', 'INC', 'PAR', 'MIS', 'SPU']
    types = ['type', 'partial', 'exact', 'strict']
    entity_types = df['entity_type_gold'].dropna().unique()
    
    # Initialize a list to store the counts
    counts_list = []
    
    # Iterate over each entity type and count occurrences
    for entity in entity_types:
        entity_df = df[df['entity_type_gold'] == entity]
        counts = {measure: {measure_type: 0 for measure_type in types} for measure in measures}
        for _, row in entity_df.iterrows():
            for measure in measures:
                for measure_type in types:
                    if row[measure_type] == measure:
                        counts[measure][measure_type] += 1
        # Convert the dictionary to a DataFrame and add entity type as a column
        count_df = pd.DataFrame(counts).transpose().reset_index().rename(columns={'index': 'Measure'})
        count_df.columns = ['Measure', 'Type', 'Partial', 'Exact', 'Strict']
        count_df['entity_type'] = entity
        counts_list.append(count_df)
    
    # Concatenate all the counts into a single DataFrame
    final_counts_df = pd.concat(counts_list, ignore_index=True)
    return final_counts_df

# Count the measures by entity type in the final result DataFrame
entity_result_counts = count_measures_by_entity(final_result)
entity_result_counts['entity_type'] = entity_result_counts['entity_type'].str.replace('EAS', 'MEA')
entity_result_counts
#entity_result_counts.to_csv('/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Research Desk/2024MedNER/model_lab/Result_strict_partial_calculator/data/entity_result_counts.csv', index=False)

def calculate_metrics(df):
    metrics = []
    for entity_type in df['entity_type'].unique():
        entity_df = df[df['entity_type'] == entity_type]
        COR = entity_df.loc[entity_df['Measure'] == 'COR', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
        INC = entity_df.loc[entity_df['Measure'] == 'INC', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
        PAR = entity_df.loc[entity_df['Measure'] == 'PAR', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
        MIS = entity_df.loc[entity_df['Measure'] == 'MIS', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
        SPU = entity_df.loc[entity_df['Measure'] == 'SPU', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
        
        POS = COR + INC + PAR + MIS
        ACT = COR + INC + PAR + SPU

        exact_precision = COR / ACT if ACT else 0
        exact_recall = COR / POS if POS else 0
        exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if (exact_precision + exact_recall) else 0

        partial_precision = (COR + 0.5 * PAR) / ACT if ACT else 0
        partial_recall = (COR + 0.5 * PAR) / POS if POS else 0
        partial_f1 = 2 * partial_precision * partial_recall / (partial_precision + partial_recall) if (partial_precision + partial_recall) else 0
        
        metrics.append({
            'entity_type': entity_type,
            'POS': POS,
            'ACT': ACT,
            'Exact_Precision': exact_precision,
            'Exact_Recall': exact_recall,
            'Exact_F1': exact_f1,
            'Partial_Precision': partial_precision,
            'Partial_Recall': partial_recall,
            'Partial_F1': partial_f1
        })

    # Calculate overall metrics
    overall_COR = df.loc[df['Measure'] == 'COR', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
    overall_INC = df.loc[df['Measure'] == 'INC', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
    overall_PAR = df.loc[df['Measure'] == 'PAR', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
    overall_MIS = df.loc[df['Measure'] == 'MIS', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
    overall_SPU = df.loc[df['Measure'] == 'SPU', ['Type', 'Partial', 'Exact', 'Strict']].sum().sum()/4
    
    overall_POS = overall_COR + overall_INC + overall_PAR + overall_MIS
    overall_ACT = overall_COR + overall_INC + overall_PAR + overall_SPU

    overall_exact_precision = overall_COR / overall_ACT if overall_ACT else 0
    overall_exact_recall = overall_COR / overall_POS if overall_POS else 0
    overall_exact_f1 = 2 * overall_exact_precision * overall_exact_recall / (overall_exact_precision + overall_exact_recall) if (overall_exact_precision + overall_exact_recall) else 0

    overall_partial_precision = (overall_COR + 0.5 * overall_PAR) / overall_ACT if overall_ACT else 0
    overall_partial_recall = (overall_COR + 0.5 * overall_PAR) / overall_POS if overall_POS else 0
    overall_partial_f1 = 2 * overall_partial_precision * overall_partial_recall / (overall_partial_precision + overall_partial_recall) if (overall_partial_precision + overall_partial_recall) else 0

    metrics.append({
        'entity_type': 'Overall',
        'POS': overall_POS,
        'ACT': overall_ACT,
        'Exact_Precision': overall_exact_precision,
        'Exact_Recall': overall_exact_recall,
        'Exact_F1': overall_exact_f1,
        'Partial_Precision': overall_partial_precision,
        'Partial_Recall': overall_partial_recall,
        'Partial_F1': overall_partial_f1
    })

    metrics_df = pd.DataFrame(metrics)
    return metrics_df

metrics_df = calculate_metrics(entity_result_counts)
print(metrics_df)
