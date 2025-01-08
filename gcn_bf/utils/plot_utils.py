import matplotlib.pyplot as plt
import pandas as pd

def boxplot_delta_e(delta_e, ind, ind_class='secondary'):
    delta_e_flattened = []
    ind_flattened = []

    # We flatten here
    for delta_e_sublist, ind_sublist in zip(delta_e, ind):
        if len(delta_e_sublist) == len(ind_sublist):
            delta_e_flattened.extend(delta_e_sublist)
            ind_flattened.extend(ind_sublist)
    
    if ind_class == 'cdr_status':
        ind_flattened = [1 if sec in [3, 4, 5] else 0 for sec in ind_flattened]  

    data = pd.DataFrame({'Class': ind_flattened, 'Value': delta_e_flattened})
    xlabel = ''

    plt.figure()
    if ind_class == 'secondary':
        data['Secondary'] = data['Class'].apply(lambda x: 'Strand (FR)' if x in [1] else 'Irregular (FR)' if x in [2] else 'Helix (FR)' if x in [0] else 'Strand (CDR)' if x in [4] else 'Irregular (CDR)' if x in [5] else 'Helix (CDR)')
        data.boxplot(column='Value', by=ind_class.title(), grid=False, showmeans=True, showfliers=False)
    elif ind_class == 'cdr_status':
        data['CDR status'] = data['Class'].apply(lambda x: 'FR' if x == 0 else 'CDR')
        data.boxplot(column='Value', by='CDR status', grid=False, showmeans=True, showfliers=False)
    elif ind_class in ['epitope', 'paratope']:
        data = data[data['Class'].isin([0, 1])]
        data[ind_class.title()+' membership'] = data['Class'].apply(lambda x: ind_class.title() if x == 1 else 'Non-'+ind_class)
        data.boxplot(column='Value', by=ind_class.title()+' membership', grid=False, showmeans=True, showfliers=False)
    elif ind_class == 'entropy':
        bins = [0, 1, 2, float('inf')] # binning
        labels = ['0-1', '1-2', '>2']
        xlabel = 'Entropy bins'
        ds_bin = pd.cut(ind_flattened, bins=bins, labels=labels, right=False)
        bin_counts = ds_bin.value_counts().sort_index()
        print(bin_counts)

        grouped_delta_e = {label: [] for label in labels}
        for value, bin_label in zip(delta_e_flattened, ds_bin):
            if pd.notna(bin_label): 
                grouped_delta_e[bin_label].append(value)
        plt.boxplot([grouped_delta_e[label] for label in labels], labels=labels, patch_artist=True, showfliers=False, showmeans=True)

    plt.title('')  
    plt.xlabel(xlabel)
    plt.ylabel('$\Delta$e')
    plt.show()

def plot_consecutive_secondary(delta_e, secondary, secondary_type='helix'):
    delta_e_flattened = []
    secondary_flattened = []

    # We flatten here
    for delta_e_sublist, secondary_sublist in zip(delta_e, secondary):
        if len(delta_e_sublist) == len(secondary_sublist):
            delta_e_flattened.extend(delta_e_sublist)
            secondary_flattened.extend(secondary_sublist)
    
    cdr_status = [1 if sec in [3, 4, 5] else 0 for sec in secondary_flattened]  

    def count_sequences_and_average_delta_e(lst, cdr_status, delta_e_flattened, M, valid_values, processed_indices):
        counts = {'FR': 0, 'CDR': 0, 'Total': 0}
        delta_e_sums = {'FR': 0, 'CDR': 0, 'Total': 0}
        delta_e_counts = {'FR': 0, 'CDR': 0, 'Total': 0}

        i = 0
        while i <= len(lst) - M:
            segment = lst[i:i+M]
            if all(val in valid_values for val in segment) and all(idx not in processed_indices for idx in range(i, i+M)):
                region_type = 'FR' if cdr_status[i] == 0 else 'CDR'
                counts[region_type] += 1
                counts['Total'] += 1

                delta_e_segment = delta_e_flattened[i:i+M]
                delta_e_sums[region_type] += sum(delta_e_segment)
                delta_e_sums['Total'] += sum(delta_e_segment)
                delta_e_counts[region_type] += M
                delta_e_counts['Total'] += M

                processed_indices.update(range(i, i+M))
                i += M
            else:
                i += 1

        avg_delta_e = {
            'FR': delta_e_sums['FR'] / delta_e_counts['FR'] if delta_e_counts['FR'] > 0 else 0,
            'CDR': delta_e_sums['CDR'] / delta_e_counts['CDR'] if delta_e_counts['CDR'] > 0 else 0,
            'Total': delta_e_sums['Total'] / delta_e_counts['Total'] if delta_e_counts['Total'] > 0 else 0,
        }

        return counts, avg_delta_e

    consec_counts = []
    avg_delta_e_per_M = []
    if secondary_type == 'helix':
        valid_values = {0, 3}
    elif secondary_type == 'strand':
        valid_values = {1, 4}
    else:
        valid_values = {2, 5}
    processed_indices = set()

    for M in range(12, 0, -1):
        counts, avg_delta_e = count_sequences_and_average_delta_e(
            secondary_flattened, cdr_status, delta_e_flattened, M, valid_values, processed_indices
        )
        consec_counts.append(counts)
        avg_delta_e_per_M.append(avg_delta_e)

        print(f"{M} consecutive {secondary_type} - FR: {counts['FR']}, CDR: {counts['CDR']}, Total: {counts['Total']}")
        print(f"Average delta_e for {M} consecutive {secondary_type} - FR: {avg_delta_e['FR']}, CDR: {avg_delta_e['CDR']}, Total: {avg_delta_e['Total']}")

    fr_counts = [counts['FR'] for counts in consec_counts[::-1]]
    cdr_counts = [counts['CDR'] for counts in consec_counts[::-1]]
    total_counts = [counts['Total'] for counts in consec_counts[::-1]]

    plt.plot(range(1, 13), fr_counts, 'o-', label='FR', c='green')
    plt.plot(range(1, 13), cdr_counts, 'o-', label='CDR', c='red')
    plt.plot(range(1, 13), total_counts, 'o-', label='Total', c='blue')
    plt.xlabel(f'Consecutive {secondary_type} (M)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

    fr_avg_delta_e = [avg['FR'] for avg in avg_delta_e_per_M[::-1]]
    cdr_avg_delta_e = [avg['CDR'] for avg in avg_delta_e_per_M[::-1]]
    total_avg_delta_e = [avg['Total'] for avg in avg_delta_e_per_M[::-1]]

    plt.plot(range(1, 13), fr_avg_delta_e, 'o-', label='FR', c='green')
    plt.plot(range(1, 13), cdr_avg_delta_e, 'o-', label='CDR', c='red')
    plt.plot(range(1, 13), total_avg_delta_e, 'o-', label='Total', c='blue')
    plt.xlabel(f'Consecutive {secondary_type} (M)')
    plt.ylabel(r'Average $\Delta e$')
    plt.legend()
    plt.show()


def plot_prediction_errors(mse, mse_seq, residue_ids):
    cdr_positions = [residue_ids.index(el) for el in ['26H', '32H', '52H', '56H', '95H', '102H']] + [residue_ids.index(el) for el in ['24L', '34L', '50L', '56L', '89L', '97L']]

    plt.plot(range(len(mse_seq)), mse_seq, marker='o', linestyle='-', label='Sequence only')
    plt.plot(range(len(mse)), mse, marker='o', linestyle='-', label='Sequence and graph')
    plt.axvspan(residue_ids.index('111H'), residue_ids.index('1L'), color='black', linestyle='--', linewidth=0.2)
    plt.legend()
    for i in range(len(cdr_positions)//2):
        plt.axvspan(cdr_positions[2*i], cdr_positions[2*i+1], alpha=0.1, color='green')

    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Residue index', fontsize=14)
    plt.show()