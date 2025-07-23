import pandas as pd
from functools import reduce
from sailor.Planner.sailor_planner.gcp import V100, P100, T4, N1_STD, N1_HM, N1_HC, A100_40, A100_80, L4_G2, CARTESIAN_PRODUCT


def generate_dataset():
    '''
    Generate a dataframe with all the possible combinations of GPU and CPU from GCP.
    The values are saved in constants module and imported from Google Cloud Platform.
    The information is saved in a pandas dataframe and the columns are:
        'num_GPUs', 'gpu_memory', 'gpu_memory_unit', 'gpu_memory_bandwidth',
       'gpu_memory_bandwidth_unit', 'throughput_FP32', 'throughput_FP16_32',
       'throughput_unit', 'NVLink', 'NVLink_bandwidth', 'NVLink_unit',
       'compute_requirement', 'max_network_egress', 'cost_unit', 'zone',
       'time_input_data', 'num_CPUs', 'cpu_memory', 'cpu_memory_unit',
       'network_bandwidth', 'network_bandwidth_unit', 'type', 'name',
       'cost_per_hour', 'cost_per_hour_preemptible', 'memory_bandwidth_unit
    '''
    # Generate cartesian product for non-fixed combinations in GCP
    df_gpus = [pd.DataFrame(gpu) for gpu in [V100, P100, T4]]
    df_cpus = [pd.DataFrame(cpu) for cpu in [N1_STD, N1_HM, N1_HC]]
    gpus = pd.concat(list(df_gpus))
    cpus = pd.concat(list(df_cpus))
    cp = pd.merge(gpus, cpus, how='cross')

    # Constraint the combinations to the ones that are possible in GCP
    filtered_rows = []
    for _, row in cp.iterrows():
        if row['name_y'] in CARTESIAN_PRODUCT.get(row['name_x'], []):
            filtered_rows.append(row)
    cp = pd.DataFrame(filtered_rows)

    # Filter the duplicated columns and treat them
    CONSTANT_COLUMN = ['cost_unit', 'zone', 'time_input_data']
    SUM_COLUMN = ['type', 'name', 'cost_per_hour', 'cost_per_hour_preemptible']
    for d in CONSTANT_COLUMN:
        cp.drop(d+'_y', axis=1, inplace=True)
        cp.rename(columns={d+'_x': d}, inplace=True)
    for d in SUM_COLUMN:
        if d in ('type', 'name'):
            cp[d] = cp[d+'_x'] + '-' + cp[d+'_y']
        if d in ('cost_per_hour', 'cost_per_hour_preemptible'):
            cp[d] = cp[d+'_x'] + cp[d+'_y']
        cp.drop(d+'_y', axis=1, inplace=True)
        cp.drop(d+'_x', axis=1, inplace=True)
    cp.keys()

    # Generate fixed combinations. IN GCP the A100 and L4 GPUs are already assigned a fixed CPU characteristics.
    combi = [pd.DataFrame(c) for c in [A100_40, A100_80, L4_G2]]
    fixed_combi = pd.concat(list(combi))

    gcp_dataset = pd.concat([cp, fixed_combi])

    return gcp_dataset


def print_help_dataframe(DATASET):
    '''
    Helper function to print the columns and values of the dataset.
    '''
    print("\n")
    print("-------- STARTING HELP DATASET ---------")
    print("The data is extracted from GCP and cost values are from 12-2023 from zone us-west1")
    COLUMNS_TO_FILTER_BASIC = ['num_GPUs', 'gpu_memory', 'num_CPUs',
                               'cpu_memory', 'name', 'cost_per_hour', 'cost_per_hour_preemptible']
    COLUMNS_TO_FILTER_ADVANCED = ['gpu_memory_badwidth', 'throughput_FP32', 'throughput_FP16_32',
                                  'NVLink', 'NVLink_bandwidth', 'max_network_egress', 'network_bandwidth']
    print("The following columns are basic filters: ", COLUMNS_TO_FILTER_BASIC)
    for basic in COLUMNS_TO_FILTER_BASIC:
        if basic == 'name':
            print(
                f"    - Column {basic} contains values: {set(DATASET[basic])}")
        else:
            print(
                f"    - Column {basic} contains values between: {min(DATASET[basic])} and {max(DATASET[basic])}")
    print("\n")
    print("The following columns are advanced filters: ",
          COLUMNS_TO_FILTER_ADVANCED)
    print("-------- FINISHING HELP DATASET ---------")
    print("\n")


EXAMPLE_FILTER_V100 = {
    'name': ('contains', ['V100', 'n1-standard']),
    'num_GPUs': ('==', [1]),
    'gpu_memory': ('>=', [16]),
    'num_CPUs': ('==', [8]),
}

EXAMPLE_FILTER_V100_A100_1GPU = {
    'name_GPU': ('OR', ['V100', 'A100-40', 'A100-80']),
    'name': ('OR', ['n1-standard', 'a2']),
    'num_GPUs': ('==', [1]),
    'gpu_memory': ('>=', [16]),
    'num_CPUs': ('>=', [8]),
}

EXAMPLE_FILTER_V100_A100_40_1GPU = {
    'name_GPU': ('OR', ['V100', 'A100-40',]),
    'name': ('OR', ['n1-standard', 'a2']),
    'gpu_memory': ('>=', [16]),
    'num_GPUs': ('==', [1]),
    'num_CPUs': ('>=', [8]),
    'num_CPUs': ('<', [64]),
}

EXAMPLE_FILTER_V100_A100 = {
    'name_GPU': ('OR', ['V100', 'A100-40', 'A100-80']),
    'name': ('OR', ['n1-standard', 'a2']),
    'gpu_memory': ('>=', [16]),
    'num_GPUs': ('==', [1]),
    'num_CPUs': ('>=', [8]),
    'num_CPUs': ('<', [64]),
}

COLUMNS_OPTIMIZATION = [
    'name_GPU', 'num_GPUs', 'gpu_memory', 'gpu_memory_bandwidth',
    'num_CPUs', 'cpu_memory',
    'network_bandwidth', 'name',
    'cost_per_hour', 'cost_per_hour_preemptible'
]


def add_cpu_combinations(combinations, version='full'):
    '''
        There is a training setup where on-demand VMs only have CPUs and spot VMs only have GPUs (strategy 3).
        We can consider the training combinations fron GCP with any hardware combination of CPU.
        We modify the dataset to include them:
    '''
    if version == 'full':
        cpus_df = [pd.DataFrame(df) for df in [N1_STD, N1_HM, N1_HC]]
        cpus = pd.concat([cpu for cpu in cpus_df])
        gpu_cpus = pd.merge(combinations, cpus, how='cross', suffixes=('_spot', '_dem'))

        # We remove the columns that refer to CPUs and on-demand costs that come from the training dataset,
        # and the columns that refer to spot costs that come from the CPU dataset.
        DELETE_COLUMNS = ['num_CPUs_spot', 'cpu_memory_spot', 'cost_per_hour_spot', 'cost_per_hour_preemptible_dem']
        for col in DELETE_COLUMNS:
            gpu_cpus.drop(col, axis=1, inplace=True)

        # We rename the columns of training to the normal names
        RENAME_SPOT = ['network_bandwidth_spot', 'cost_per_hour_spot', 'cost_per_hour_preemptible_spot']
        for col in RENAME_SPOT:
            gpu_cpus.rename(columns={col: col.replace('_spot', '')}, inplace=True)

        # We rename the columns of on-demand CPUs to the normal names
        RENAME_DEMAND = ['num_CPUs_dem', 'cpu_memory_dem', 'cost_per_hour_dem']
        for col in RENAME_DEMAND:
            gpu_cpus.rename(columns={col: col.replace('_dem', '')}, inplace=True)

        # Rename the name column to include all information
        gpu_cpus['name'] = gpu_cpus['name'+'_spot'] + '-' + gpu_cpus['name'+'_dem']
        gpu_cpus.drop('name_spot', axis=1, inplace=True)
        gpu_cpus.drop('name_dem', axis=1, inplace=True)

    elif version == 'simple':
        # TODO: add fixed combinations for 1 type of hardware
        pass
    else:
        raise ValueError(f"Version {version} not implemented")

    print(f"Final dataset has {len(gpu_cpus)} combinations")
    return gpu_cpus


def filter_dataframe(filters, original_dataframe):
    """
    Filter the original DataFrame based on the provided column names and filtered values.

    Parameters:
    - filters: A dictionary of list of tuples where each kye is a column and
               each tuple contains an operator with the corresponding filtered value.
    - original_dataframe: The original DataFrame to be filtered.

    Returns:
    - filtered_dataframe: The DataFrame after applying the filters.
    """
    # TODO: add more filter
    filtered_dataframe = original_dataframe.copy()

    for column, (operator, values) in filters.items():
        if operator == '==':
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column].isin(
                values)]
        elif operator == 'contains':  # AND
            condition = reduce(lambda x, y: x & y, [
                               filtered_dataframe[column].str.contains(val) for val in values])
            filtered_dataframe = filtered_dataframe[condition]
        elif operator == '>=':
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column] >= values[0]]
        elif operator == '<':
            filtered_dataframe = filtered_dataframe[filtered_dataframe[column] < values[0]]
        elif operator == 'OR':
            condition = reduce(lambda x, y: x | y, [
                               filtered_dataframe[column].str.contains(val) for val in values])
            filtered_dataframe = filtered_dataframe[condition]
        else:
            raise ValueError(f"Operator {operator} not implemented")
    filtered_dataframe.reset_index(drop=True, inplace=True)
    return filtered_dataframe
