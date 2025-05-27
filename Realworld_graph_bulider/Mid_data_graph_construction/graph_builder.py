"""
Script for processing the MID dataset and generating graph or hypergraph edgelist files.
"""
import pandas as pd
import os

# The data files should be stored in these given locations.
COUNTRY_CODES_FILENAME = "adta/COW-country-codes.csv"
DYADIC_MID_FILENAME = "data/dyadic_mid_4.01.csv"
GRAPHS_DIRECTORY = "graphs"


def override_data_location(country_codes_filename, mid_filename, graphs_directory):
    global COUNTRY_CODES_FILENAME, DYADIC_MID_FILENAME, GRAPHS_DIRECTORY
    COUNTRY_CODES_FILENAME = country_codes_filename
    DYADIC_MID_FILENAME = mid_filename
    GRAPHS_DIRECTORY = graphs_directory


def edgelist_filename(start_year, end_year):
    return f"{GRAPHS_DIRECTORY}/dyadic_mid_{start_year}_{end_year}.edgelist"


def vertex_filename(start_year, end_year):
    return f"{GRAPHS_DIRECTORY}/dyadic_mid_{start_year}_{end_year}.vertices"


def build_edge_weights(mid_dataframe):
    """
    Given a dataframe containing the MID data, construct a dictionary of edge weights.

    :param mid_dataframe:
    :return:
    """
    edge_weights_dict = {}
    for i, row in mid_dataframe.iterrows():
        ccode_a = row['statea']
        ccode_b = row['stateb']
        hostlev = row['hihost']

        # We don't care about the order of the countries, so sort them by country code.
        edge_tuple = (min(ccode_a, ccode_b), max(ccode_a, ccode_b))

        # Set the edge weight according to the hostility level
        if hostlev < 2:
            continue
        elif hostlev == 2:
            weight = 1
        elif hostlev == 3:
            weight = 3
        elif hostlev == 4:
            weight = 3
        elif hostlev == 5:
            weight = 30
        else:
            raise Exception("Should never get here!")

        # Update the weight of this edge
        if edge_tuple not in edge_weights_dict:
            edge_weights_dict[edge_tuple] = weight
        else:
            edge_weights_dict[edge_tuple] += weight
    return edge_weights_dict


def generate_graph(start_year, end_year):
    """
    Generate a graph edgelist from the disputes in the Dyadic MID dataset.
    :return:
    """
    print("yooo")
    # Ensure the graphs directory exists
    os.makedirs(GRAPHS_DIRECTORY, exist_ok=True)

    # If the graph already exists, we do not need to create it again
    if os.path.exists(edgelist_filename(start_year, end_year)) and os.path.exists(vertex_filename(start_year, end_year)):
        return

    # If the data is not in the expected place in the data folder, then raise an exception
    if not (os.path.exists(COUNTRY_CODES_FILENAME) and os.path.exists(DYADIC_MID_FILENAME)):
        raise Exception("Cannot generate graph - MID data not found.")
    
    # Get the mappings from code to state name
    df_states = pd.read_csv(COUNTRY_CODES_FILENAME)
    ccode_to_vertex = {}
    vertex_labels = []

    # Load the disputes data into a dataframe
    df = pd.read_csv(DYADIC_MID_FILENAME).query(f"year >= {start_year} & year <= {end_year}")

    # Look through the disputes and build the weights dictionary
    edge_weights_dict = build_edge_weights(df)

    # Print the edgelist to the output file
    with open(edgelist_filename(start_year, end_year), 'w') as f_out:
        for edge_tuple in edge_weights_dict:
            # Add edge only if weight is 3 or greater
            if edge_weights_dict[edge_tuple] >= 3:
                # Add these country labels to the dictionary and array storing them
                for country_code in edge_tuple:
                    if country_code not in ccode_to_vertex:
                        country_name = df_states.query(f"CCode == {country_code}").iloc[0, 2]
                        next_index = len(vertex_labels)
                        ccode_to_vertex[country_code] = next_index
                        vertex_labels.append(country_name)

                # Write to the edgelist file
                weight = edge_weights_dict[edge_tuple]
                f_out.write(f"{ccode_to_vertex[edge_tuple[0]]} {ccode_to_vertex[edge_tuple[1]]} {weight}\n")

    # Save the vertex labels to the vertex file
    with open(vertex_filename(start_year, end_year), 'w') as f_out:
        for index, country_name in enumerate(vertex_labels):
            f_out.write(f"{index} {country_name}\n")  # Save as `index country_name`

    print(f"Edgelist and vertex list files for {start_year}-{end_year} saved successfully.")
