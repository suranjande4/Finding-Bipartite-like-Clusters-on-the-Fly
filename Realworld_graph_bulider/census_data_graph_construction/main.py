import networkx as nx
import csv
import collections

_CENSUS_DATA_FILE = "/home/suranjan/Desktop/Bipartite Sparsification/census_data_graph_construction./census_data_graph_construction/data/US_2000_CENSUS.TXT"


def load_census_data() -> tuple[dict[str, str], dict[tuple[str, str], int]]:
    """
    Functions to load the census data.

    Returns:
        fips_to_name: A dictionary mapping FIPS codes to county names
        county_to_county_migration: A dictionary mapping source and target county FIPS codes to migration amounts
    """
    fips_to_name = {}
    county_id_to_county_id_migration = collections.defaultdict(int)

    with open(_CENSUS_DATA_FILE, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            row = [field for field in row if field != ""]

            # get the numeric positions for the target county numbers.
            # these are first new digits after the 1st 4 digits.
            loc_target_county_nums = [
                i + 4 for i, field in enumerate(row[4:]) if field.isdigit()
            ]

            # e.g., "01001" first 2 digits are state code and next 3 are
            # county code (FIPS CODES: https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt)
            source_county_number = f"{row[0]}{row[1]}"

            # the name of the county is all the non numeric fields between the first 2 county codes
            # and the target county codes
            source_county_name = " ".join(row[4 : loc_target_county_nums[0]])

            target_county_number = (
                f"{row[loc_target_county_nums[0]]}{row[loc_target_county_nums[1]]}"[1:]
            )

            # all remaining non numeric fields correspond to the target county name
            target_county_name = " ".join(
                field for field in row[loc_target_county_nums[1] :] if field.isalpha()
            )
            if target_county_number[-3:] == "000":
                # target county is a country, not a county. Skip these
                continue

            total_migration = int(row[-1])  # Last column is the migration amount

            if target_county_name not in fips_to_name:
                fips_to_name[target_county_number] = target_county_name
            else:
                assert fips_to_name[target_county_number] == target_county_name

            county_id_to_county_id_migration[
                (source_county_number, target_county_number)
            ] = total_migration

    return fips_to_name, county_id_to_county_id_migration


def _get_unique_order_invariant_tuples(tuple_list):
    unique_tuples = set()
    for t in tuple_list:
        # Sort the tuple to make it order-invariant
        sorted_t = tuple(sorted(t))
        unique_tuples.add(sorted_t)
    return list(unique_tuples)


def construct_migration_graph(
    county_id_to_county_id_migration: dict[tuple[str, str], int],
    include_self_loops: bool = False,
) -> nx.DiGraph:
    """
    Function to construct a directed graph from the migration data.

    The nodes in the graph are counties, identified by their unique FIPS codes.

    Args:
        county_to_county_migration: A dictionary mapping source and target county FIPS codes to migration amounts

    Returns:
        migration_graph: A directed graph where the nodes are counties and the edges are net migration amounts
    """
    # get unique pairs of counties
    unique_county_pairs = _get_unique_order_invariant_tuples(
        county_id_to_county_id_migration.keys()
    )

    migration_graph = nx.DiGraph()
    for source, target in unique_county_pairs:
        if not include_self_loops and source == target:
            continue
        source_to_target = county_id_to_county_id_migration[(source, target)]
        target_to_source = county_id_to_county_id_migration[(target, source)]
        if source_to_target + target_to_source > 0:
            net_normalised_migration = source_to_target - target_to_source / (
                source_to_target + target_to_source
            )
            if net_normalised_migration > 0:
                migration_graph.add_edge(
                    source,
                    target,
                    weight=net_normalised_migration,
                )
            else:
                migration_graph.add_edge(
                    target,
                    source,
                    weight=-net_normalised_migration,
                )

    for edge in migration_graph.edges(data=True):
        assert edge[2]["weight"] > 0

    return migration_graph


def main():
    fips_to_name, county_id_to_county_id_migration = load_census_data()
    #print(len(fips_to_name))
    migration_graph = construct_migration_graph(county_id_to_county_id_migration)
    print(migration_graph.number_of_edges())
    print(migration_graph.number_of_nodes())
    sorted_nodes = sorted(migration_graph.nodes)  # Sort the nodes
    print(sorted_nodes[3010])
    mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_nodes)}
    renamed_graph = nx.relabel_nodes(migration_graph, mapping)
    '''
    with open("Renamed_US_Migration_directed_graph.edgelist", "w") as f:
        for u, v, data in renamed_graph.edges(data=True):
            weight = data.get('weight', 1.0)  # Default weight if none exists
            f.write(f"{u} {v} {weight}\n")
    lowest_label = min(renamed_graph.nodes)
    highest_label = max(renamed_graph.nodes)
    print(f"Lowest node label: {lowest_label}")
    print(f"Highest node label: {highest_label}")
    '''
if __name__ == "__main__":
    main()