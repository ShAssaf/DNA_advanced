# data source is https://cov-lineages.org/lineage_list.html
import os
import random

import pandas as pd
import requests

VARIANT_NUMBER = 10
SAMPLE_NUMBER = 100


def download_lineage_meta():
    url = 'https://www.ebi.ac.uk/ebisearch/ws/rest/embl-covid19/download?query=&format=tsv&fields=id,lineage,collection_date,country,center_name,host,TAXON,coverage,who'
    print("downloading lineage_meta.tsv")

    response = requests.get(url)
    print("finished downloading lineage_meta.tsv")

    if response.status_code == 200:
        data = response.text
        # print(data)
        with open('./data/lineage_meta.tsv', 'w') as f:
            f.write(data)

    else:
        print("An error occurred. Status code:", response.status_code)
        exit(1)


def get_data_for_classifier(classifier_type: str = 'common'):
    """classifier_type: common or latest"""
    # check if lineage_meta.tsv exists
    if not os.path.exists('./data/lineage_meta.tsv'):
        download_lineage_meta()
    lineage_table_df = pd.read_html('https://cov-lineages.org/lineage_list.html')[0]

    # lineage_meta_df.to_csv('./data/lineage_info_table.csv', index=False)  # for future support
    lineage_meta_df = pd.read_table('./data/lineage_meta.tsv', low_memory=False)

    if classifier_type == 'common':
        lineages = lineage_table_df.sort_values(by='# assigned', ascending=False).head(VARIANT_NUMBER) \
            ['Lineage'].to_list()
    elif classifier_type == 'latest':
        lineages = lineage_table_df.sort_values(by='Earliest date', ascending=False).head(VARIANT_NUMBER) \
            ['Lineage'].to_list()
    else:
        raise ValueError('classifier_type should be common or latest')

    lineages_accession_id_dict = lineage_meta_df[lineage_meta_df['lineage'].isin(lineages)] \
        .groupby('lineage')['id'].apply(list).to_dict()

    for lineage in lineages_accession_id_dict.keys():
        # get random SAMPLE_NUMBER accession id
        lineages_accession_id_dict[lineage] = random.sample(lineages_accession_id_dict[lineage], SAMPLE_NUMBER)
        download_lineage_accessions(lineages_accession_id_dict[lineage], f'./data/{classifier_type}/{lineage}.fasta')
        print(f"running request ./data/{classifier_type}/{lineage}.fasta")


def download_lineage_accessions(accession_ids: list = None, output_file: str = None):
    url = 'https://www.ebi.ac.uk/ena/browser/api/fasta/' + ','.join(accession_ids)
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text
        # print(data)
        with open(output_file, 'w') as f:
            f.write(data)

    else:
        print("An error occurred. Status code:", response.status_code)
        exit(1)


if __name__ == '__main__':
    get_data_for_classifier()
