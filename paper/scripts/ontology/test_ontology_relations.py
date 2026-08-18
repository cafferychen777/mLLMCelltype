import requests
import json
from pprint import pprint

def get_cl_term_info(term_label):
    """从Cell Ontology获取术语信息"""
    print(f"\n=== Testing relations for: {term_label} ===")

    # 1. 首先查找术语
    print("\n1. Finding term...")
    search_params = {
        'q': term_label,
        'ontology': 'cl',
        'exact': 'true',
        'queryFields': 'label,synonym'
    }

    response = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=search_params)
    if response.status_code != 200:
        print(f"Error searching term: {response.status_code}")
        return None

    data = response.json()
    if data['response']['numFound'] == 0:
        print(f"Term not found: {term_label}")
        return None

    # 找到CL本体中的术语
    cl_term = None
    for doc in data['response']['docs']:
        if doc['ontology_prefix'] == 'CL':
            cl_term = doc
            break

    if not cl_term:
        print(f"Term not found in Cell Ontology: {term_label}")
        return None

    term_iri = cl_term['iri']
    term_id = cl_term['obo_id']
    print(f"Found term: {term_id} ({cl_term['label']})")
    print(f"IRI: {term_iri}")

    return {'iri': term_iri, 'id': term_id, 'label': cl_term['label']}

def get_term_relations(term_info):
    """获取术语的关系"""
    if not term_info:
        return

    encoded_iri = requests.utils.quote(term_info['iri'], safe='')

    # 首先获取术语详情
    print("\n获取术语详情...")
    response = requests.get(f"https://www.ebi.ac.uk/ols4/api/terms?iri={encoded_iri}")
    if response.status_code != 200:
        print(f"Error getting term details: {response.status_code}")
        return

    data = response.json()
    if '_embedded' not in data or 'terms' not in data['_embedded']:
        print("No term details found")
        return

    # 找到CL本体中的术语
    cl_term = None
    for term in data['_embedded']['terms']:
        if term.get('ontology_prefix') == 'CL':
            cl_term = term
            break

    if not cl_term:
        print("Term not found in Cell Ontology")
        return

    # 获取关系链接
    links = cl_term.get('_links', {})

    # 2. 获取父节点
    print("\n2. Getting parents...")
    if 'parents' in links:
        parents_url = links['parents']['href']
        response = requests.get(parents_url)
        if response.status_code == 200:
            data = response.json()
            if '_embedded' in data and 'terms' in data['_embedded']:
                parents = data['_embedded']['terms']
                print(f"Found {len(parents)} parents:")
                for parent in parents:
                    print(f"- {parent['obo_id']}: {parent['label']}")
            else:
                print("No parents found")
        else:
            print(f"Error getting parents: {response.status_code}")
    else:
        print("No parents link available")

    # 3. 获取子节点
    print("\n3. Getting children...")
    if 'children' in links:
        children_url = links['children']['href']
        response = requests.get(children_url)
        if response.status_code == 200:
            data = response.json()
            if '_embedded' in data and 'terms' in data['_embedded']:
                children = data['_embedded']['terms']
                print(f"Found {len(children)} children:")
                for child in children:
                    print(f"- {child['obo_id']}: {child['label']}")
            else:
                print("No children found")
        else:
            print(f"Error getting children: {response.status_code}")
    else:
        print("No children link available")

    # 4. 获取祖先
    print("\n4. Getting ancestors...")
    if 'ancestors' in links:
        ancestors_url = links['ancestors']['href']
        response = requests.get(ancestors_url)
        if response.status_code == 200:
            data = response.json()
            if '_embedded' in data and 'terms' in data['_embedded']:
                ancestors = data['_embedded']['terms']
                print(f"Found {len(ancestors)} ancestors:")
                for ancestor in ancestors:
                    print(f"- {ancestor['obo_id']}: {ancestor['label']}")
            else:
                print("No ancestors found")
        else:
            print(f"Error getting ancestors: {response.status_code}")
    else:
        print("No ancestors link available")

    # 5. 获取后代
    print("\n5. Getting descendants...")
    if 'descendants' in links:
        descendants_url = links['descendants']['href']
        response = requests.get(descendants_url)
        if response.status_code == 200:
            data = response.json()
            if '_embedded' in data and 'terms' in data['_embedded']:
                descendants = data['_embedded']['terms']
                print(f"Found {len(descendants)} descendants:")
                for descendant in descendants:
                    print(f"- {descendant['obo_id']}: {descendant['label']}")
            else:
                print("No descendants found")
        else:
            print(f"Error getting descendants: {response.status_code}")
    else:
        print("No descendants link available")

if __name__ == "__main__":
    # 测试一些典型的细胞类型
    test_cases = [
        "T cell",  # 一个较高层级的细胞类型
        "CD4-positive, alpha-beta T cell",  # 一个具体的细胞类型
        "thymocyte",  # 与胸腺相关的细胞类型
        "double-positive thymocyte"  # 更具体的胸腺细胞类型
    ]

    for term in test_cases:
        term_info = get_cl_term_info(term)
        if term_info:
            get_term_relations(term_info)
        print("\n" + "="*80)
