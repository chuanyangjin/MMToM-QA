import random
random.seed(42)

TEST_INDICES = [1, 6, 13, 14, 23, 39, 43, 51, 54, 61, 64, 65, 88, 93, 96, 103, 110, 114, 118, 120, 124, 130, 131, 135, 139, 142, 143, 146, 147, 155, 162, 164, 176, 179, 187, 190, 192, 194, 199, 201, 202, 207, 210, 218, 225, 226, 230, 231, 236, 252, 258, 260, 265, 274, 287, 289, 291, 296, 317, 323, 330, 333, 334, 337, 340, 341, 345, 358, 375, 385, 399, 402, 403, 418, 422, 441, 444, 445, 447, 450, 452, 460, 462, 464, 472, 475, 482, 484, 485, 487, 492, 503, 506, 509, 513, 517, 521, 523, 525, 527, 530, 555, 556, 558, 559, 563, 565, 569, 582, 588, 589, 611, 619, 620, 630, 648, 657, 663, 665, 682, 683, 695, 709, 712, 714, 716, 724, 745, 747, 755, 760, 761, 764, 769, 781, 787, 788, 798, 800, 801, 804, 811, 833, 840, 844, 848, 857, 865, 883, 888, 889, 890, 891, 901, 906, 914, 921, 931, 942, 943, 948, 953, 954, 964, 966, 972, 973, 982, 984, 995, 1001, 1025, 1026, 1031, 1043, 1058, 1062, 1073, 1079, 1083, 1092, 1100, 1121, 1124, 1127, 1132, 1135, 1140, 1141, 1145, 1146, 1150, 1159, 1168, 1170, 1175, 1178, 1179, 1185, 1196]
TEST_INDICES_SCENARIO1 = [6, 23, 43, 51, 54, 61, 65, 114, 124, 130, 131, 135, 139, 142, 143, 147, 155, 164, 187, 190, 201, 202, 207, 236, 252, 258, 274, 289, 291, 296, 317, 323, 334, 358, 375, 399, 402, 422, 441, 445, 460, 462, 482, 484, 485, 487, 492, 503, 506, 509, 513, 521, 523, 525, 527, 569, 582, 630, 657, 663, 682, 683, 714, 745, 755, 761, 764, 781, 798, 800, 801, 811, 840, 865, 888, 889, 890, 891, 906, 931, 964, 966, 973, 984, 995, 1031, 1043, 1079, 1092, 1100, 1121, 1124, 1132, 1141, 1145, 1146, 1150, 1179, 1185, 1196]
TEST_INDICES_SCENARIO2 = [1, 13, 14, 39, 64, 88, 93, 96, 103, 110, 118, 120, 146, 162, 176, 179, 192, 194, 199, 210, 218, 225, 226, 230, 231, 260, 265, 287, 330, 333, 337, 340, 341, 345, 385, 403, 418, 444, 447, 450, 452, 464, 472, 475, 517, 530, 555, 556, 558, 559, 563, 565, 588, 589, 611, 619, 620, 648, 665, 695, 709, 712, 716, 724, 747, 760, 769, 787, 788, 804, 833, 844, 848, 857, 883, 901, 914, 921, 942, 943, 948, 953, 954, 972, 982, 1001, 1025, 1026, 1058, 1062, 1073, 1083, 1127, 1135, 1140, 1159, 1168, 1170, 1175, 1178]

ROOM_LIST = [
    "bathroom", 
    "bedroom",
    "kitchen",
    "livingroom", 
]

SURFACE_LIST = [
    "coffeetable",
    "desk",
    "kitchentable",
    "sofa",
]

CONTAINER_LIST = [
    "kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

NEW_CONTAINER_LIST = [
    "1st kitchencabinet from left to right",
    "2nd kitchencabinet from left to right",
    "3rd kitchencabinet from left to right",
    "4th kitchencabinet from left to right",
    "5th kitchencabinet from left to right",
    "6th kitchencabinet from left to right",
    "7th kitchencabinet from left to right",
    "8th kitchencabinet from left to right",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]

OBJECT_LIST = [ 
    "apple",
    "book",
    "chips",
    "condimentbottle",
    "cupcake",
    "dishbowl",
    "plate",
    "remotecontrol",
    "salmon",
    "waterglass",
    "wine",
    "wineglass",
]

CHARACTER_LIST = [
    "character"
]

ALL_LIST = ROOM_LIST + SURFACE_LIST + CONTAINER_LIST + OBJECT_LIST + CHARACTER_LIST

POSSIBLE_BELIEF = [
    "1st kitchencabinet",
    "2nd kitchencabinet",
    "3rd kitchencabinet",
    "4th kitchencabinet",
    "5th kitchencabinet",
    "6th kitchencabinet",
    "7th kitchencabinet",
    "8th kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
    "coffeetable",
    "desk"
    "kitchentable",
    "sofa"
]

POSSIBLE_CONTAINER = [
    "1st kitchencabinet",
    "2nd kitchencabinet",
    "3rd kitchencabinet",
    "4th kitchencabinet",
    "5th kitchencabinet",
    "6th kitchencabinet",
    "7th kitchencabinet",
    "8th kitchencabinet",
    "cabinet",
    "bathroomcabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "stove",
]


def get_id2node(graph):
    return {node["id"]: node for node in graph["nodes"]}


def filter_graph(graph, ROOM_LIST=ROOM_LIST, OBJ_CLASSES=ALL_LIST):
    if 'edge' in graph: graph['edges'] = graph['edge']

    new_graph = {}
    new_graph["nodes"] = [
        node
        for node in graph["nodes"]
        if node["class_name"] in ROOM_LIST or node["class_name"] in OBJ_CLASSES
    ]
    idx_list = [node["id"] for node in new_graph["nodes"]]
    new_graph["edges"] = [
        edge for edge in graph["edges"] if (edge["from_id"] in idx_list) or (edge["to_id"] in idx_list)
    ]
    return new_graph


def transform_graph(graph, new_names):
    edges = graph['edges']

    # Convert ids to names
    id2node = get_id2node(graph)
    id2name = {id: node["class_name"] for id, node in id2node.items()}
    id2name_new = {id: new_names[id] if id in new_names else node["class_name"] for id, node in id2node.items()}

    # Remove duplicate edges
    edges = [dict(t) for t in set(tuple(d.items()) for d in edges)]
    
    for edge in edges:
        from_id = edge['from_id']
        to_id = edge['to_id']

        edge['from_name'] = id2name.get(from_id)
        edge['to_name'] = id2name.get(to_id)
        edge['from_name_new'] = id2name_new.get(from_id)
        edge['to_name_new'] = id2name_new.get(to_id)


    results = ""
    rooms = list(set(ROOM_LIST) & set([edge['to_name'] for edge in edges]))
    results += f"There is {' and '.join('a ' + room for room in rooms)}. "
    for room in rooms:
        containers_count = {}
        containers_new = []
        for edge in edges:
            if edge['from_name'] in (CONTAINER_LIST + SURFACE_LIST) and edge["to_name"] == room:
                containers_count[edge['from_name']] = containers_count.get(edge['from_name'], 0) + 1
                containers_new.append([edge['from_name_new'], edge['from_id']])
            
        containers = []
        for key, value in containers_count.items():
            if value == 1:
                containers.append(f'a {key}')
            else:
                containers.append(f'{value} {key}s')
        
        if len(containers) != 0:
            prep = "is" if len(containers_new) == 1 else "are"
            results += f"\n{' and '.join(container for container in containers)} {prep} in the {room}. "

            for container in containers_new:
                objects_count = {}
                for edge in edges:
                    if edge['from_name'] in OBJECT_LIST and edge["to_id"] == container[1]:
                        objects_count[edge['from_name']] = objects_count.get(edge['from_name'], 0) + 1

                    objects = []
                    for key, value in objects_count.items():
                        if value == 1:
                            objects.append(f'a {key}')
                        else:
                            objects.append(f'{value} {key}s')
                        
                if len(objects) != 0:
                    prep = "is" if len(objects) == 1 and 'a ' in objects[0] else "are"
                    relation = "inside" if container[0] in NEW_CONTAINER_LIST else "on"
                    results += f"{' and '.join(object for object in objects)} {prep} {relation} the {container[0]}. "
                else:
                    if container[0] in NEW_CONTAINER_LIST:
                        results += f"There is nothing inside the {container[0]}. "

    results = results.replace('waterglasss', 'waterglasses')
    results = results.replace('wineglasss', 'wineglasses')
    results = results.replace('a wine ', 'a bottle of wine ')
    results = results.replace('wines', 'bottles of wine')
    results = results.replace('a chips', 'a bag of chips')
    results = results.replace('chipss', 'bags of chips')
    results = results.replace('a apple', 'an apple')

    return results


def transform_graph_training(graph, new_names):
    results = transform_graph(graph, new_names)

    edges = graph['edges']

    # Convert ids to names
    id2node = get_id2node(graph)
    id2name = {id: new_names[id] if id in new_names else node["class_name"] for id, node in id2node.items()}

    for edge in edges:
        from_id = edge['from_id']
        to_id = edge['to_id']

        edge['from_name'] = id2name.get(from_id)
        edge['to_name'] = id2name.get(to_id)

    shuffled_all_list = ALL_LIST.copy()
    random.shuffle(shuffled_all_list)
    shuffled_container_list = CONTAINER_LIST.copy()
    random.shuffle(shuffled_container_list)

    # Describe the person's location
    results += "\n"
    for edge in edges:
        if edge['from_name'] == 'character' and edge['to_name'] in shuffled_all_list:
            relation = 'inside' if edge['relation_type'] == 'INSIDE' else 'close to'
            results += f"The person is {relation} the {edge['to_name']}. "
    
    # Describe the containers' state
    results += "\n"
    for node in graph['nodes']:
        if node['class_name'] in shuffled_container_list:
            if 'CLOSED' in node['states']:
                results += f"The {id2name[node['id']]} is closed. "
            if 'OPEN' in node['states']:
                results += f"The {id2name[node['id']]} is open. "
    
    results = results.replace('waterglasss', 'waterglasses')
    results = results.replace('wineglasss', 'wineglasses')
    results = results.replace('a wine ', 'a bottle of wine ')
    results = results.replace('wines', 'bottles of wine')
    results = results.replace('a chips', 'a bag of chips')
    results = results.replace('chipss', 'bags of chips')
    results = results.replace('a apple', 'an apple')

    return results


def get_inside_objects(g, container_id, id2node):
    objects = []
    for e in g["edges"]:
        if e["relation_type"] == "INSIDE" and e["to_id"] == container_id:
            object_name = id2node[e["from_id"]]["class_name"]
            if (
                object_name != "character"
                and object_name in ALL_LIST
                and object_name not in objects
            ):
                objects.append(object_name)
    return objects


def get_on_objects(g, container_id, id2node):
    objects = []
    for e in g["edges"]:
        if e["relation_type"] == "ON" and e["to_id"] == container_id:
            object_name = id2node[e["from_id"]]["class_name"]
            if (
                object_name != "character"
                and object_name in ALL_LIST
                and object_name not in objects
            ):
                objects.append(object_name)
    return objects


def get_init_room(graph):
    id2node = get_id2node(graph)
    edges = graph['edges']
    for edge in edges:
        if edge['from_id'] == 1 and edge['relation_type'] == 'INSIDE':
            init_room_id = edge['to_id']

    init_room = id2node[init_room_id]['class_name']
    return init_room