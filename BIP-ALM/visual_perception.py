import cv2
import math
import json
import pickle
import imageio
import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from graph_utils import ALL_LIST, ROOM_LIST, CONTAINER_LIST, SURFACE_LIST, OBJECT_LIST
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
sys.path.append('..')


def load_pickles(path, folder=True):
    if folder:
        pickle_data = {}
        for file in os.listdir(path):
            if file.endswith(".pik"):
                with open(os.path.join(path, file), 'rb') as f:
                    data = pickle.load(f)
                    pickle_data[file] = data
    else:
        if path.endswith(".pik"):
            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
    return pickle_data


def read_frame_intervals(parent_path):
    path = parent_path + 'frame_intervals.pik'
    with open(path, 'rb') as f:
        intervals = pickle.load(f)
    return intervals


def find_pixel_id(time, parent_path, parent_path_video):
    path = parent_path + 'instanace_colors.pik'
    with open(path, 'rb') as f:
        id2color = pickle.load(f)

    path = parent_path + 'init_graph.pik'
    with open(path, 'rb') as f:
        g = pickle.load(f)

    id2classname = {n["id"]: n["class_name"] for n in g["nodes"]}
    id2color_filtered = {
            int(i): [int(x * 255) for x in rgb]
            for i, rgb in id2color.items()
            if int(i) in id2classname and id2classname[int(i)] in ALL_LIST
        }

    path = parent_path_video + f'Action_{time:04d}_0_seg_inst.png'
    imgs_seg = cv2.imread(path)
    img_shape = imgs_seg.shape
    height, width, channels = imgs_seg.shape
    img_id = np.empty((height, width, 1))
    object_list = []
    for x in range(img_shape[0]):
        for y in range(img_shape[1]):
            curr_rgb = imgs_seg[x, y, :]
            found_id = None
            for object_id, rgb in id2color_filtered.items():
                match = True
                for c_id in range(3):
                    if abs(curr_rgb[2 - c_id] - rgb[c_id]) > 1:
                        match = False
                        break
                if match:
                    found_id = object_id
                    break

            img_id[x, y, 0] = -1
            if found_id is not None:
                if found_id in id2classname:
                    object_classname = id2classname[found_id]
                    if object_classname in ALL_LIST:
                        img_id[x, y, 0] = found_id
                        if object_classname not in object_list:
                            object_list.append(object_classname)

    return img_id, id2classname


def inverse_rot(currrot):
    new_matrix = np.zeros((4,4))
    new_matrix[:3, :3] = currrot[:3,:3].transpose()
    new_matrix[-1,-1] = 1
    new_matrix[:-1, -1] = np.matmul(currrot[:3,:3].transpose(), -currrot[:-1, -1])
    
    return new_matrix


def project_pixel_to_3d(rgb, time, num, parent_path, parent_path_video, scale=1, ax=None, ij=None):
    path = parent_path_video + f'Action_{time:04d}_0_depth.exr'
    depth = imageio.imread(path)
    depth = depth[:,:,0]

    path = parent_path + f'camera_data_{num}.pik'
    with open(path, 'rb') as f:
        params = pickle.load(f)


    (hs, ws, _) = rgb.shape
    naspect = float(ws/hs)
    aspect = params['aspect']
    
    w = np.arange(ws)
    h = np.arange(hs)
    projection = np.array(params['projection_matrix']).reshape((4,4)).transpose()
    view = np.array(params['world_to_camera_matrix']).reshape((4,4)).transpose()
    
    # Build inverse projection
    inv_view = inverse_rot(view)
    rgb = rgb.reshape(-1,3)
    col= rgb
    
    xv, yv = np.meshgrid(w, h)
    npoint = ws*hs
    
    # Normalize image coordinates to -1 to 1
    xp = xv.reshape((-1))
    yp = yv.reshape((-1))
    if ij is not None:
        index_interest = ij[0]*ws + ij[1]
    else:
        index_interest = None
     
    x = xv.reshape((-1)) * 2./ws - 1
    y = 2 - (yv.reshape((-1)) * 2./hs) - 1
    z = depth.reshape((-1))
    
    nump = x.shape[0]
    
    m00 = projection[0,0]
    m11 = projection[1,1]
    
    xn = x*z / m00
    yn = y*z / m11
    zn = -z
    XY1 = np.concatenate([xn[:, None], yn[:, None], zn[:, None], np.ones((nump,1))], 1).transpose()

    # World coordinates
    XY = np.matmul(inv_view, XY1).transpose()

    x, y, z = XY[:, 0], XY[:, 1], XY[:, 2]
    if ij is not None:
        print("3D point", x[index_interest], y[index_interest], z[index_interest])
    return x,y,z


def find_bounding_boxes(pixel2id, x, y, z, id2classname):
    id_array = pixel2id.reshape(-1)

    grouped_positions = defaultdict(list)
    for position, id_value in enumerate(id_array):
        if id_value != 255:
            grouped_positions[id_value].append(position)
    grouped_positions = dict(grouped_positions)

    bounding_boxes = {}  
    for id_value, positions in grouped_positions.items():
        if id_value in id2classname:
            bounding_box = {}
            bounding_box['class_name'] = id2classname[id_value]
            x_values = [x[pos] for pos in positions]
            bounding_box['x_max'], bounding_box['x_min'] = max(x_values), min(x_values)
            y_values = [y[pos] for pos in positions]
            bounding_box['y_max'], bounding_box['y_min'] = max(y_values), min(y_values)
            z_values = [z[pos] for pos in positions]
            bounding_box['z_max'], bounding_box['z_min'] = max(z_values), min(z_values)
            bounding_boxes[id_value] = bounding_box
    return bounding_boxes


def calculate_1d_distance(dim1, dim2):
    if dim1['max'] < dim2['min']:
        return dim2['min'] - dim1['max']
    elif dim2['max'] < dim1['min']:
        return dim1['min'] - dim2['max']
    else:
        return 0.0


def calculate_distance_from_object(bounding_boxes, id):
    try:
        object_1 = bounding_boxes[id]
    except KeyError:
        return None

    # Prevent comparison with itself
    self_id = id
    
    distances = {}
    for id, obj_data in bounding_boxes.items():
        if id == self_id:
            continue

        x_dist = calculate_1d_distance({'min': object_1['x_min'], 'max': object_1['x_max']}, {'min': obj_data['x_min'], 'max': obj_data['x_max']})
        y_dist = calculate_1d_distance({'min': object_1['y_min'], 'max': object_1['y_max']}, {'min': obj_data['y_min'], 'max': obj_data['y_max']})
        z_dist = calculate_1d_distance({'min': object_1['z_min'], 'max': object_1['z_max']}, {'min': obj_data['z_min'], 'max': obj_data['z_max']})
        dist = math.sqrt(x_dist**2 + z_dist**2)

        distances[id] = dist
    return distances


def find_close_to(distances, id2classname, threshold = 1):
    close_to_items = []
    for id, distance in distances.items():
        if distance <= threshold and id2classname[id] not in ROOM_LIST:
            close_to_items.append(id)

    return list(set(close_to_items))


def find_closeness(times, parent_path, parent_path_video):
    all_closeness = {}
    all_bounding_boxes = {}
    num = 0
    for time in times:
        pixel2id, id2classname = find_pixel_id(time, parent_path, parent_path_video)
        x, y, z = project_pixel_to_3d(pixel2id, time, num, parent_path, parent_path_video)
        bounding_boxes = find_bounding_boxes(pixel2id, x, y, z, id2classname)
        all_bounding_boxes[num] = bounding_boxes

        distances = calculate_distance_from_object(bounding_boxes, 1)
        if distances == None:
            close_to_items = []
        else:
            close_to_items = find_close_to(distances, id2classname)

        close_to_items = [id2classname[id] for id in close_to_items]
        all_closeness[num] = list(set(close_to_items))
        num += 1

    return all_bounding_boxes, all_closeness


def find_GT_closeness(parent_path):
    all_closeness = {}
    task = load_pickles(parent_path, folder=True)
    time = 0
    while True:
        graph_file_name = f'graph_{time}.pik'
        if graph_file_name not in task:
            break
        graph = task[graph_file_name]
        
        closeness = []
        edges = graph['edges']
        id2node = {node["id"]: node for node in graph["nodes"]}
        id2name = {id:  node["class_name"] for id, node in id2node.items()}

        for edge in edges:
            from_id = edge['from_id']
            to_id = edge['to_id']

            edge['from_name'] = id2name.get(from_id)
            edge['to_name'] = id2name.get(to_id)

        for edge in edges:
            if edge['from_name'] == 'character' and edge['to_name'] in ALL_LIST:
                if edge['relation_type'] == 'CLOSE':
                    closeness.append(edge['to_name'])
        
        all_closeness[time] = list(set(closeness))
        time += 1

    return all_closeness


def f1_score(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    true_positives = len(set1.intersection(set2))
    false_positives = len(set2 - set1)
    false_negatives = len(set1 - set2)

    if true_positives == 0:
        return 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def find_GT_inside(parent_path):
    all_inside = {}
    task = load_pickles(parent_path, folder=True)
    time = 0
    while True:
        graph_file_name = f'graph_{time}.pik'
        if graph_file_name not in task:
            break
        graph = task[graph_file_name]
        
        inside = []
        edges = graph['edges']
        id2node = {node["id"]: node for node in graph["nodes"]}
        id2name = {id:  node["class_name"] for id, node in id2node.items()}

        for edge in edges:
            from_id = edge['from_id']
            to_id = edge['to_id']

            edge['from_name'] = id2name.get(from_id)
            edge['to_name'] = id2name.get(to_id)

        for edge in edges:
            if edge['relation_type'] == 'INSIDE':
                if edge['from_name'] in CONTAINER_LIST and edge["to_name"] in ROOM_LIST:
                    inside.append(f"{edge['from_name']} inside {edge['to_name']}")
                if edge['from_name'] in OBJECT_LIST and edge["to_name"] in CONTAINER_LIST:
                    inside.append(f"{edge['from_name']} in {edge['to_name']}")
            if edge['relation_type'] == 'ON':
                if edge['from_name'] in OBJECT_LIST and edge["to_name"] in SURFACE_LIST:
                    inside.append(f"{edge['from_name']} on {edge['to_name']}")
        
        all_inside[time] = inside
        time += 1

    return all_inside


def find_inside_and_open(times, parent_path, episode_bounding_boxes):
    all_inside = {}
    all_opened = {}
    num = 0
    for time in times:
        path = parent_path + 'init_graph.pik'
        with open(path, 'rb') as f:
            g = pickle.load(f)
        id2classname = {n["id"]: n["class_name"] for n in g["nodes"]}

        bounding_boxes = episode_bounding_boxes[str(num)]
        bounding_boxes = {int(float(key)): value for key, value in bounding_boxes.items()}

        inside = []
        containers = [id for id in id2classname.keys() if id2classname[id] in CONTAINER_LIST]
        for id in containers:
            distances = calculate_distance_from_object(bounding_boxes, id)
            if distances != None:
                for id2, distance in distances.items():
                    if distance == 0 and id2classname[id2] in OBJECT_LIST:
                        inside.append(f'A {id2classname[id2]} is in the {id2classname[id]}. ')
        
        opened = []
        if num != 0:
            previous_bounding_boxes = episode_bounding_boxes[str(num-1)]
            previous_bounding_boxes = {int(float(key)): value for key, value in previous_bounding_boxes.items()}
            containers = [id for id in id2classname.keys() if id2classname[id] in CONTAINER_LIST]
            for id in containers:
                if id in bounding_boxes and id in previous_bounding_boxes:
                    current = bounding_boxes[id]
                    previous = previous_bounding_boxes[id]
                    if (current['x_max'] - current['x_min'] > previous['x_max'] - previous['x_min'] + 0.2) or \
                        (current['y_max'] - current['y_min'] > previous['y_max'] - previous['y_min'] + 0.2) or \
                        (current['z_max'] - current['z_min'] > previous['z_max'] - previous['z_min'] + 0.2):
                        opened.append(id2classname[id])

        all_inside[num] = inside
        all_opened[num] = list(set(opened))
        num += 1

    return all_inside, all_opened


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_video_path", type=str, default='videos')
    args = parser.parse_args()

    CLOSENESS = True
    INSIDE_OPEN = True

    all_closeness = {}
    all_boxes = {}
    all_inside = {}
    all_opened = {}
    episodes = [1, 6, 13, 14, 23, 39, 43, 51, 54, 61, 65, 88, 93, 96, 103, 114, 118, 120, 124, 130, 
                131, 135, 139, 142, 143, 146, 147, 155, 176, 179, 187, 190, 192, 199, 201, 202, 207, 225, 230, 231, 
                236, 252, 260, 274, 289, 291, 317, 330, 333, 334, 337, 340, 341, 345, 358, 375, 402, 418, 422, 441, 
                445, 447, 450, 460, 462, 464, 475, 482, 487, 492, 509, 517, 521, 525, 527, 530, 556, 611, 619, 630, 
                657, 663, 682, 712, 745, 747, 760, 761, 764, 769, 798, 800, 804, 811, 833, 848, 865, 883, 890, 891, 
                901, 914, 921, 943, 953, 954, 964, 966, 972, 973, 982, 984, 1001, 1025, 1026, 1031, 1058, 1062, 1073, 1079, 
                1083, 1092, 1100, 1124, 1127, 1132, 1135, 1146, 1150, 1170, 1178, 1179, 1185, 1196]
    for episode in tqdm(episodes):
        try:
            parent_path = f'{args.benchmark_video_path}/task_{episode}/'
            parent_path_video = f'{args.benchmark_video_path}/task_{episode}/script/0/'

            intervals = read_frame_intervals(parent_path)
            times = [action[1] for action in intervals]


            if CLOSENESS:
                boxes, closeness = find_closeness(times, parent_path, parent_path_video)
                GT_closeness = find_GT_closeness(parent_path)
                accuracy = []
                for i in range(len(closeness)):
                    accuracy.append(f1_score(GT_closeness[i], closeness[i]))
                print(sum(accuracy) / len(accuracy))
                all_boxes[episode] = boxes
                all_closeness[episode] = closeness

            if INSIDE_OPEN:
                with open('boxes.json', 'rb') as f:
                    all_bounding_boxes = json.load(f)
                episode_bounding_boxes = all_bounding_boxes[str(episode)]
                all_inside[episode], all_opened[episode] = find_inside_and_open(times, parent_path, episode_bounding_boxes)      
        
        except FileNotFoundError:
            continue
    

    if CLOSENESS:
        file_path = "closeness.json"
        with open(file_path, "w") as file:
            pass
        with open(file_path, 'a') as file:
            json.dump(all_closeness, file)
            file.write('\n')
        
        file_path = "boxes.json"
        with open(file_path, "w") as file:
            pass
        with open(file_path, 'a') as file:
            json.dump(all_boxes, file)
            file.write('\n')

    if INSIDE_OPEN:
        file_path = "inside.json"
        with open(file_path, "w") as file:
            pass
        with open(file_path, 'a') as file:
            json.dump(all_inside, file)
            file.write('\n')
    
        file_path = "opened.json"
        with open(file_path, "w") as file:
            pass
        with open(file_path, 'a') as file:
            json.dump(all_opened, file)
            file.write('\n')