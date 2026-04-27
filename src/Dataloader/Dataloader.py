import copy
import utm
import pickle
import torch as t
import numpy as np
import pandas as pd
import os, pyproj
import datetime
import xml.etree.ElementTree as xml
import random

from math import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import namedtuple

from tool import *

THETA_ANGLE = 300
THETA_ACC = 0.006

Lanelet_ind = namedtuple("Lanelet", ['left', 'right', 'tags'])
Lanelet_interaction = namedtuple("Lanelet", ['left', 'right', 'regulations', 'tags'])

MS2S = 0.001 
S2MS = 1000
TRAINDATASET_RATIO = 0.7
TESTDATASET_RATIO = 0.3
DATASET_SPLIT_RANDOM_SEED = 0

device = 'cuda' if t.cuda.is_available() else 'cpu'
data_root = ''

class Node:
    """
    e.g. a point of interest, or a constituent point of a
    line feature such as a road
    """

    def __init__(self, id: int, x: float, y: float, height=None):
        """
        Args:
            id: representing unique node ID
            x: x-coordinate in city reference system
            y: y-coordinate in city reference system

        Returns:
            None
        """
        self.id = id
        self.x = x
        self.y = y
        self.height = height

class LaneSegment:
    def __init__(
        self,
        id: int,
        has_traffic_control: bool,
        turn_direction: str,
        is_intersection: bool,
        l_neighbor_id,
        r_neighbor_id,
        predecessors,
        successors,
        centerline: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            has_traffic_control:
            turn_direction: 'RIGHT', 'LEFT', or 'NONE'
            is_intersection: Whether or not this lane segment is an intersection
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            predecessors: The IDs of the lane segments that come after this one
            successors: The IDs of the lane segments that come before this one.
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.has_traffic_control = has_traffic_control
        self.turn_direction = turn_direction
        self.is_intersection = is_intersection
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline
        

# 编码车辆为0，行人为1
class Dataloader_root(DataLoader):
    def __init__(self, args, mode='train', isSource=True,
                 is_DA=False):
        self.args = args
        self.mode = mode
        self.isSource = isSource
        self.is_DA = is_DA

        self.ind_index2location_dict = {'location1': list(range(1,7)),
            'location2': list(range(7,18)),
            'location3': list(range(18,30)),
            'location4': list(range(30,33))}

        self.ind_location2index_dict = dict(zip(list(range(1,7))+\
                    list(range(7,18))+\
                    list(range(18,30))+\
                    list(range(30,33)),
                    ['location1']*6+['location2']*11+['location3']*12+['location4']*3))

        self.ind_global_location_utm_coord = {'location1':(297631.3187,5629917.34465),
                                              'location2':(293487.1224,5629711.58163),
                                              'location3':(295620.9575,5628102.04258),
                                              'location4':(300127.0853,5629091.0587)}
        
        if isSource:
            self.scenario = args.source
        else:
            self.scenario = args.target
        
        self.origin_data_root = os.path.join(data_root,'Data',self.scenario,mode,'data')
        self.origin_lanedata_root = os.path.join(data_root,'Data',self.scenario,mode,'lanelets')
        
        if isSource:
            self.preprocessed_data_root = os.path.join(data_root,f'Preprocessed_data',self.args.source, mode)
        else:
            self.preprocessed_data_root = os.path.join(data_root,f'Preprocessed_data',self.args.target, mode)
        
        from tool import graph_args, create_file
        
        create_file(self.preprocessed_data_root)
        self.num_node = graph_args['num_node']
        self.max_hop = graph_args['max_hop']

        # self.previous_length = args.previous_length
        # self.future_length = args.future_length
        self.TIMESTEP = args.TIMESTEP
        self.predicted_type = args.predicted_type

        self.mode = mode
        # self.previous_frame_horizen = self.previous_length // self.TIMESTEP  # 过去时间的间隔
        # self.future_frame_horizen = self.future_length // self.TIMESTEP
        # self.full_frame_horizen = self.previous_frame_horizen + self.future_frame_horizen

        self.previous_len = args.previous_length // (self.TIMESTEP*args.downsample_rate)
        self.future_len = args.future_length // (self.TIMESTEP*args.downsample_rate)
        self.full_len = self.previous_len + self.future_len
    
    def get_manu(self, xy):
        '''
        get_manu 的 Docstring
        
        :param self: 说明
        :param xy: np.array [T,3]  x+y+psi_degree/lane

        highd and ngsim: 根据lane来判断左转还是右转
        '''
        # 计算角加速度和速度
        horizon_s = xy.shape[0]*self.args.downsample_rate / 10
        theta_a = (np.linalg.norm(xy[1:,:2]-xy[:-1,:2],axis=-1)/ \
            self.args.downsample_rate / 10 / horizon_s).max()

        if theta_a > 2*THETA_ACC:
            lon_manu = 'ACC'
        elif theta_a < THETA_ACC:
            lon_manu = 'DEC'
        else:
            lon_manu = 'CON'
        
        if self.scenario in ['highd', 'ngsim']:
            origin_lane = xy[0,-1]
            candidate_lane = np.unique(xy[1:,-1])

            target_lane = random.choice(candidate_lane)
            if target_lane > origin_lane:
                lat_manu='LEFT'
            elif target_lane < origin_lane:
                lat_manu='RIGHT'
            else:
                lat_manu='STR'

        else:
            thete_degree = (xy[:,[-1]].max()-xy[:,[-1]].min()).item()
        
            if thete_degree > THETA_ANGLE:
                lat_manu = 'LEFT'
            elif THETA_ANGLE-270 < thete_degree < THETA_ANGLE:
                lat_manu = 'RIGHT'
            else:
                lat_manu = 'STR' 

        return MANU2ONEHOT[f'{lon_manu}_{lat_manu}']
    
    def get_lane_relation(self, map_fpath):
        e = xml.parse(map_fpath).getroot()
        way_dict = {}
        lane_param_dict = {'l_neighbor_id':'None',
                           "r_neighbor_id":'None'}

        # load way
        for way in e.findall('way'):
            way_dict[way.get('id')] = copy.deepcopy(lane_param_dict)

        content = dict()
        for relation in e.findall('relation'):
            content = dict()
            for member in relation.findall('member'):
                role = member.get('role')
                if role == 'left':
                    assert member.get('type') == 'way'
                    content['left'] = int(member.get('ref'))
                elif role == 'right':
                    assert member.get('type') == 'way'
                    content['right'] = int(member.get('ref'))
                elif role == 'regulatory_element':
                    assert member.get('type') == 'relation'
                    if 'regulations' not in content:
                        content['regulations'] = []
                    content['regulations'].append(int(member.get('ref')))
                # elif role == 'outer':
                #     assert member.get('type') == 'way'
                #     if 'outer' not in content:
                #         content['outer'] = []
                #     content['outer'].append(int(member.get('ref')))

            content['tags'] = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
            rtype = content['tags'].pop('type')
            rid = int(relation.get('id'))

            if rtype == "lanelet":
                if self.scenario == 'interaction':
                    lanelet = Lanelet_interaction(**content)
                else:
                    if 'left' in content and 'right' in content:
                        lanelet = Lanelet_ind(**content)
                    
                way_dict[str(lanelet.left)]['r_neighbor_id'] = str(lanelet.right)
                way_dict[str(lanelet.right)]['l_neighbor_id'] = str(lanelet.left)

        return way_dict
    
    def build_centerline_index(self, xml_fpath) :
        """
        Build dictionary of centerline for each city, with lane_id as key

        Returns:
            city_lane_centerlines_dict:  Keys are city names, values are dictionaries
                                        (k=lane_id, v=lane info)
        """
        city_lane_centerlines_dict = {}
        city_lane_centerlines_dict[self.location_name] = self.load_lane_segments_from_xml(xml_fpath)

        return city_lane_centerlines_dict

    def extract_node_from_ET_element(self, child):
        """
        Given a line of XML, build a node object. The "node_fields" dictionary will hold "id", "x", "y".
        The XML will resemble:

            <node id="0" x="3168.066310258233" y="1674.663991981186" />

        Args:
            child: xml.etree.ElementTree element

        Returns:
            Node object
        """
        node_fields = child.attrib
        node_id = int(node_fields["id"])
        if "height" in node_fields.keys():
            return Node(
                id=node_id,
                x=float(node_fields["x"]),
                y=float(node_fields["y"]),
                height=float(node_fields["height"]),
            )
        
        if 'location' in self.location_name:        
            utm_x, utm_y, _, _ = utm.from_latlon(float(node_fields["lat"]),
                                            float(node_fields["lon"]), 32, 'U')
            
            local_x, local_y = utm_x-self.ind_global_location_utm_coord[self.location_name][0],\
                                utm_y-self.ind_global_location_utm_coord[self.location_name][1]
        else:
            local_x, local_y = float(node_fields["x"]), float(node_fields["y"])
            
        return Node(id=node_id, x=local_x, y=local_y)
    
    def load_lane_segments_from_xml(self, map_fpath):
        """
        Load lane segment object from xml file

        Args:
           map_fpath: path to xml file

        Returns:
           lane_objs: List of LaneSegment objects
        """
        e = xml.parse(map_fpath).getroot()

        all_graph_nodes = {}
        lane_objs = {}
        # all children are either Nodes or Ways
        for child in list(e):
            if child.tag == "node":
                node_obj = self.extract_node_from_ET_element(child)
                all_graph_nodes[node_obj.id] = node_obj
            elif child.tag == "way":
                lane_obj, lane_id = self.extract_lane_segment_from_ET_element(child, all_graph_nodes)
                lane_objs[lane_id] = lane_obj

        return lane_objs
    
    def extract_lane_segment_from_ET_element(
            self, child, all_graph_nodes
    ):
        """
        We build a lane segment from an XML element. A lane segment is equivalent
        to a "Way" in our XML file. Each Lane Segment has a polyline representing its centerline.
        The relevant XML data might resemble::

            <way lane_id="9604854">
                <tag k="has_traffic_control" v="False" />
                <tag k="turn_direction" v="NONE" />
                <tag k="is_intersection" v="False" />
                <tag k="l_neighbor_id" v="None" />
                <tag k="r_neighbor_id" v="None" />
                <nd ref="0" />
                ...
                <nd ref="9" />
                <tag k="predecessor" v="9608794" />
                ...
                <tag k="predecessor" v="9609147" />
            </way>

        Args:
            child: xml.etree.ElementTree element
            all_graph_nodes

        Returns:
            lane_segment: LaneSegment object
            lane_id
        """
        lane_obj = {}
        lane_id = self.get_lane_identifier(child)
        node_id_list = []
        for element in child:
            # The cast on the next line is the result of a typeshed bug.  This really is a List and not a ItemsView.
            # way_field = self.cast(List[Tuple[str, str]], list(element.items()))
            way_field = list(element.items())
            field_name = way_field[0][0]
            if field_name == "k":
                key = way_field[0][1]
                if key in {"predecessor", "successor"}:
                    self.append_additional_key_value_pair(lane_obj, way_field)
                else:
                    self.append_unique_key_value_pair(lane_obj, way_field)
            else:
                node_id_list.append(self.extract_node_waypt(way_field))

        lane_obj["centerline"] = self.convert_node_id_list_to_xy(node_id_list, all_graph_nodes)
        lane_segment = self.convert_dictionary_to_lane_segment_obj(lane_id, lane_obj)
        return lane_segment, lane_id

    def extract_node_waypt(self,
                           way_field) -> int:
        """
        Given a list with a reference node such as [('ref', '0')], extract out the lane ID.

        Args:
           way_field: key and node id pair to extract

        Returns:
           node_id: unique ID for a node waypoint
        """
        key = way_field[0][0]
        node_id = way_field[0][1]
        assert key == "ref"
        return int(node_id)


    def append_additional_key_value_pair(self, lane_obj, way_field) :
        """
        Key name was either 'predecessor' or 'successor', for which we can have multiple.
        Thus we append them to a list. They should be integers, as lane IDs.

        Args:
           lane_obj: lane object
           way_field: key and value pair to append

        Returns:
           None
        """
        assert len(way_field) == 2
        k = way_field[0][1]
        v = int(way_field[1][1])
        lane_obj.setdefault(k, []).append(v)
        
    def append_unique_key_value_pair(self, lane_obj,
                                     way_field) -> None:
        """
        For the following types of Way "tags", the key, value pair is defined only once within
        the object:
            - has_traffic_control, turn_direction, is_intersection, l_neighbor_id, r_neighbor_id

        Args:
           lane_obj: lane object
           way_field: key and value pair to append

        Returns:
           None
        """
        assert len(way_field) == 2
        k = way_field[0][1]
        v = way_field[1][1]
        lane_obj[k] = v

    def get_lane_identifier(self, child) -> int:
        """
        Fetch lane ID from XML ET.Element.

        Args:
           child: ET.Element with information about Way

        Returns:
           unique lane ID
        """
        if 'lane_id' in child.attrib:
            return int(child.attrib["lane_id"])
        elif 'id' in child.attrib:
            return int(child.attrib["id"])
        else:
            assert False
            
    def convert_dictionary_to_lane_segment_obj(self, lane_id: int, lane_dictionary) :
        """
        Not all lanes have predecessors and successors.

        Args:
           lane_id: representing unique lane ID
           lane_dictionary: dictionary with LaneSegment attributes, not yet in object instance form

        Returns:
           ls: LaneSegment object
        """
        predecessors = lane_dictionary.get("predecessor", None)
        successors = lane_dictionary.get("successor", None)
        # has_traffic_control = self.str_to_bool(lane_dictionary["has_traffic_control"])
        # is_intersection = self.str_to_bool(lane_dictionary["is_intersection"])
        has_traffic_control = False
        is_intersection = False

        lnid = self.lane_relation[self.location_name][str(lane_id)]["l_neighbor_id"]
        rnid = self.lane_relation[self.location_name][str(lane_id)]["r_neighbor_id"]
        l_neighbor_id = None if lnid == "None" else int(lnid)
        r_neighbor_id = None if rnid == "None" else int(rnid)
        # ls = LaneSegment(
        #     lane_id,
        #     has_traffic_control,
        #     lane_dictionary["turn_direction"],
        #     is_intersection,
        #     l_neighbor_id,
        #     r_neighbor_id,
        #     predecessors,
        #     successors,
        #     lane_dictionary["centerline"],
        # )
        ls = LaneSegment(
            lane_id,
            has_traffic_control,
            None,
            is_intersection,
            l_neighbor_id,
            r_neighbor_id,
            predecessors,
            successors,
            lane_dictionary["centerline"],
        )

        return ls
    
    def convert_node_id_list_to_xy(self,
                                   node_id_list, all_graph_nodes) -> np.ndarray:
        """
        convert node id list to centerline xy coordinate

        Args:
           node_id_list: list of node_id's
           all_graph_nodes: dictionary mapping node_ids to Node

        Returns:
           centerline
        """
        num_nodes = len(node_id_list)

        if all_graph_nodes[node_id_list[0]].height is not None:
            centerline = np.zeros((num_nodes, 3))
        else:
            centerline = np.zeros((num_nodes, 2))
        for i, node_id in enumerate(node_id_list):
            if all_graph_nodes[node_id].height is not None:
                centerline[i] = np.array(
                    [
                        all_graph_nodes[node_id].x,
                        all_graph_nodes[node_id].y,
                        all_graph_nodes[node_id].height,
                    ]
                )
            else:
                centerline[i] = np.array([all_graph_nodes[node_id].x, 
                                          all_graph_nodes[node_id].y])

        return centerline
    
    def getLaneData(self):
        location_name_list, path_list = self.getFullLaneData_list()

        self.city_name_to_city_id_dict = {}
        self.lane_relation = {}
        self.city_lane_centerlines_dict = {}
        
        for location_name, path in zip(location_name_list, path_list):
            self.location_name = location_name
            
            self.city_name_to_city_id_dict[location_name] = {}
            self.lane_relation[location_name] = self.get_lane_relation(path)
            self.city_lane_centerlines_dict = {**self.city_lane_centerlines_dict,
                                               **self.build_centerline_index(path)}
            
            (self.city_halluc_bbox_table, \
                self.city_halluc_tablelaneid_to_index_map,\
                self.city_halluc_tableindex_to_laneid_map) = self.build_hallucinated_lane_bbox_index()
            self.city_rasterized_ground_height_dict = self.build_city_ground_height_index()

    def build_hallucinated_lane_bbox_index(
            self,
    ):
        """
        Populate the pre-computed hallucinated extent of each lane polygon, to allow for fast
        queries.

        Returns:
            city_halluc_bbox_table
            city_id_to_halluc_tableidx_map
        """

        city_halluc_bbox_table = {}     # bbox
        city_halluc_tablelaneid_to_index_map = {} # mapping from index (str) to lane_id (int)
        city_halluc_tableindex_to_laneid_map = {}

        for city_name, city_id in self.city_name_to_city_id_dict.items():
            centerline = np.zeros((0,4))
            for index, lane_id in enumerate(self.city_lane_centerlines_dict[city_name].keys()):
                centerline = np.concatenate([centerline,\
                                            np.concatenate([self.city_lane_centerlines_dict[city_name][lane_id].centerline.min(0),\
                                                            self.city_lane_centerlines_dict[city_name][lane_id].centerline.max(0)])[None]])

            city_halluc_bbox_table[city_name] = centerline
            city_halluc_tablelaneid_to_index_map[city_name] = dict(zip(self.city_lane_centerlines_dict[city_name].keys(),
                                                                    range(len(self.city_lane_centerlines_dict[city_name].keys()))))
            city_halluc_tableindex_to_laneid_map[city_name] = dict(zip(city_halluc_tablelaneid_to_index_map[city_name].values(),
                                                                   city_halluc_tablelaneid_to_index_map[city_name].keys()))
            
        return city_halluc_bbox_table, city_halluc_tablelaneid_to_index_map, \
            city_halluc_tableindex_to_laneid_map
    
    def build_city_ground_height_index(self):
        """
        Build index of rasterized ground height.

        Returns:
            city_ground_height_index: a dictionary of dictionaries. Key is city_name, and values
                    are dictionaries that store the "ground_height_matrix" and also the
                    city_to_pkl_image_se2: SE(2) that produces takes point in pkl image to city
                    coordinates, e.g. p_city = city_Transformation_pklimage * p_pklimage
        """
        city_rasterized_ground_height_dict = {}
        for city_name, city_id in self.city_name_to_city_id_dict.items():
            city_rasterized_ground_height_dict[city_name] = {}

            # load the file with rasterized values
            city_rasterized_ground_height_dict[city_name]["ground_height"] = np.zeros((3674,1482))
            city_rasterized_ground_height_dict[city_name]["npyimage_to_city_se2"] = np.eye(3)

        return city_rasterized_ground_height_dict
    
    def process_origin_csv(self, csv_path, unified_column=["frame", "ped", "x", "y"],
                          is_DA=False):

        if self.scenario == 'interaction':
            veh_data = pd.read_csv(csv_path[0])

            # 需要把agent_type改成0 1编码，车辆为0，行人为1
            veh_data['agent_type'] = veh_data['agent_type'].str.replace('car', '0')
            veh_data['agent_type'] = veh_data['agent_type'].astype(int)
            veh_data['track_id'] = veh_data['track_id'].astype(int)
            if os.path.exists(csv_path[1]):
                ped_data = pd.read_csv(csv_path[1])
                ped_data['track_id'] = ped_data['track_id'].str.lstrip('P')
                ped_data['length'] = 0
                ped_data['width'] = 0
                ped_data['psi_rad'] = 0
                ped_data['agent_type'] = ped_data['agent_type'].str.replace('pedestrian/bicycle', '1')
                ped_data['agent_type'] = ped_data['agent_type'].astype(int)
                ped_data['track_id'] = ped_data['track_id'].astype(int)
                # 重新打编号
                max_veh_track_id = veh_data['track_id'].max()
                ped_data['track_id'] += max_veh_track_id
                record_data = pd.concat([veh_data, ped_data], axis=0)
            else:
                record_data = veh_data
            
            # 在这里就进行降采样了
            max_frame = record_data['frame_id'].max()
            min_frame = record_data['frame_id'].min()
            target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))

            record_data = record_data.set_index(['frame_id'])
            valid_indices = record_data.index.intersection(target_indices)
            # 筛选数据
            record_data = record_data.loc[valid_indices, :].reset_index()

            # 调整顺序
            record_data = record_data[['frame_id', 'track_id', 'agent_type', 'x',
                                                'y']].sort_values(['frame_id', 'track_id'])

            df = record_data[['frame_id','track_id','y','x']]            

        elif self.scenario == 'highd':
            track_df = pd.read_csv(csv_path[0])
            track_meta_df = pd.read_csv(os.path.join(csv_path[1]))

            df, track_meta_df = self.downsample_highd(track_df, track_meta_df)

            # 调整顺序
            df = df[['frame','id','x','y']]

        elif self.scenario == 'ind':
            df = pd.read_csv(csv_path[0])
            track_meta_df = pd.read_csv(csv_path[1])

            df, track_meta_df = self.downsample_ind(df, track_meta_df)

            # 调整顺序
            df = df[['frame','trackId','yCenter','xCenter']]
            del track_meta_df

        elif self.scenario == 'ngsim':
            df = pd.read_csv(csv_path[0])
            df = self.downsample_ngsim(df)

            # 调整顺序
            df = df.sort_values(['Frame_ID', 'Vehicle_ID'])

        else:
            assert False
        
        df.columns = unified_column
        df =  df.sort_values([unified_column[0],
                             unified_column[1]])
        return df
    
    def downsample_ind(self, df, track_meta_df):
        df = df[df['frame']%4==1]     # 和interaction数据集作统一
        df = pd.merge(df, track_meta_df, on=['trackId','recordingId'], how='left')

        # 在这里就进行降采样了
        frame_map = dict(zip(sorted(df['frame'].unique()), 
                                list(range(1,len(sorted(df['frame'].unique()))+1))))

        # 筛选数据
        df['frame'] = df['frame'].map(frame_map)
        # 在这里就进行降采样了
        max_frame = df['frame'].max()
        min_frame = df['frame'].min()
        target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))
        df = df.set_index(['frame'])
        valid_indices = df.index.intersection(target_indices)
        # 筛选数据
        df = df.loc[valid_indices, :].reset_index()

        return df, track_meta_df

    def downsample_ngsim(self, df):
        df = df[['Frame_ID','Vehicle_ID','Global_Y','Global_X']]
        # 在这里就进行降采样了
        frame_map = dict(zip(sorted(df['Frame_ID'].unique()), 
                                list(range(1,len(sorted(df['Frame_ID'].unique()))+1))))

        # 筛选数据
        df['Frame_ID'] = df['Frame_ID'].map(frame_map)
        # 在这里就进行降采样了
        max_frame = df['Frame_ID'].max()
        min_frame = df['Frame_ID'].min()
        target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))
        df = df.set_index(['Frame_ID'])
        valid_indices = df.index.intersection(target_indices)

        # df = df.set_index(['Frame_ID'])
        # valid_indices = df.index.intersection(target_indices)
        # 筛选数据
        df = df.loc[valid_indices, :].reset_index()

        return df

    def downsample_highd(self, track_df, track_meta_df):
        track_df = track_df[track_df['frame']%4==1]     # 和interaction数据集作统一
        df = pd.merge(track_df, track_meta_df, on=['id'], how='left')
        del track_df
        # 需要把agent_type改成0 1编码，车辆为0，行人为1
        df['timestamp_ms'] = df['frame']*100
        df = df[['id','frame','timestamp_ms','class','x','y',
                'xVelocity','yVelocity','drivingDirection','height_y','width_x']]

        df = df[['frame','id','y','x']]
        frame_map = dict(zip(sorted(df['frame'].unique()), 
                                list(range(1,len(sorted(df['frame'].unique()))+1))))

        # 筛选数据
        df['frame'] = df['frame'].map(frame_map)

        # 在这里就进行降采样了
        max_frame = df['frame'].max()
        min_frame = df['frame'].min()
        target_indices = list(range(min_frame, max_frame, self.args.downsample_rate))
        df = df.set_index(['frame'])
        valid_indices = df.index.intersection(target_indices)

        # 筛选数据
        df = df.loc[valid_indices, :].reset_index()
                    
        return df, track_meta_df

    def getFullLaneData_list(self):
        location_name_list = []
        path_list = []
        if self.scenario == 'ind':
            for lane_file in list(filter(lambda x:x.endswith('.osm'),
                                         os.listdir(self.origin_lanedata_root))):
                path_list.append(os.path.join(self.origin_lanedata_root,
                                              lane_file))
                location_name_list.append(lane_file[:-4])
                
        elif self.scenario == 'interaction':
            for lane_file in list(filter(lambda x:x.endswith('.osm_xy'),
                                         os.listdir(self.origin_lanedata_root))):
                path_list.append(os.path.join(self.origin_lanedata_root,
                                              lane_file))
                location_name_list.append(lane_file[:-7])
        
        return location_name_list, path_list
            
    def getFullData_list(self, is_DA):
        file_list = []

        if self.scenario == 'interaction':
            for s in os.listdir(self.origin_data_root):
                file_len = max(list(map(lambda x: int(x), filter(lambda x: x != '', map(lambda x: x[-7:-4].lstrip('0'),
                                                                                    os.listdir(os.path.join(self.origin_data_root,f'{s}')))))))
                for record in range(1, file_len + 1):
                    r = '%03d' % record
                    file_list.append((os.path.join(self.origin_data_root,s,f'vehicle_tracks_{r}.csv'),
                                    os.path.join(self.origin_data_root,s,f'pedestrian_tracks_{r}.csv')))

        elif self.scenario == 'ind':
            file_name_list = list(filter(lambda x:'tracks.csv' in x, os.listdir(self.origin_data_root)))
            for file_name in file_name_list:
                index = file_name[:2]
                # print(f'正在处理 ind:{file_name}')
                file_list.append((os.path.join(self.origin_data_root, file_name),
                                  os.path.join(self.origin_data_root, 
                                                     f'{index}_tracksMeta.csv')))
        
        elif self.scenario == 'highd':
            file_name_list = list(filter(lambda x:'tracks.csv' in x, os.listdir(self.origin_data_root)))
            for file_name in file_name_list:
                index = file_name[:2]
                # print(f'正在处理 highd:{file_name}')
                file_list.append((os.path.join(self.origin_data_root, file_name),
                                 os.path.join(self.origin_data_root, f'{index}_tracksMeta.csv')))
        
        elif self.scenario == 'ngsim':
            scenario_list = os.listdir(self.origin_data_root)
            for s in scenario_list:
                file_name_list = os.listdir(os.path.join(self.origin_data_root, s))
                for file_name in file_name_list:
                    # print(f'正在处理  ngsim:{s} file_name:{file_name}')
                    file_list.append([os.path.join(self.origin_data_root,s,file_name),
                                      None])
        else:
            assert False

        if not self.isSource and is_DA:
            target_file_len = ceil(len(file_list)*self.args.DA_ratio)
            return file_list[:target_file_len]
        else:
            return file_list

    # def __len__(self):
    #     return len(self.item_data)

    # # 不用清洗getItem,当frame大于间隔时即可枚举
    # def __getitem__(self, idx):
    #     s1 = time.time()

    #     frame = self.item_data.iloc[idx, 0]
    #     track_id = self.item_data.iloc[idx, 1]
    #     agent_type = self.item_data.iloc[idx, 3]
    #     if self.predicted_type == 0 or self.predicted_type == 1:
    #         assert agent_type == self.predicted_type
    #     # source_id = self.item_data.iloc[idx, 'track_id']     # 抽到谁,谁就是source

    #     # 获取了当前的frame预测上下限以后,在完整的数据集中去取.获取x_seq
    #     previous_frame = frame - self.previous_length // self.TIMESTEP + 1
    #     future_frame = frame + self.args.future_length // self.TIMESTEP

    #     test_time = time.time()
    #     data_df = self.full_data.loc[(slice(previous_frame,future_frame)),:]
    #     ego_data = data_df[data_df['track_id'] == track_id]
    #     # print(f'{time.time() - test_time}')
    #     assert ego_data.shape[0] == self.frame_horizen

    #     ego_angle = t.tensor([ego_data['psi_rad'].mean()]).to(device)
    #     # print(f'Part 1:{time.time() - s1}')
    #     s2 = time.time()

    #     if self.mode == 'test':
    #         BU_data_df = data_df[data_df['track_id'] != track_id]
    #         BU_data_list = list(BU_data_df[['x', 'y', 'track_id']].groupby('track_id'))

    #         if BU_data_list:
    #             BU_data = t.stack(list(map(lambda x:t.from_numpy(x[1][['x','y']].values), BU_data_list)))   # [agents, T, 2]
    #             self.BU_data_bs.append(BU_data)
    #         else:
    #             self.BU_data_bs.append(t.zeros((0,ego_data.shape[0], ego_data.shape[1])).to(device))

    #     ego_data = t.from_numpy(ego_data.loc[:, ['x', 'y']].values).to(device).unsqueeze(0).float()  # [1,30,2]
    #     if self.isRotate:
    #         ego_data, rotate_matrix = self.rotate(ego_data, ego_angle)
    #     else:
    #         rotate_matrix = t.tensor([[[1,0],
    #                                   [0,1]] for _ in range(ego_data.shape[0])])
    #     # print(f'Part 2:{time.time() - s2}')
    #     return ego_data, rotate_matrix

    # def collate_fn(self, batch):
    #     s3 = time.time()
    #     data_tuple = tuple(map(lambda x:x[0],batch))
    #     rotate_matrix_tuple = tuple(map(lambda x:x[1],batch))

    #     data_bs = t.stack(data_tuple)       # [bs,T,2]
    #     rotate_matrix_bs = t.stack(rotate_matrix_tuple) # [bs,2,2]

    #     if t.cuda.is_available():
    #         data_bs = data_bs.cuda()
    #         rotate_matrix_bs = rotate_matrix_bs.cuda()

    #     data_bs = data_bs.squeeze(1)
    #     rotate_matrix_bs = rotate_matrix_bs.squeeze(1)
    #     # print(f'Part 3:{time.time() - s3}')

    #     s4 = time.time()
    #     if self.mode == 'train':
    #         # print(f'Part 4:{time.time() - s4}')
    #         return data_bs, rotate_matrix_bs
    #     else:
    #         bs_list = list(map(lambda x:x.shape[0],self.BU_data_bs))
    #         BU_data = t.cat(list(filter(lambda x:x.shape[0],self.BU_data_bs)), dim=0).to(device)
    #         self.BU_data_bs = []
    #         # print(f'Part 4:{time.time() - s4}')

    #         return data_bs, rotate_matrix_bs, BU_data, bs_list

    # # 获得完整的数据
    # def getFullData(self, scenario, agent_type='v'):
    #     assert isinstance(agent_type, str) or isinstance(agent_type, list)

    #     full_data = pd.DataFrame()
    #     if isinstance(scenario, str):
    #         file_len = max(list(map(lambda x: int(x), filter(lambda x: x != '', map(lambda x: x[-7:-4].lstrip('0'),
    #                                                                                 os.listdir(
    #                                                                                     rf'Data/origindata/{scenario}'))))))
    #         full_data = pd.DataFrame()
    #         for record in range(file_len + 1):
    #             r = '%03d' % record
    #             if agent_type.lower() == 'p':
    #                 file_name = rf'Data/origindata/{scenario}/pedestrian_tracks_{r}.csv'
    #             elif agent_type.lower() == 'v':
    #                 file_name = rf'Data/origindata/{scenario}/vehicle_tracks_{r}.csv'
    #             data = pd.read_csv(file_name)
    #             assert len(data['agent_type'].unique()) == 1

    #             # 得打个record标注，不然后面不知道是哪个record的数据
    #             data['record'] = record
    #             # 需要把agent_type改成0 1编码，车辆为0，行人为1
    #             data['agent_type'] = data['agent_type'].str.replace('car', '0')
    #             data['agent_type'] = data['agent_type'].str.replace('pedestrian/bicycle', '1')
    #             data['agent_type'] = data['agent_type'].astype(int)
    #             # 把track_id的格式进行修改,添加加速度
    #             if agent_type.lower() == 'p':
    #                 data['track_id'] = data['track_id'].str.lstrip('P')
    #                 data['length'] = 0
    #                 data['width'] = 0

    #             data['a'] = ((data.groupby(['track_id'])['vx'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP + (
    #                 data.groupby(['track_id'])['vy'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP).apply(sqrt)
    #             data['ax'] = ((data.groupby(['track_id'])['vx'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP).apply(sqrt)
    #             data['ay'] = ((data.groupby(['track_id'])['vy'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP).apply(sqrt)
    #             data['track_id'] = data['track_id'].astype(int)
    #             data['scenario'] = scenario
    #             full_data = pd.concat([full_data, data], axis=0)
    #             full_data['scenario'] = scenario

    #     elif isinstance(scenario, np.ndarray) or isinstance(scenario, list):
    #         for s in scenario:
    #             file_len = max(list(map(lambda x: int(x), filter(lambda x: x != '', map(lambda x: x[-7:-4].lstrip('0'),
    #                                                                                     os.listdir(
    #                                                                                         rf'Data/origindata/{s}/{self.mode}'))))))
    #             for record in range(1, file_len + 1):
    #                 r = '%03d' % record
    #                 if agent_type.lower() == 'p':
    #                     file_name = rf'Data/origindata/{s}/{self.mode}/pedestrian_tracks_{r}.csv'
    #                 elif agent_type.lower() == 'v':
    #                     file_name = rf'Data/origindata/{s}/{self.mode}/vehicle_tracks_{r}.csv'

    #                 # 如果是高速且枚举人, 那么直接跳出循环
    #                 if 'Merging' in file_name and agent_type.lower() == 'p':
    #                     break
    #                 data = pd.read_csv(file_name)
    #                 assert len(data['agent_type'].unique()) == 1

    #                 # 得打个record标注，不然后面不知道是哪个record的数据
    #                 data['record'] = record
    #                 # 需要把agent_type改成0 1编码，车辆为0，行人为1
    #                 data['agent_type'] = data['agent_type'].str.replace('car', '0')
    #                 data['agent_type'] = data['agent_type'].str.replace('pedestrian/bicycle', '1')
    #                 data['agent_type'] = data['agent_type'].astype(int)
    #                 # 把track_id的格式进行修改,添加加速度
    #                 if agent_type.lower() == 'p':
    #                     data['track_id'] = data['track_id'].str.lstrip('P')
    #                     data['length'] = 0
    #                     data['width'] = 0

    #                 data['a'] = ((data.groupby(['track_id'])['vx'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP + (
    #                     data.groupby(['track_id'])['vy'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP).apply(sqrt)
    #                 data['ax'] = ((data.groupby(['track_id'])['vx'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP).apply(sqrt)
    #                 data['ay'] = ((data.groupby(['track_id'])['vy'].diff().fillna(0)) ** 2 / S2MS * self.TIMESTEP).apply(sqrt)
    #                 data['track_id'] = data['track_id'].astype(int)
    #                 data['scenario'] = s
    #                 full_data = pd.concat([full_data, data], axis=0)
    #     return full_data

    # # 获取用于枚举的Item数据
    # def getItemData(self, clean_data, result_frame):

    #     item_full_data = pd.DataFrame([])
    #     self.frame_horizen = self.previous_frame_horizen + self.future_frame_horizen       # 20个frame的总时间间隔
    #     clean_data_copy = copy.deepcopy(clean_data)
    #     clean_data_copy = clean_data_copy.set_index(['frame_id'])
    #     for start_frame, end_frame in result_frame:
    #         l_time = start_frame + self.previous_frame_horizen - 1
    #         r_time = end_frame - self.future_frame_horizen
    #         if self.predicted_type == 0 or self.predicted_type == 1:
    #             # getData = clean_data[(clean_data['frame_id'] >= l_time) & \
    #             #                    (clean_data['frame_id'] <= r_time) & \
    #             #                    (clean_data['agent_type'] == self.predicted_type)]
    #             getData = clean_data_copy.loc[(clean_data_copy.index>=l_time)&(clean_data_copy.index<=r_time)]
    #             getData = getData[getData['agent_type'] == self.predicted_type]
    #             item_full_data = pd.concat([item_full_data, getData], axis=0)
    #         else:
    #             # getData = clean_data[(clean_data['frame_id'] >= l_time) & \
    #             #                      (clean_data['frame_id'] <= r_time)]
    #             getData = clean_data_copy.loc[(clean_data_copy.index >= l_time) & (clean_data_copy.index <= r_time)]
    #             item_full_data = pd.concat([item_full_data, getData], axis=0)

    #         # 保证每次枚举都有数据, 否则就是在选择帧的时候出错了
    #         assert getData.shape[0] != 0

    #     item_full_data = item_full_data.reset_index()
    #     del clean_data_copy
    #     return item_full_data

    # # 清洗数据
    # def get_effective_frame_range(self, full_data):
    #     '''
    #     a = full_data.groupby(['frame_id'], as_index=True)
    #     1.连续帧筛选
    #     b = a['track_id'].apply(lambda x:x.unique())
    #     b = pd.DataFrame(b)
    #     b['group_index'] = b.index - np.arange(b.shape[0])
    #     b = b.reset_index()

    #     2.agent_type筛选
    #     c = a[['agent_type','frame_id']].filter(lambda x:self.predicted_type in x['agent_type'].unique())
    #     c = pd.DataFrame(c)

    #     3.结合1 2筛选
    #     d = b.loc[b.index.isin(c['frame_id'])]
    #     d['track_id'] = d['track_id'].astype(str)
    #     result_frame = d.groupby(['group_index', 'track_id']).apply(lambda x: [x.index.values[0],x.index.values[-1]]).values
    #     result_frame = list(filter(lambda x: x[1] - x[0] >= self.frame_horizen, result_frame))
    #     '''
    #     s1 = time.time()
    #     a = full_data.groupby(['frame_id'])
    #     per_frame_df = list(a)    # 主要是这个比较久
    #     self.frame_horizen = (self.previous_length + self.future_length) // self.TIMESTEP
    #     last_agentId_set = set(per_frame_df[0][1]['track_id'])
    #     result_frame = []
    #     start_frame = full_data['frame_id'].min()
    #     end_frame = start_frame
    #     print(f'groupby耗时:{time.time() - s1}')

    #     s2 = time.time()
    #     # 从第二个开始遍历每一个分好组的frame
    #     for index, (frame, frame_df) in enumerate(per_frame_df[1:],start=1):
    #         if not isinstance(frame, int):
    #             frame = frame[0]
    #         ## 根据所预测的对象判断区域末端的终止条件, 如果是车辆或人
    #         if self.predicted_type == 0 or self.predicted_type == 1:
    #             # 假如当前frame中出现的个体编号和上次出现的编号均相同, 并且个体中需要出现车辆或人
    #             if (set(per_frame_df[index][1]['track_id']) == last_agentId_set) and \
    #                 self.predicted_type in per_frame_df[index][1]['agent_type'].unique():
    #                 end_frame = frame

    #             # 如果不相同,则需要保存结果
    #             else:
    #                 result_frame.append([start_frame, end_frame])
    #                 last_agentId_set = set(per_frame_df[index][1]['track_id'])
    #                 start_frame = frame
    #                 end_frame = frame
    #         else:
    #             if (set(per_frame_df[index][1]['track_id']) == last_agentId_set) and \
    #                 self.predicted_type in per_frame_df[index][1]['agent_type'].unique():
    #                 end_frame = frame

    #             # 如果不相同,则需要保存结果
    #             else:
    #                 result_frame.append([start_frame, end_frame])
    #                 last_agentId_set = set(per_frame_df[index][1]['track_id'])
    #                 start_frame = frame
    #                 end_frame = frame

    #     # 筛选出result_frame数量小于horizen的
    #     result_frame = list(filter(lambda x: x[1] - x[0] >= self.frame_horizen, result_frame))
    #     print(f'循环分组耗时:{time.time() - s2}')

    #     return result_frame

    # # 获取obj过去数据
    # def getPreviousTrack(self, obj_track_tensor):
    #     # 总窗口长度可以通过args两个参数获取
    #     previous_index_length = self.previous_length // self.TIMESTEP
    #     return obj_track_tensor[:, :previous_index_length, :]

    # # 获取obj未来数据
    # def getFutureTrack(self, obj_track_tensor):
    #     previous_index_length = self.args.future_length // self.TIMESTEP
    #     return obj_track_tensor[:, previous_index_length:, :]

    # # 划分训练集，测试集和验证集
    # def getSubDataset(self, data,
    #                   TRAINDATASET_RATIO = TRAINDATASET_RATIO,
    #                   TESTDATASET_RATIO = TESTDATASET_RATIO
    #                   ):

    #     train_dataset = data.sample(frac=TRAINDATASET_RATIO, random_state=self.args.seed)
    #     data = data.drop(labels=train_dataset.index)
    #     test_dataset = data.sample(frac=TESTDATASET_RATIO, random_state=self.args.seed)

    #     return train_dataset, test_dataset

    # # resort_frameId_and_trackId
    # def resort_frameId_and_trackId(self, item_full_data_df):
    #     '''
    #     item_full_data_df:这个是在原始dataset中用来枚举的对象,在这里我们只需要根据agent_id和record对track_id进行修改即可
    #     '''

    #     item_full_data_df['track_id'] = item_full_data_df['track_id'].astype(np.int64)
    #     full_data = pd.DataFrame(columns=item_full_data_df.columns)
    #     # track_id不管是否连续,只需要管frame_id即可
    #     last_frame, last_track = 0, 0
    #     for s in self.scenario:
    #         for record in range(1, item_full_data_df.loc[item_full_data_df['scenario'] == s,'record'].max()+1):

    #             record_veh_data = item_full_data_df[(item_full_data_df['scenario'] == s) &
    #                                                 (item_full_data_df['record'] == record) &
    #                                                 (item_full_data_df['agent_type'] == 0)]
    #             record_ped_data = item_full_data_df[(item_full_data_df['scenario'] == s) &
    #                                                 (item_full_data_df['record'] == record) &
    #                                                 (item_full_data_df['agent_type'] == 1)]
    #             record_ped_data['track_id'] += (record_veh_data['track_id'].max() + last_track)
    #             per_record_data = pd.concat([record_veh_data, record_ped_data], axis=0)

    #             per_record_data['frame_id'] += last_frame
    #             assert per_record_data.shape[0] == item_full_data_df[(item_full_data_df['scenario'] == s) &
    #                                                 (item_full_data_df['record'] == record)].shape[0]

    #             full_data = pd.concat([full_data, per_record_data], axis=0)
    #             last_track = full_data['track_id'].max()
    #             last_frame = full_data['frame_id'].max()

    #     # 保证输出输出相同
    #     assert item_full_data_df.shape[0] == full_data.shape[0]
    #     return full_data

    # # Preprocessing
    # def rotate(self, traj_tensor, psi_rad_bs):
    #     '''
    #     traj_tensor: [nums, T, 2]
    #     psi_rad_bs: [nums]
    #     '''
    #     assert isinstance(traj_tensor, t.Tensor) and isinstance(psi_rad_bs, t.Tensor)
    #     assert traj_tensor.ndim == 3 and psi_rad_bs.ndim == 1

    #     rotate_matrix = t.tensor([[[cos(psi_rad), sin(psi_rad)],
    #                                [-sin(psi_rad), cos(psi_rad)]] for psi_rad in psi_rad_bs]).to(device)

    #     traj_tensor_rotate = traj_tensor @ rotate_matrix
    #     return traj_tensor_rotate, rotate_matrix

    # def getAction(self, ego_df):
    #     start_psi_rad = ego_df['psi_rad'].iloc[0]
    #     end_psi_rad = ego_df['psi_rad'].iloc[-1]

    #     rotate_deg = (end_psi_rad-start_psi_rad)*180/np.pi
    #     degree_thr = 25
    #     if rotate_deg > degree_thr:
    #         return 'left'
    #     elif rotate_deg < -degree_thr:
    #         return 'straight'
    #     else:
    #         return 'right'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='param for LSTM model')
    # scenario
    parser.add_argument('--scenario', type=list, default=['DR_DEU_Roundabout_OF'],
                        help="DR_DEU_Roundabout_OF, DR_USA_Intersection_EP0,\
                                  'DR_USA_Intersection_EP1', 'DR_USA_Intersection_GL',\
                                  'DR_USA_Intersection_MA', 'DR_USA_Roundabout_EP',\
                                  'DR_USA_Roundabout_FT', 'DR_USA_Roundabout_SR'")

    # Preprocessing
    parser.add_argument('--rotate', type=bool, default=False,
                        help='rotate for trajectory')

    # Training params
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random number generators')
    parser.add_argument('--train_epoches', type=int, default=300,
                        help='iterations to run and train agent')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--batch_size', type=float, default=32,
                        help='batchsize')
    parser.add_argument('--predicted_type', type=float, default=0,
                        help='predicted_type, 0:vehicle 1:pedestrain 2:all')
    parser.add_argument('--Attack', type=bool, default=False,
                        help='Under Attack?')
    parser.add_argument('--Attack_type', type=str, default='None',
                        help='Social attack, Random attack')

    # Time params
    parser.add_argument('--previous_length', type=int, default=1000,
                        help='previous length horizen, unit:ms')
    parser.add_argument('--future_length', type=int, default=2000,
                        help='previous length horizen, unit:ms')
    parser.add_argument('--future_list', type=list, default=[500, 1000, 1500, 2000],
                        help='future length horizen, unit:ms')
    parser.add_argument('--downsample_rate', type=int, default=1,
                        help='downsample_rate')
    parser.add_argument('--TIMESTEP', type=int, default=100,
                        help='time step, unit:ms')
    parser.add_argument('--dir_name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                        help='time step, unit:ms')

    # Saving params
    parser.add_argument('--save_per_epoch', type=int, default=10,
                        help='iterations to run and train agent')

    # Attack params
    parser.add_argument('--attack_thresold', type=float, default=0.3,
                        help='attack thresold')
    args = parser.parse_args()
    Dataset = Dataloader_root(args, mode='test')
    Dataloader_test = DataLoader(Dataset, shuffle=True, batch_size=32,
                                 collate_fn=Dataset.collate_fn)
    Dataloader_train = DataLoader(Dataloader_root(args, mode='train'), shuffle=True, batch_size=32)
    print(len(Dataloader_train), len(Dataloader_test), round(len(Dataloader_train)/len(Dataloader_test),3))
    for d in tqdm(Dataloader_test):
        print()

    # for s in [
    #           'DR_USA_Roundabout_FT', ]:
    #     for mode in ['train','test']:
    #         root = f'..\\data\\origindata\\{s}\\{mode}'
    #         l = list(np.unique(list(map(lambda x: int(x[-7:-4]), os.listdir(root)))))
    #         num = max(l)
    #         whileBreak = True
    #         while whileBreak:
    #             for file in os.listdir(root):
    #                 if num == int(file[-7:-4]):
    #                     rename_index = '%03d' % (int(file[-7:-4]) +1)
    #                     os.rename(os.path.join(root, file),
    #                               os.path.join(root, file[:-7]+rename_index+file[-4:]))
    #                     if file.startswith('v'):
    #                         num -= 1
    #             if num == 0:
    #                 whileBreak = False
    #                 break
