import yaml
import json
from .meva_activity_mapping import MevaActivityType, MevaActivityMapping


class MevaAnnotationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MevaVideoAnnotation):
            return obj.to_dict()

        return super().default(obj)


def MevaAnnotationDecoder(dct):
    if isinstance(dct, dict) and "objects" in dct and "activities" in dct:
        annotation = MevaVideoAnnotation()
        annotation._annotations_dict = dct
        return annotation

    return dct


class MevaVideoAnnotation:
    """
    Class to load and parse the geometry annotations of the MEVA dataset and the VIRAT 2020 dataset.
    """

    def __init__(self, ann_dict: dict[str, dict[str, int | str | list]] | None = None):
        self._annotations_dict = ann_dict

    def __getitem__(self, key):
        return self._annotations_dict[key]

    def get(self, key, default=None):
        return self._annotations_dict.get(key, default)

    def __contains__(self, key):
        return key in self._annotations_dict

    def to_dict(self) -> dict:
        return self._annotations_dict

    @staticmethod
    def load_anno_lines(yaml_filepath: str):
        with open(yaml_filepath, "r") as f:
            lines = f.readlines()
            is_list = all([line.lstrip().startswith("-") for line in lines])

            if not is_list:
                raise ValueError("The YAML file must be a list of dictionaries.")

        return lines

    @staticmethod
    def parse_line(line):
        return yaml.safe_load(line)[0]  # Only one annotation per line

    @staticmethod
    def load_types_from_yml(yml_types_filepath: str) -> dict[int, str]:
        # Load the YAML file
        with open(yml_types_filepath, "r") as file:
            data = yaml.safe_load(file)

        # Initialize an empty dictionary
        result = {}

        # Iterate through the loaded data
        for item in data:
            if "types" in item:
                id1 = int(item["types"]["id1"])
                type_name = list(item["types"]["cset3"].keys())[0]
                result[id1] = type_name

        return result

    @staticmethod
    def load_objects_from_yml(yml_filepath: str, types_dict: dict[int, str]) -> dict:
        with open(yml_filepath, "r") as file:
            data = yaml.safe_load(file)

        frame_objects = {}
        meta = []
        for raw_line in data:
            if "meta" in raw_line:
                if "was marked empty" in raw_line["meta"]:
                    continue

                meta.append(raw_line["meta"])
            elif "geom" in raw_line:
                frame_id = int(raw_line["geom"]["ts0"])
                bbox = raw_line["geom"]["g0"]
                id1 = int(raw_line["geom"]["id1"])  # Object ID

                if frame_id not in frame_objects:
                    frame_objects[frame_id] = []

                frame_objects[frame_id].append({
                    "id1": id1,
                    "type": types_dict[id1],
                    "bbox": [int(x) for x in bbox.split(" ")],
                })

        return frame_objects, meta

    @staticmethod
    def load_activities_from_yml(yaml_file_path: str) -> dict:
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)

        result = []

        for item in data:
            if "act" not in item:
                continue

            # Extract the main activity information
            act_data = item["act"]
            act_name_data = act_data["act2"]

            # Skip empty activity
            if "actors" not in act_data or len(act_data["actors"]) == 0:
                continue

            # Get the activity name (key of the act2 dictionary)
            activity_name = list(act_name_data.keys())[0]

            # Extract timespan - should be exactly one span
            timespans = act_data["timespan"]
            if len(timespans) != 1:
                raise Exception("Activity must have exactly one timespan. " +
                                f"Found {len(timespans)} in {activity_name} in file {yaml_file_path}")

            tsr_data = timespans[0]
            if len(tsr_data) != 1:
                raise Exception("Timespan must have exactly one tsr entry. " +
                                f"Found {len(tsr_data)} in {activity_name} in file {yaml_file_path}")

            timespan_values = list(tsr_data.values())[0]
            if len(timespan_values) != 2:
                raise Exception("A timespan must be of the form [start, end]. " +
                                f"Found {len(timespan_values)} values in {activity_name} in file {yaml_file_path}")

            # Extract actors information
            actors_data = act_data["actors"]
            actors = {}

            for actor in actors_data:
                actor_id = actor["id1"]
                actor_timespans = actor["timespan"]

                # Extract all timespans for this actor
                actor_spans = []
                for timespan in actor_timespans:
                    for tsr_values in timespan.values():
                        if len(tsr_values) == 2:
                            actor_spans.append(tsr_values)
                        else:
                            raise Exception(
                                f"Actor timespan must be of the form [start, end]. Found {len(tsr_values)} values.")

                # Add actor to the list
                actors[str(actor_id)] = actor_spans

            # Create the result dictionary
            result_dict = {
                "name": activity_name,
                "timespan": timespan_values,
                "actors": actors
            }

            result.append(result_dict)

        return result

    def import_yaml(self, yml_filepath: str) -> None:
        # Load types
        yml_types_filepath = yml_filepath.replace(".geom.yml", ".types.yml")
        types_dict = self.load_types_from_yml(yml_types_filepath)

        # Load activities
        yml_activities_filepath = yml_filepath.replace(".geom.yml", ".activities.yml")
        activities_dict = self.load_activities_from_yml(yml_activities_filepath)

        # Load frame-wise objects
        objects_dict, meta = self.load_objects_from_yml(yml_filepath, types_dict)

        self._annotations_dict = {"activities": activities_dict, "objects": objects_dict}

        if meta and len(meta) > 0:
            self._annotations_dict["meta"] = meta

    @staticmethod
    def get_act_bboxes(objects: dict, act_timespan: list[int], actor_ids: list) -> list[list[int]]:
        """
        For a given activity timespan, gets the object ID and bounding box for each frame.
        """
        data = []
        for frame_id in range(act_timespan[0], act_timespan[1]):
            frame_objects = objects.get(str(frame_id), [])

            # Filter objects in frame_objects that match actor_ids
            frame_objects = [obj for obj in frame_objects if str(obj["id1"]) in actor_ids]

            if len(frame_objects) > 0:
                data.append({f"{frame_id}": frame_objects})

        return data

    @staticmethod
    def get_overlapping_acts(main_act: dict, target_acts: list[str]) -> list[dict]:
        main_timespan = main_act["timespan"]
        act_name = main_act["name"]
        main_start = main_timespan[0]
        main_end = main_timespan[1]

        result = []
        for act in target_acts:
            if act["name"] == act_name:
                continue

            act_timespan = act["timespan"]
            # If act_timespan and main_timespan strictly intersects, add to result
            intersected = max(main_start, act_timespan[0]) < min(main_end, act_timespan[1])
            if intersected:
                result.append(act["name"])

        return result

    @staticmethod
    def get_union_bbox(bboxes: list[list[int]]):
        x1_min = float('inf')
        y1_min = float('inf')
        x2_max = float('-inf')
        y2_max = float('-inf')

        for frame_dict in bboxes:  # For each frame
            for bbox_list in frame_dict.values():  # Extract the only object in this dict
                for obj in bbox_list:  # For each object
                    x1, y1, x2, y2 = obj["bbox"]
                    x1_min = min(x1_min, x1)
                    y1_min = min(y1_min, y1)
                    x2_max = max(x2_max, x2)
                    y2_max = max(y2_max, y2)

        # Resulting union bbox
        return [x1_min, y1_min, x2_max, y2_max]

    def extract_structured_activities(self, target_acts: set[MevaActivityType]) -> list[dict]:
        """Extract activity data matching specified activity types.
        Args:
            target_acts (set[MevaActivityType]): Only activities whose name maps to one of these types will be included.
        Returns:
            list[dict]: List of dictionaries, each representing a filtered activity with keys
        """
        acts = self["activities"]
        video_act_data = []
        for act in acts:
            if MevaActivityType.from_string(act["name"]) not in target_acts:
                continue

            act_bboxes = self.get_act_bboxes(self["objects"], act["timespan"], list(act["actors"].keys()))
            union_bbox = self.get_union_bbox(act_bboxes)
            overlapping_acts = self.get_overlapping_acts(act, acts)
            act_dict = {
                "name": act["name"],
                "timespan": act["timespan"],
                "bbox": union_bbox,
                "act_bboxes": act_bboxes}

            if len(overlapping_acts) > 0:
                act_dict["overlapping_acts"] = overlapping_acts

            video_act_data.append(act_dict)

        return video_act_data

    @staticmethod
    def augment_activities(video_act_data: list[dict], target_acts: set[MevaActivityType]) -> list[dict]:
        """Generate new augmented activity items by replacing certain activities with their antonyms.        
        """
        act_types_to_argument = [MevaActivityMapping.antonym(act) for act in target_acts if MevaActivityMapping.is_derived(act)]

        aug_video_act_data = []
        for act in video_act_data:
            act_type = MevaActivityType.from_string(act["name"])
            if act_type in act_types_to_argument:
                augmented_act = {
                    "name": MevaActivityMapping.antonym(act_type).value,
                    "timespan": list(reversed(act["timespan"])),
                    "bbox": act["bbox"],
                    "act_bboxes": list(reversed(act["act_bboxes"])),
                    "source": "augmented"
                }
                aug_video_act_data.append(augmented_act)

        return aug_video_act_data

    def write_to_json(self, json_filepath: str) -> None:
        with open(json_filepath, "w") as f:
            json.dump(self.to_dict(), f, cls=MevaAnnotationEncoder, indent=4)

    @classmethod
    def read_from_json(cls, json_filepath: str) -> 'MevaVideoAnnotation':
        with open(json_filepath, "r") as f:
            data = json.load(f, object_hook=MevaAnnotationDecoder)

        # Ensure we got a MevaVideoAnnotations instance
        if isinstance(data, cls):
            return data
        else:
            raise ValueError(f"Expected MevaVideoAnnotations instance, got {type(data)}")
