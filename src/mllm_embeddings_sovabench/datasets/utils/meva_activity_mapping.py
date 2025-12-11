from typing import Iterable
from enum import Enum
from types import MappingProxyType

activity_mapping_virat_to_meva = MappingProxyType({
    # Vehicle activities
    "vehicle_starting": "vehicle_starts",
    "vehicle_stopping": "vehicle_stops",
    "vehicle_turning_left": "vehicle_turns_left",
    "vehicle_turning_right": "vehicle_turns_right",

    # Human/vehicle interactions
    "Entering": "person_enters_vehicle",
    "Exiting": "person_exits_vehicle",
    "Loading": "person_loads_vehicle",
    "Unloading": "person_unloads_vehicle",
    "DropOff_Person_Vehicle": "vehicle_drops_off_person",
    "PickUp_Person_Vehicle": "vehicle_picks_up_person",
    "Closing_Trunk": "person_closes_trunk",
    "Open_Trunk": "person_opens_trunk",
    "Closing": "person_closes_vehicle_door",      # assuming "Closing" refers to vehicle door in context
    "Opening": "person_opens_vehicle_door",      # assuming "Opening" refers to vehicle door in context
    "Riding": "person_rides_bicycle",

    # Human activities
    "Transport_HeavyCarry": "person_carries_heavy_object",
    "Object_Transfer": "person_transfers_object"
})


class MevaActivityType(Enum):
    """
    MEVA Activity definitions — clean enum with no logic.
    """
    # Vehicle Activities
    vehicle_starts = "vehicle_starts"
    vehicle_stops = "vehicle_stops"
    vehicle_reverses = "vehicle_reverses"
    vehicle_turns_left = "vehicle_turns_left"
    vehicle_turns_right = "vehicle_turns_right"

    # Human-Vehicle Interaction
    person_enters_vehicle = "person_enters_vehicle"
    person_exits_vehicle = "person_exits_vehicle"
    person_loads_vehicle = "person_loads_vehicle"
    person_unloads_vehicle = "person_unloads_vehicle"
    vehicle_drops_off_person = "vehicle_drops_off_person"
    vehicle_picks_up_person = "vehicle_picks_up_person"
    person_closes_trunk = "person_closes_trunk"
    person_opens_trunk = "person_opens_trunk"
    person_closes_vehicle_door = "person_closes_vehicle_door"
    person_opens_vehicle_door = "person_opens_vehicle_door"

    # Augmented Activity (derived)
    vehicle_drives_forward = "vehicle_drives_forward"

    # Ignore List
    hand_interacts_with_person = "hand_interacts_with_person"
    person_closes_facility_door = "person_closes_facility_door"
    person_opens_facility_door = "person_opens_facility_door"
    person_picks_up_object = "person_picks_up_object"
    person_puts_down_object = "person_puts_down_object"
    person_transfers_object = "person_transfers_object"
    person_talks_to_person = "person_talks_to_person"
    person_embraces_person = "person_embraces_person"
    person_reads_document = "person_reads_document"
    person_talks_on_phone = "person_talks_on_phone"
    person_texts_on_phone = "person_texts_on_phone"
    person_interacts_with_laptop = "person_interacts_with_laptop"
    person_purchases = "person_purchases"
    vehicle_makes_u_turn = "vehicle_makes_u_turn"

    # Misc Activities
    person_abandons_package = "person_abandons_package"
    person_carries_heavy_object = "person_carries_heavy_object"
    person_steals_object = "person_steals_object"
    person_sits_down = "person_sits_down"
    person_stands_up = "person_stands_up"
    person_enters_scene_through_structure = "person_enters_scene_through_structure"
    person_exits_scene_through_structure = "person_exits_scene_through_structure"
    person_rides_bicycle = "person_rides_bicycle"

    # VIRAT specific
    specialized_throwing = "specialized_throwing"
    activity_running = "activity_running"
    activity_sitting = "activity_sitting"
    Interacts = "Interacts"
    specialized_using_tool = 'specialized_using_tool'
    specialized_talking_phone = 'specialized_talking_phone'

    @classmethod
    def from_string(cls, value, default=None):
        """Safely convert string to enum member."""
        try:
            return cls(value)
        except ValueError:
            return default


class MevaActivityMapping:
    """
    Helper class to provide group, antonym, and label mappings for MevaActivityType.
    """

    _group_map = {
        # Human-Vehicle
        MevaActivityType.person_enters_vehicle: "human_vehicle",
        MevaActivityType.person_exits_vehicle: "human_vehicle",
        MevaActivityType.person_loads_vehicle: "human_vehicle",
        MevaActivityType.person_unloads_vehicle: "human_vehicle",
        MevaActivityType.person_closes_trunk: "human_vehicle",
        MevaActivityType.person_opens_trunk: "human_vehicle",
        MevaActivityType.person_closes_vehicle_door: "human_vehicle",
        MevaActivityType.person_opens_vehicle_door: "human_vehicle",

        # Vehicle
        MevaActivityType.vehicle_starts: "vehicle",
        MevaActivityType.vehicle_stops: "vehicle",
        MevaActivityType.vehicle_reverses: "vehicle",
        MevaActivityType.vehicle_turns_left: "vehicle",
        MevaActivityType.vehicle_turns_right: "vehicle",

        # Augmented
        MevaActivityType.vehicle_drives_forward: "vehicle_augmented",

        # Ignore
        MevaActivityType.hand_interacts_with_person: "ignore",
        MevaActivityType.person_closes_facility_door: "ignore",
        MevaActivityType.person_opens_facility_door: "ignore",
        MevaActivityType.person_picks_up_object: "ignore",
        MevaActivityType.person_puts_down_object: "ignore",
        MevaActivityType.person_transfers_object: "ignore",
        MevaActivityType.person_talks_to_person: "ignore",
        MevaActivityType.person_embraces_person: "ignore",
        MevaActivityType.person_reads_document: "ignore",
        MevaActivityType.person_talks_on_phone: "ignore",
        MevaActivityType.person_texts_on_phone: "ignore",
        MevaActivityType.person_interacts_with_laptop: "ignore",
        MevaActivityType.person_purchases: "ignore",
        MevaActivityType.vehicle_makes_u_turn: "ignore",

        # Other
        MevaActivityType.person_abandons_package: "misc",
        MevaActivityType.person_carries_heavy_object: "misc",
        MevaActivityType.person_steals_object: "misc",
        MevaActivityType.person_sits_down: "misc",
        MevaActivityType.person_stands_up: "misc",
        MevaActivityType.person_enters_scene_through_structure: "misc",
        MevaActivityType.person_exits_scene_through_structure: "misc",
        MevaActivityType.vehicle_drops_off_person: "misc",
        MevaActivityType.vehicle_picks_up_person: "misc",
        MevaActivityType.person_rides_bicycle: "misc"
    }

    _antonym_map = {
        MevaActivityType.person_enters_vehicle: MevaActivityType.person_exits_vehicle,
        MevaActivityType.person_exits_vehicle: MevaActivityType.person_enters_vehicle,
        MevaActivityType.person_loads_vehicle: MevaActivityType.person_unloads_vehicle,
        MevaActivityType.person_unloads_vehicle: MevaActivityType.person_loads_vehicle,
        MevaActivityType.vehicle_drops_off_person: MevaActivityType.vehicle_picks_up_person,
        MevaActivityType.vehicle_picks_up_person: MevaActivityType.vehicle_drops_off_person,
        MevaActivityType.person_closes_trunk: MevaActivityType.person_opens_trunk,
        MevaActivityType.person_opens_trunk: MevaActivityType.person_closes_trunk,
        MevaActivityType.person_closes_vehicle_door: MevaActivityType.person_opens_vehicle_door,
        MevaActivityType.person_opens_vehicle_door: MevaActivityType.person_closes_vehicle_door,
        MevaActivityType.vehicle_starts: MevaActivityType.vehicle_stops,
        MevaActivityType.vehicle_stops: MevaActivityType.vehicle_starts,
        MevaActivityType.vehicle_reverses: MevaActivityType.vehicle_drives_forward,
        MevaActivityType.vehicle_drives_forward: MevaActivityType.vehicle_reverses,
        MevaActivityType.vehicle_turns_left: MevaActivityType.vehicle_turns_right,
        MevaActivityType.vehicle_turns_right: MevaActivityType.vehicle_turns_left,
        MevaActivityType.person_sits_down: MevaActivityType.person_stands_up,
        MevaActivityType.person_stands_up: MevaActivityType.person_sits_down,
    }

    _action_label_map = {
        MevaActivityType.vehicle_starts: "starting",
        MevaActivityType.vehicle_stops: "stopping",
        MevaActivityType.vehicle_turns_left: "turning left",
        MevaActivityType.vehicle_turns_right: "turning right",
        MevaActivityType.vehicle_reverses: "reversing",
        MevaActivityType.vehicle_drives_forward: "driving forward",
        MevaActivityType.person_enters_vehicle: "entering vehicle",
        MevaActivityType.person_exits_vehicle: "exiting vehicle",
        MevaActivityType.person_loads_vehicle: "loading vehicle",
        MevaActivityType.person_unloads_vehicle: "unloading vehicle",
        MevaActivityType.person_closes_trunk: "closing trunk",
        MevaActivityType.person_opens_trunk: "opening trunk",
        MevaActivityType.person_closes_vehicle_door: "closing vehicle door",
        MevaActivityType.person_opens_vehicle_door: "opening vehicle door",
        MevaActivityType.person_rides_bicycle: "riding bicycle",
    }

    @staticmethod
    def group(activity: MevaActivityType) -> str:
        return MevaActivityMapping._group_map.get(activity, "unknown")

    @staticmethod
    def antonym(activity: MevaActivityType) -> MevaActivityType | None:
        return MevaActivityMapping._antonym_map.get(activity)

    @staticmethod
    def action_label(activity: MevaActivityType) -> str:
        return MevaActivityMapping._action_label_map.get(activity, activity.value.replace("_", " "))

    @staticmethod
    def is_ignored(activity: MevaActivityType) -> bool:
        return MevaActivityMapping.group(activity) == "ignore"

    @staticmethod
    def is_derived(activity: MevaActivityType) -> bool:
        return activity == MevaActivityType.vehicle_drives_forward

    @staticmethod
    def by_group(group_name: str) -> set[MevaActivityType]:
        return {act for act in MevaActivityType if MevaActivityMapping.group(act) == group_name}

    @staticmethod
    def by_groups(group_names: Iterable[str]) -> set[MevaActivityType]:
        return {act for act in MevaActivityType if MevaActivityMapping.group(act) in group_names}

    @staticmethod
    def all_activities() -> set[MevaActivityType]:
        return set(MevaActivityType)

    @staticmethod
    def subject_name(activity: MevaActivityType) -> str | None:
        return None if MevaActivityMapping.is_ignored(activity) else activity.value.split("_")[0]
