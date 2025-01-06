import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from fair.agent import BaseAgent, LegacyStudent
from fair.constraint import (
    CourseTimeConstraint,
    LinearConstraint,
    MutualExclusivityConstraint,
    PreferenceConstraint,
)
from fair.feature import Course, Section, Slot, Weekday, slots_for_time_range
from fair.item import ScheduleItem
from fair.valuation import ConstraintSatifactionValuation
from scipy.stats import truncnorm

from qsurvey import parser

DEFAULT_CAPACITY = 30
STATUS_LABEL_MAP = {
    1: "Fresh",
    2: "Soph",
    3: "Jun",
    4: "Sen",
    5: "MS",
    6: "MS/PhD",
}


def get_status_relevant(status, all_courses, course_map, status_crs_prefix_map):
    relevant_courses = [
        course
        for course in all_courses
        if course_map[course]["course num"][0] in status_crs_prefix_map[status]
    ]
    relevant_idxs = [all_courses.index(course) for course in relevant_courses]

    return relevant_idxs


def scale_up_responses(responses, relevant_idxs, n):
    new_responses = np.zeros((responses.shape[0], n))
    new_responses[:, relevant_idxs] = responses

    return new_responses


def top_preferred(course_map, schedule, course, response, pref_thresh):
    all_courses = list(course_map.values())
    order = np.argsort(response)[::-1]

    top_course_nums = set()
    idxs = []

    for idx in order:
        if len(top_course_nums) >= pref_thresh or response[idx] == 1:
            break

        if all_courses[idx]["course num"] not in top_course_nums:
            same_value_indices = [i for i in order if response[i] == response[idx]]
            idxs.extend(same_value_indices)
            top_course_nums.update(
                all_courses[index]["course num"] for index in same_value_indices
            )

    preferred_courses = [schedule[j] for j in idxs]
    return preferred_courses


def synthesize_students(
    num_samples,
    course,
    section,
    features,
    schedule,
    qs,
    surveys,
    distribution,
    course_map,
    max_courses,
    relevant_idxs,
    rng,
    pref_thresh,
):
    num_students = len(surveys)
    total_course_list = [int(survey.data().sum()) for survey in surveys]
    data = np.vstack([survey.data() for survey in surveys])
    data = scale_up_responses(data, relevant_idxs, len(course_map))
    # denormalize data
    data = 7 * data + 1
    # generate synthetic samples
    synth_data = []
    while len(synth_data) < num_samples:
        sdata = distribution.sample()
        sdata = scale_up_responses(sdata, relevant_idxs, len(course_map))
        # denormalize synthetic data
        sdata = np.round(7 * sdata) + 1
        if sdata.max() == 1:
            continue
        synth_data.append(sdata)
    synth_data = np.vstack(synth_data)
    data = np.vstack([data, synth_data])
    students = SurveyStudent.from_responses(
        data[num_students:],
        total_course_list,
        course,
        section,
        course_map,
        [
            qs.course_time_constr(features, schedule),
            qs.course_sect_constr(features, schedule),
        ],
        schedule,
        rng=rng,
        pref_thresh=pref_thresh,
        max_total_courses=max_courses,
    )

    return students, data


class SurveyStudent(BaseAgent):
    """A manifestation of BaseAgent according to survey responses from a student"""

    @staticmethod
    def from_responses(
        responses: np.ndarray,
        total_course_list: list[int],
        course: Course,
        section: Section,
        course_map,
        global_constraints: list[LinearConstraint],
        schedule: list[ScheduleItem],
        rng: np.random.Generator,
        pref_thresh: int,
        max_total_courses: int = sys.maxsize,
        sparse: bool = False,
        memoize: bool = True,
    ):
        """Create list of SurveyStudents from response vector and schedule of same dimension

        The total courses assigned to a student are set by fitting a truncated normal distribution
        to the total_course_list provided. The total courses set for each student will range between
        1 and max_total_courses unless max_total_courses is set to sys.maxsize in which case the
        distribution is still based on total_course_list, but it is not truncated on the upper tail.

        Args:
            responses (np.ndarray): Survey responses
            total_course_list (list[int]): The total courses desired by each of the students in responses.
            course (Course): Course feature
            global_constraints (list[LinearConstraint]): Previously constructed global constraints
            schedule (list[ScheduleItem]): A list of items corresponding to responses
            rng (np.random.Generator): Random number generator
            threshold (int, optional): What response value constitutes a preference for the item. Defaults to 1.
            max_total_courses (int, optional): Total courses that can be assigned to the student. Defaults to sys.maxsize.
            sparse (bool, optional): Should sparse matrices be used for constraints. Defaults to False.
            memoize (bool, optional): Should results be cached. Defaults to True.

        Returns:
            list[SurveyStudent]: A list of SurveyStudents constructed from responses
        """
        total_course_list = [
            max(1, min(max_total_courses, tot)) for tot in total_course_list
        ]
        params = truncnorm.fit(total_course_list)

        students = []
        for i in range(responses.shape[0]):
            preferred_courses = top_preferred(
                course_map, schedule, course, responses[i], pref_thresh
            )
            total_courses = int(
                min(max_total_courses, truncnorm.rvs(*params, random_state=rng))
            )
            students.append(
                SurveyStudent(
                    preferred_courses,
                    total_courses,
                    course,
                    section,
                    global_constraints,
                    schedule,
                    sparse,
                    memoize,
                )
            )

        return students

    def __init__(
        self,
        preferred_courses: list[ScheduleItem],
        total_courses: int,
        course: Course,
        section: Section,
        global_constraints: list[LinearConstraint],
        schedule: list[ScheduleItem],
        sparse: bool = False,
        memoize: bool = True,
    ):
        """
        Args:
            preferred_courses (list[ScheduleItem]): A list of course items preferred by the student
            total_courses (int): The maximum number of courses the student wishes to take
            course (Course): Feature for course
            global_constraints (List[LinearConstraint]): Constraints not specific to this agent
            schedule (List[ScheduleItem], optional): All possible items in the student's schedule. Defaults to None.
            sparse (bool, optional): Should sparse matrices be used for constraints. Defaults to False.
            memoize (bool, optional): Should results be cached. Defaults to True
        """
        self.preferred_courses = preferred_courses
        self.total_courses = total_courses
        self.quantities = [total_courses]
        self.preferred_topics = [preferred_courses]

        all_courses = [(item.value(course), item.value(section)) for item in schedule]
        self.all_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [all_courses],
            [self.total_courses],
            [course, section],
            sparse,
        )

        undesirable_courses = [
            (item.value(course), item.value(section))
            for item in schedule
            if item not in self.preferred_courses
        ]
        self.undesirable_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [undesirable_courses],
            [0],
            [course, section],
            sparse,
        )

        preferred_values = [
            (item.value(course), item.value(section)) for item in self.preferred_courses
        ]
        self.preferred_courses_constraint = PreferenceConstraint.from_item_lists(
            schedule,
            [preferred_values],
            [self.total_courses],
            [course, section],
            sparse,
        )

        constraints = global_constraints + [
            self.undesirable_courses_constraint,
            self.preferred_courses_constraint,
        ]

        super().__init__(ConstraintSatifactionValuation(constraints, memoize))


class QSurvey:

    def __init__(self, in_file, mp, included_courses=None):
        df = pd.read_csv(in_file)
        self.questions = ["1", "2", "3", "4"] + [f"5#1_{i}" for i in range(1, 12)]
        self.cics_courses = [col for col in df.columns if re.match("7 _\d+$", col)]
        self.compsci_courses = [col for col in df.columns if re.match("7_\d+$", col)]
        self.info_courses = [col for col in df.columns if re.match("7 _\d\.", col)]
        self.all_courses = self.cics_courses + self.compsci_courses + self.info_courses
        course_map = mp.mapping(self.all_courses)
        if included_courses is None:
            included_courses = self.all_courses
        else:
            included_courses = [
                crs
                for crs in self.all_courses
                if crs in course_map
                and course_map[crs]["course num"] in included_courses
            ]
        self.all_courses = [crs for crs in self.all_courses if crs in included_courses]
        self.df = df[self.questions + self.all_courses]

    def course_time_constr(self, features, schedule, sparse=False):
        _, slot, weekday, _ = features

        return CourseTimeConstraint.from_items(schedule, slot, weekday, sparse)

    def course_sect_constr(self, features, schedule, sparse=False):
        course, _, _, _ = features

        return MutualExclusivityConstraint.from_items(schedule, course, sparse)

    def students(
        self,
        course_map,
        all_courses,
        features,
        schedule,
        pref_thresh,
        sparse=False,
    ):
        course, _, _, section = features

        students = []
        responses = []
        statuses = []
        for _, row in self.df.iterrows():
            response = [row[crs] if row[crs] > 0 else 1 for crs in all_courses]
            preferred = top_preferred(
                course_map, schedule, course, response, pref_thresh
            )
            total_num_courses = row["3"]

            if np.isnan(total_num_courses):
                warnings.warn("total courses not specified; skipping student")
                continue

            responses.append([row[crs] for crs in all_courses])
            statuses.append(row["1"])
            student = SurveyStudent(
                preferred,
                total_num_courses,
                course,
                section,
                [
                    self.course_time_constr(features, schedule, sparse),
                    self.course_sect_constr(features, schedule, sparse),
                ],
                schedule,
                sparse=sparse,
            )
            legacy_student = LegacyStudent(student, student.preferred_courses, course)
            legacy_student.student.valuation.valuation = (
                legacy_student.student.valuation.compile()
            )
            students.append(legacy_student)

        return students, np.nan_to_num(responses, nan=1.0), statuses


class QSchedule:

    def __init__(self, in_file):
        self.df = pd.read_excel(in_file)

    def capacities(self):
        crs_sec_cap_map = defaultdict(dict)
        for _, row in self.df.iterrows():
            crs_sec_cap_map[str(row["Catalog"])][row["Section"]] = row["Enrl Capacity"]

        return crs_sec_cap_map


class QMapper:

    def __init__(self, in_file):
        self.df = pd.read_csv(in_file, sep="|")

    def desc_for_ques(self, ques):
        return self.df[self.df.question == ques].description.values[0]

    def mapping(self, questions):
        # construct course information map
        course_map = {}
        for crs in questions:
            raw_description = self.desc_for_ques(crs)
            catalog, course_num, section, description = parser.extract_course_info(
                raw_description
            )
            instructor = parser.extract_instructor_info(raw_description)
            days, start_time, end_time = parser.extract_schedule_info(raw_description)
            try:
                course_map[crs] = {
                    "catalog": catalog,
                    "course num": course_num,
                    "section": section,
                    "description": description,
                    "instructor": instructor,
                    "days": days,
                    "time range": start_time.strftime("%I:%M %p")
                    + " - "
                    + end_time.strftime("%I:%M %p"),
                }
            except AttributeError:
                continue

        return course_map

    @staticmethod
    def features(course_map):
        # construct features
        course = Course([entry["course num"] for entry in course_map.values()])
        slot = Slot.from_time_ranges(
            [entry["time range"] for entry in course_map.values()], "15T"
        )
        weekday = Weekday()
        section = Section([entry["section"] for entry in course_map.values()])
        features = [course, slot, weekday, section]

        return features

    @staticmethod
    def schedule(course_map, crs_sec_cap_map, features, drop_on_warning=True):
        # construct schedule
        schedule = []
        days = Weekday().days
        for idx, (crs, map) in enumerate(course_map.items()):
            crs = str(map["course num"])
            slt = slots_for_time_range(map["time range"], features[1].times)
            sec = map["section"]
            capacity = DEFAULT_CAPACITY
            if crs not in crs_sec_cap_map:
                warnings.warn(f"No capacity information for course {crs}")
                if drop_on_warning:
                    continue
            elif int(sec) not in crs_sec_cap_map[crs]:
                warnings.warn(
                    f"No capacity information for course {crs} and section {sec}"
                )
                if drop_on_warning:
                    continue
            else:
                capacity = crs_sec_cap_map[crs][int(sec)]
            dys = tuple([day for day in days if day in map["days"]])
            schedule.append(
                ScheduleItem(
                    features, [crs, slt, dys, sec], index=idx, capacity=capacity
                )
            )

        return schedule
