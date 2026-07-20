from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CourseMetadataGenerationRequest:
    tenant_id: str
    course_title: str
    existing_categories: List[str] = field(default_factory=list)
    existing_course_codes: List[str] = field(default_factory=list)


@dataclass
class CourseMetadataResult:
    course_code: str
    short_description: str
    difficulty: str
    category: str
    learning_objectives: List[str]
    skills_covered: List[str]
    estimated_duration_hours: float
    recommended_prerequisites: str
    suggested_tags: List[str]


@dataclass
class CourseModulesGenerationRequest:
    tenant_id: str
    course_title: str
    short_description: str = ""
    category: str = ""
    difficulty: str = ""
    learning_objectives: List[str] = field(default_factory=list)
    skills_covered: List[str] = field(default_factory=list)


@dataclass
class GeneratedLesson:
    title: str
    content_type: str
    duration_minutes: Optional[float] = None


@dataclass
class GeneratedModule:
    title: str
    description: str
    lessons: List[GeneratedLesson]


@dataclass
class CourseModulesResult:
    modules: List[GeneratedModule]
