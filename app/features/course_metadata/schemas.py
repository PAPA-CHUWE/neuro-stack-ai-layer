from pydantic import BaseModel


class GenerateCourseMetadataBody(BaseModel):
    tenantId: str
    courseTitle: str


class CourseMetadataResponse(BaseModel):
    courseCode: str
    shortDescription: str
    difficulty: str
    category: str
    learningObjectives: list[str]
    skillsCovered: list[str]
    estimatedDurationHours: float
    recommendedPrerequisites: str
    suggestedTags: list[str]
    generatedAt: str
    provider: str


class ErrorResponse(BaseModel):
    error: str
    message: str


class GenerateCourseModulesBody(BaseModel):
    tenantId: str
    courseTitle: str
    shortDescription: str = ""
    category: str = ""
    difficulty: str = ""
    learningObjectives: list[str] = []
    skillsCovered: list[str] = []


class LessonItem(BaseModel):
    title: str
    contentType: str
    durationMinutes: float | None = None


class ModuleItem(BaseModel):
    title: str
    description: str
    lessons: list[LessonItem]


class CourseModulesResponse(BaseModel):
    modules: list[ModuleItem]
    generatedAt: str
    provider: str
