# Object Inclusion Evaluation Task

Your task: Evaluate whether this specific object should be included in the final object list.

EVALUATION CRITERIA:
- INCLUDE if: Object is sufficiently visible (e.g., consider the context, though a general rule of thumb is visibility_percent at least around 60%) and identifiable
- EXCLUDE if: Object is too blurry, too distant, too cropped out, or is background/terrain (floor, wall, ceiling, sky, ground, carpet etc.)

FIELD DEFINITIONS:
- include: Boolean - true if object meets inclusion criteria
- final_name: String (1-2 words) if included, null if excluded
- reasoning: Succinct explanation of your decision

CONSTRAINTS:
- Do not include any text outside the JSON structure
- Base decision on the provided object details only

OBJECT TO EVALUATE:
{object_details}
