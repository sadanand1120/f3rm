# Object Detection Task

Your task: Analyze the provided image and identify ALL objects visible in it.

IMPORTANT INSTRUCTIONS:
1. Focus ONLY on actual physical objects (furniture, items, etc.)
2. EXCLUDE background elements and terrains (floor, wall, ceiling, sky, ground, carpet, terrain, etc.)
3. Be comprehensive - include every object you can identify, even if partially visible (e.g., if a book is partially visible, include it with appropriate visibility_percent)
4. Use your best judgment for visibility assessment
5. If multiple instances of the same object type exist, list each one separately (e.g., if there are 2 books, list "book" twice with individually different/specific details)

FIELD DEFINITIONS:
- object_name: Simple, common name (1-2 words max, no underscores)
- visibility_percent: Integer 0-100 - estimate what percentage of the WHOLE object (including its parts not in the image) is visible in this image view. Consider the object's complete size and intelligently estimate how much of it appears in the current frame (0=not visible, 100=completely visible)
- details: Detailed physical description including color, material, size, brand, texture, and any distinctive features
- notes: Verbose observations about position, visibility, condition, context, lighting, angle, occlusion, and any other relevant details

CONSTRAINTS:
- Do not include any text explanations outside the JSON
- Do not add extra fields
- If no objects are visible, return {"objects": []}
