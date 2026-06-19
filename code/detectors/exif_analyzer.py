from PIL import Image
from PIL.ExifTags import TAGS

MANIPULATION_SOFTWARE = {"photoshop", "gimp", "lightroom", "paint", "editor", "canva", "picasa"}


def analyze_exif(image_path: str) -> dict:
    result = {
        "has_exif": False,
        "is_edited": False,
        "software": "",
        "datetime_original": "",
        "camera_model": "",
    }
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
    except Exception:
        return result

    if not exif_data:
        return result

    result["has_exif"] = True
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, "")
        if tag_name == "Software":
            result["software"] = str(value)
            result["is_edited"] = any(
                w in str(value).lower() for w in MANIPULATION_SOFTWARE
            )
        elif tag_name == "DateTimeOriginal":
            result["datetime_original"] = str(value)
        elif tag_name == "Model":
            result["camera_model"] = str(value)

    return result
