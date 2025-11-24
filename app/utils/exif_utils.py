import exifread

def check_exif(path):
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f)

        if not tags:
            return {"suspicious": True, "reason": "EXIF missing", "tags": {}}

        return {"suspicious": False, "tags": {k: str(v) for k, v in tags.items()}}
    except:
        return {"suspicious": True, "reason": "EXIF error", "tags": {}}
