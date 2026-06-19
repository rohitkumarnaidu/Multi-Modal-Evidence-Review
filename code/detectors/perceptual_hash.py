from PIL import Image
import imagehash


def compute_phash(image_path: str) -> str | None:
    try:
        return str(imagehash.phash(Image.open(image_path)))
    except Exception:
        return None


def are_images_similar(hash1: str, hash2: str, threshold: int = 10) -> bool:
    try:
        return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) < threshold
    except Exception:
        return False


def find_duplicates(image_paths: list[str]) -> list[tuple[int, int, int]]:
    hashes = []
    for p in image_paths:
        h = compute_phash(p)
        hashes.append(h)
    duplicates = []
    for i in range(len(hashes)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j] is None:
                continue
            dist = imagehash.hex_to_hash(hashes[i]) - imagehash.hex_to_hash(hashes[j])
            if dist < 10:
                duplicates.append((i, j, dist))
    return duplicates


def max_phash_distance(image_paths: list[str]) -> int | None:
    hashes = []
    for p in image_paths:
        h = compute_phash(p)
        if h is not None:
            hashes.append(imagehash.hex_to_hash(h))
    if len(hashes) < 2:
        return None
    max_dist = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hashes[i] - hashes[j]
            if dist > max_dist:
                max_dist = dist
    return max_dist
