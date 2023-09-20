from test import FaceMatcher


media_folder_path = "./media"
matcher = FaceMatcher(media_folder_path)
img_path = "Sweety.jpg"
processing_time, similarity, name = matcher.match_face(img_path)
print(f"Processing Time: {processing_time} seconds")
print(f"Similarity: {similarity} %")
print(f"Identity: {name}")