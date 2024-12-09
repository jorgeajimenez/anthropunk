import kagglehub

# Download latest version
path = kagglehub.dataset_download("aleespinosa/apple-watch-and-fitbit-data")

print("Path to dataset files:", path)
